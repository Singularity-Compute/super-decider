#!/bin/bash
# Container entrypoint: serve the weights in MODEL_DIR (downloaded from MODEL_ID if missing), start vLLM on
# 127.0.0.1:8000 and the structured server on 0.0.0.0:8011. Exits if either process dies, so an orchestrator can
# restart it. Readiness: /tmp/djev-ready appears once the structured server answers (and, with WARMUP=1, after the
# warm-up requests). Optional vLLM patches, each applied before vllm serve and skipped when already applied:
#   DJEV_BIDIR_SERVE=1  per-question bidirectional prompt blocks (the model was trained this way; pair with DJEV_BIDIR=qsep)
#   DJEV_ATTN_PATCH=1   faster Triton unified attention on Ada (sm89) for those blocks
#   DJEV_THINK_PATCH=1  faster thinking path (DJEV_FP8_LMHEAD=1, DJEV_FUSED_SAMPLER=1)
set -euo pipefail

MODEL_ID=${MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
MODEL_DIR=${MODEL_DIR:-/data/models/$(basename "$MODEL_ID")}
CANVAS=${CANVAS:-128}
MAX_SEQS=${MAX_SEQS:-64}
MAX_LEN=${MAX_LEN:-16384}
KV_BYTES=${KV_BYTES:-17179869184}
KV_DTYPE=${KV_DTYPE:-fp8}  # FP8 KV: same accuracy as BF16, 22-27% more throughput
API_SERVERS=${API_SERVERS:-1}  # vLLM frontend processes; one frontend caps the engine at ~40 requests/s
FP8_DENSE=${FP8_DENSE:-0}  # 1 = quantize the BF16 attention + dense-MLP layers to FP8 at load (experts stay NVFP4): +5%, quality unchanged
WARMUP=${WARMUP:-0}  # 1 = run /cfg/warmup.py before readiness (cold first requests 8-30 s -> <1 s)
export HF_HOME=${HF_HOME:-/data/hf}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data/vllm-cache}  # torch.compile cache survives restarts
export VLLM_NO_USAGE_STATS=1
rm -f /tmp/djev-ready

echo "vllm $(python3 -c 'import vllm;print(vllm.__version__)')"
python3 -c 'import vllm.utils.diffusion, vllm.v1.core.sched.diffusion_scheduler; print("structured reads ok")'
if [ "${DJEV_BIDIR_SERVE:-0}" = "1" ]; then
  python3 /cfg/patch_vllm_bidir.py
  python3 -c 'import vllm.v1.core.sched.scheduler, vllm.v1.core.kv_cache_manager, vllm.model_executor.models.diffusion_gemma; print("bidir patch ok")'
fi
# With DJEV_ATTN_PATCH=1 the sm89 Triton unified-attention patch (/cfg/djev_attn_sm89.diff: tile skipping with per-request
# causal flags and djev ranges, range-aware sliding window, V-mask fix for ranges longer than the window, Ada tile config)
# is applied to vLLM before serving. Idempotent: skipped when already applied.
if [ "${DJEV_ATTN_PATCH:-0}" = "1" ]; then
  SP=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
  if patch -p1 -d "$SP" --dry-run -R -s -f < /cfg/djev_attn_sm89.diff > /dev/null 2>&1; then
    echo "attention patch already applied"
  else
    patch -p1 -d "$SP" -s -f < /cfg/djev_attn_sm89.diff || { echo "attention patch FAILED"; exit 1; }
    find "$SP/vllm/v1/attention/ops" -name "*.pyc" -delete
  fi
  python3 -c 'import vllm.v1.attention.ops.triton_unified_attention, vllm.v1.attention.ops.triton_attention_helpers; print("attention patch ok")'
fi
# With DJEV_THINK_PATCH=1 the thinking-path patch (/cfg/djev_think_sampler.diff: DiffusionGemma sampler with optional FP8
# full-vocab logits DJEV_FP8_LMHEAD=1 and a fused sampling kernel DJEV_FUSED_SAMPLER=1; both default off) is applied after the
# bidir patch (it edits the bidir-patched diffusion_gemma.py). Idempotent.
if [ "${DJEV_THINK_PATCH:-0}" = "1" ]; then
  SP=$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
  if patch -p1 -d "$SP" --dry-run -R -s -f < /cfg/djev_think_sampler.diff > /dev/null 2>&1; then
    echo "thinking patch already applied"
  else
    patch -p1 -d "$SP" -s -f < /cfg/djev_think_sampler.diff || { echo "thinking patch FAILED"; exit 1; }
    find "$SP/vllm/model_executor/models" -name "diffusion_gemma*.pyc" -delete
  fi
  python3 -c 'import vllm.model_executor.models.diffusion_gemma, vllm.model_executor.models.djev_kb_kernels; print("thinking patch ok")'
fi

if [ ! -f "$MODEL_DIR/.complete" ]; then
  echo "no complete model at $MODEL_DIR; downloading $MODEL_ID"
  hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
  touch "$MODEL_DIR/.complete"
fi
# Images: VISION=1 loads the vision tower and accepts up to MAX_IMAGES images per request; default = text only.
if [ "${VISION:-0}" = "1" ]; then
  MM_ARGS=(--limit-mm-per-prompt "{\"image\": ${MAX_IMAGES:-8}, \"video\": 0}")
else
  MM_ARGS=(--language-model-only)
fi
QUANT_ARGS=()
if [ "$FP8_DENSE" = "1" ]; then
  QUANT_ARGS=(--quantization-config '{"targets":{"re:^.+\\.self_attn\\.(q_proj|k_proj|v_proj|qkv_proj|o_proj)$":"fp8_per_channel","re:^.+\\.layers\\.\\d+\\.mlp\\.(gate_proj|up_proj|gate_up_proj|down_proj)$":"fp8_per_channel"}}')
fi
echo "serving $MODEL_DIR (max_len $MAX_LEN, kv $KV_DTYPE $KV_BYTES B) commit=${DJEV_NOUL_COMMIT:-0} temps=${DJEV_TEMPS:-} adaptive=${DJEV_ADAPTIVE_THINK:-off} think_steps=${DJEV_THINK_STEPS:-default} doc_first=${DJEV_DOC_FIRST:-0} bidir=${DJEV_BIDIR:-off}/${DJEV_BIDIR_SERVE:-0} vision=${VISION:-0} api_servers=$API_SERVERS fp8_dense=$FP8_DENSE warmup=$WARMUP"

vllm serve "$MODEL_DIR" --served-model-name super-decider --host 127.0.0.1 --port 8000 \
  --diffusion-config "{\"canvas_length\": $CANVAS}" --max-logprobs 128 \
  --enable-prefix-caching --max-model-len "$MAX_LEN" --max-num-seqs "$MAX_SEQS" \
  --async-scheduling --kv-cache-memory-bytes "$KV_BYTES" --kv-cache-dtype "$KV_DTYPE" "${MM_ARGS[@]}" \
  --api-server-count "$API_SERVERS" "${QUANT_ARGS[@]}" \
  ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

until curl -sf 127.0.0.1:8000/v1/models >/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm exited"; exit 1; }
  sleep 2
done
echo "vllm ready"

python3 /cfg/structured_server.py --upstream http://127.0.0.1:8000 --model super-decider \
  --tokenizer "$MODEL_DIR" --canvas "$CANVAS" --host 0.0.0.0 --port 8011 \
  ${SERVER_EXTRA_ARGS:-} &
SS_PID=$!

until curl -sf 127.0.0.1:8011/v1/models >/dev/null; do
  kill -0 $SS_PID 2>/dev/null || { echo "structured server exited"; exit 1; }
  sleep 2
done
if [ "$WARMUP" = "1" ] && [ -f /cfg/warmup.py ]; then
  python3 /cfg/warmup.py 8011 || echo "warm-up failed (serving anyway)"
fi
touch /tmp/djev-ready
echo "ready"

wait -n $VLLM_PID $SS_PID
echo "a server process exited"; exit 1

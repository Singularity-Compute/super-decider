#!/bin/bash
# Serve super-decider without Docker on a machine that has the pinned vLLM build installed
# (pip: the nightly in runtime/Dockerfile). Weights: MODEL_DIR (local dir) or MODEL_ID (downloaded on first start).
# Defaults match runtime/Dockerfile. The DJEV_*_PATCH and DJEV_BIDIR_SERVE switches patch the installed vLLM in place.
set -euo pipefail
cd "$(dirname "$0")"
export MODEL_ID=${MODEL_ID:-Singularity-Compute/super-decider} MODEL_DIR=${MODEL_DIR:-$PWD/models/super-decider}
export DJEV_DOC_FIRST=${DJEV_DOC_FIRST:-1} DJEV_BIDIR_SERVE=${DJEV_BIDIR_SERVE:-1} DJEV_BIDIR=${DJEV_BIDIR:-qsep} DJEV_BIDIR_MAXPROMPT=${DJEV_BIDIR_MAXPROMPT:-8192}
export DJEV_ATTN_PATCH=${DJEV_ATTN_PATCH:-1} DJEV_THINK_PATCH=${DJEV_THINK_PATCH:-1} DJEV_FP8_LMHEAD=${DJEV_FP8_LMHEAD:-1} DJEV_FUSED_SAMPLER=${DJEV_FUSED_SAMPLER:-1}
export DJEV_NOUL_COMMIT=${DJEV_NOUL_COMMIT:-0.8} DJEV_TEMPS=${DJEV_TEMPS:-score=0.375}
export DJEV_ADAPTIVE_THINK=${DJEV_ADAPTIVE_THINK:-choice=0.7,score=0.7,noul=0.85,budget=512,min_tokens=0,max_inflight=2,max_questions=2}
export DJEV_THINK_STEPS=${DJEV_THINK_STEPS:-8} DJEV_MANY_OPTIONS=${DJEV_MANY_OPTIONS:-chunk=20,top=3,think1=0,think2=0}
export DEFAULT_SAMPLES=${DEFAULT_SAMPLES:-1} SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:---constrained}
export MAX_LEN=${MAX_LEN:-131072} API_SERVERS=${API_SERVERS:-8} FP8_DENSE=${FP8_DENSE:-1} WARMUP=${WARMUP:-1}
export VISION=${VISION:-1} MAX_IMAGES=${MAX_IMAGES:-8} API_KEY=${API_KEY:-local}
mkdir -p /cfg 2>/dev/null || true
for f in structured_server.py warmup.py patch_vllm_bidir.py djev_attn_sm89.diff djev_think_sampler.diff; do ln -sf "$PWD/$f" /cfg/$f; done
exec bash ./entrypoint.sh

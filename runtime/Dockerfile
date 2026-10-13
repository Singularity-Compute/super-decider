# super-decider runtime: stock vLLM (pinned nightly with DiffusionGemma structured reads), small vLLM patches for
# per-question bidirectional prompt blocks, faster attention on Ada and a faster thinking path, and the structured server.
# Build:  docker build -t super-decider runtime/
# Run:    docker run --gpus all -p 8011:8011 -v $PWD/models:/data/models -e MODEL_ID=Singularity-Compute/super-decider super-decider
# The server speaks the Jev wire format on :8011 (POST /v1/systemone) and OpenAI chat completions (POST /v1/chat/completions).
FROM vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098
COPY structured_server.py warmup.py entrypoint.sh patch_vllm_bidir.py djev_attn_sm89.diff djev_think_sampler.diff /cfg/
RUN apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*
RUN chmod +x /cfg/entrypoint.sh
# Defaults. Override with -e.
ENV MODEL_ID=Singularity-Compute/super-decider \
    MODEL_DIR=/data/models/super-decider \
    DJEV_DOC_FIRST=1 \
    DJEV_BIDIR_SERVE=1 DJEV_BIDIR=qsep DJEV_BIDIR_MAXPROMPT=8192 \
    DJEV_ATTN_PATCH=1 DJEV_THINK_PATCH=1 DJEV_FP8_LMHEAD=1 DJEV_FUSED_SAMPLER=1 \
    DJEV_NOUL_COMMIT=0.8 \
    DJEV_TEMPS=score=0.375 \
    DJEV_ADAPTIVE_THINK=choice=0.7,score=0.7,noul=0.85,budget=512,min_tokens=0,max_inflight=2,max_questions=2 \
    DJEV_THINK_STEPS=8 \
    DJEV_MANY_OPTIONS=chunk=20,top=3,think1=0,think2=0 \
    DEFAULT_SAMPLES=1 SERVER_EXTRA_ARGS=--constrained \
    MAX_LEN=131072 KV_DTYPE=fp8 KV_BYTES=17179869184 CANVAS=128 MAX_SEQS=64 \
    API_SERVERS=8 FP8_DENSE=1 WARMUP=1 VISION=1 MAX_IMAGES=8 \
    API_KEY=local
EXPOSE 8011
ENTRYPOINT ["/cfg/entrypoint.sh"]

# NOTICE

super-decider. Copyright 2026 Singularity Compute. Licensed under the Apache License, Version 2.0 (see LICENSE).

## Base model
- google/diffusiongemma-26B-A4B-it (Google DeepMind), Apache-2.0 (https://ai.google.dev/gemma/docs/gemma_4_license).
- nvidia/diffusiongemma-26B-A4B-it-NVFP4, NVIDIA's ModelOpt NVFP4 checkpoint of the above, Apache-2.0.

## Runtime
- `runtime/structured_server.py` is derived from mmastrac/djev (https://github.com/mmastrac/djev, Apache-2.0) and
  modified by Singularity Compute.
- vLLM (Apache-2.0), pinned nightly build in `runtime/Dockerfile`.

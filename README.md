# super-decider

super-decider answers typed questions about a state: `choice`, yes/no (`noul`) and `score`, each with a calibrated
probability distribution, in one call. It is a fine-tune of Google's DiffusionGemma 26B-A4B (discrete-diffusion MoE,
4B active parameters) that reads every answer slot in one denoising step and runs on one NVIDIA L40S.

- API: the Jev / TypeSafe "System One" request format (`POST /v1/systemone`) and OpenAI-compatible chat completions;
  text, JSON and up to 8 images per request. Reference: [`docs/API.md`](docs/API.md).
- Runtime: stock vLLM (pinned nightly), small vLLM patches (each question's prompt block read in both directions,
  faster attention on Ada GPUs, a faster thinking path) and a thin structured server, in [`runtime/`](runtime/).
- Weights and model card: [Singularity-Compute/super-decider](https://huggingface.co/Singularity-Compute/super-decider)
  on Hugging Face (NVFP4, 18 GB).
- Licences: [`LICENSE`](LICENSE) (Apache-2.0), [`NOTICE.md`](NOTICE.md).

## How it works
- **Fine-tuned for decisions.** DiffusionGemma 26B-A4B is fine-tuned on permissively licensed and synthetic decision
  data, then merged into NVIDIA's NVFP4 checkpoint.
- **One-step reads.** All answer slots of a request are filled in one denoising step, so several questions cost little
  more than one.
- **Bidirectional attention within each question.** The prompt block of every question is read in both directions, not
  only left to right. The model was trained this way, and the runtime applies the same attention through a small vLLM
  patch (`DJEV_BIDIR_SERVE=1`, `DJEV_BIDIR=qsep`).
- **Adaptive thinking.** When the model is unsure of an answer, it writes a short thought and reads the answer again.
  A request can turn this off, keep it on under load, or think on every question with `"thinking"` (see
  [`docs/API.md`](docs/API.md)).

## Run it
```bash
docker build -t super-decider runtime/
docker run --gpus all -p 8011:8011 -v $PWD/models:/data/models -e MODEL_ID=Singularity-Compute/super-decider super-decider
curl localhost:8011/v1/systemone -H 'Authorization: Bearer local' -H 'Content-Type: application/json' \
  -d '{"state":"Order 8812 shipped Oct 1; the customer wants to cancel.","questions":{"allowed":{"type":"noul","instructions":"Is cancelling allowed?"}}}'
```
The container needs a 48 GB GPU (L40S, A6000, H100) and downloads the weights on first start. Defaults are set in
`runtime/Dockerfile`; override them with `-e`. `runtime/run.sh` runs the same setup without Docker and patches the
installed vLLM in place.

## Training
The model was fine-tuned on permissively licensed and synthetic decision data and merged into NVIDIA's NVFP4
checkpoint. No benchmark test items were used for training.

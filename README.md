<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark-bg.svg">
    <img src="assets/logo-light-bg.svg" alt="Singularity Compute" height="72">
  </picture>
</p>

<h1 align="center">Super Decider</h1>

<p align="center">
  The open-weights decision model that thinks when it's unsure.<br>
  Send a state and typed questions, get back a choice, a yes or no, or a score, with a calibrated probability for every answer.
</p>

<p align="center">
  <a href="https://benchmarkheaven.com/jev-models/api"><img alt="JevBench API board: #1 of 48" src="https://img.shields.io/badge/JevBench%20API-%231%20of%2048-0a7d3b"></a>
  <a href="https://huggingface.co/Singularity-Compute/super-decider"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-weights-ffd21e"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <a href="https://asicloud.cudos.org/"><img alt="Hosted on ASI:Cloud" src="https://img.shields.io/badge/hosted-ASI%3ACloud-5634f8"></a>
  <a href="docs/API.md"><img alt="API: Jev System One compatible" src="https://img.shields.io/badge/API-Jev%20System%20One-111111"></a>
</p>

<p align="center">
  <img src="assets/demo.svg" alt="A Super Decider request and its answer with a probability for every option" width="880">
</p>

Super Decider is built for the decisions inside your software: which team gets this ticket, is this email phishing, can the agent run this tool call, does this invoice add up. It never writes prose, so there is nothing to parse. Your code reads the probability and decides whether to act, ask a person or try a bigger model.

- **Open weights, Jev-compatible API.** Run it yourself or call our hosted endpoint. Moving from Jev means changing the URL and the model name.
- **Thinks only where it needs to.** Easy questions are answered in one pass. When the model is unsure, it writes a short thought first. Set `"thinking": "always"` to reason through every question.
- **Fast and cheap by design.** A diffusion model reads every answer slot in one step, and its mixture-of-experts backbone runs 4B of 26B parameters per decision: about 0.1 s per decision on a single GPU, $0.03 per million input tokens with output free.

## Benchmarks

### JevBench

Super Decider is #1 of 48 hosted decision APIs on the [JevBench API leaderboard](https://benchmarkheaven.com/jev-models/api) by Benchmark Heaven (board v1.7.48, 24 Oct 2026). The Composite score is the harmonic mean of intelligence, calibration, speed and cost over 1,500 decisions, 1,200 of them sealed. Capability is the mean of intelligence and calibration.

| System | Composite | Capability | Cost per 1,000 decisions | Median latency |
|---|---|---|---|---|
| Super Decider | **78.5** (#1) | 84.9 | $0.016 | 0.25 s |
| Super Decider, thinking always | 74.5 (#8) | 85.9 | $0.028 | 0.89 s |
| kettei-fat | 77.7 (#2) | **86.3** | $0.026 | 0.20 s |
| Fastino GLiDE | 76.4 (#3) | 84.5 | $0.023 | 0.36 s |
| Jev 1.13 | 71.4 (#14) | 77.1 | $0.032 | 0.24 s |

Both Super Decider rows were measured through our hosted API on 21 Oct 2026 and published at our request; see the [Super Decider page](https://benchmarkheaven.com/jev-models/singularity-super-decider) for the run details. Other rows come from the same board.

### Decision tasks

| Benchmark | Super Decider | Jev 1.14 | Clef |
|---|---|---|---|
| [When2Call](https://arxiv.org/abs/2504.18851): call a tool, ask, or decline | **85.6** | 81.7 | 72.5 |
| [PhishNChips](https://github.com/R-Lit/PhishNChips): should an email agent follow this link | **85.9** | 83.8 | 79.4 |
| [BFCL tool selection](https://gorilla.cs.berkeley.edu/leaderboard.html): which functions to call | 96.9 | 96.4 | **98.5** |
| [DecisionBench](https://github.com/Hanno-Labs/decision-bench): 23,900 decisions | **75.1** | 72.0 (Jev 1.13) | – |

When2Call, PhishNChips and BFCL are the versions in the [Jev Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index). We ran Super Decider on every row of each task with the Decision Index kit; Jev and Clef scores come from the public leaderboards. On our own held-out set of 1,119 long documents from domains not seen in training, Super Decider scores 94.5% with no drop on documents over 8k tokens.

## Quickstart

### Hosted API

Get a key on [ASI:Cloud](https://asicloud.cudos.org/) or write to [tech@singularitycompute.com](mailto:tech@singularitycompute.com).

```bash
curl https://llm.a.singularitycompute.com/v1/systemone \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "super-decider",
    "state": "Order 8812 shipped on Oct 1. The customer writes: \"I want to cancel and get my money back.\" Policy: shipped orders cannot be cancelled; they can be returned within 30 days of delivery for a full refund.",
    "questions": {
      "next_step": {"type": "choice", "instructions": "What should support do next?",
                    "criteria": {"refund_now": "Refund without a return", "start_return": "Start a return", "escalate": "Hand over to a human"}},
      "allowed":   {"type": "noul", "instructions": "Is cancelling this order allowed by the policy?"},
      "urgency":   {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]}
    }
  }'
```
```json
{"answers": {
  "next_step": {"type": "choice", "choice": "start_return",
                "probabilities": {"refund_now": 0.02, "start_return": 0.96, "escalate": 0.02}, "confidence": 0.96},
  "allowed":   {"type": "noul", "noul": 0.03},
  "urgency":   {"type": "score", "score": 0.59, "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.43, "1": 0.54, "2": 0.02}, "confidence": 0.54}},
 "model": "super-decider", "usage": {"input_tokens": ..., "output_tokens": ...}}
```

### With thinking on
Add `"thinking": "always"` when the answer needs a calculation or several steps. The model writes a short thought before
answering; the thought itself is not returned.
```bash
curl https://llm.a.singularitycompute.com/v1/systemone \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "super-decider",
    "thinking": "always",
    "state": "Invoice total 1,240 EUR. The customer paid 2 x 400 EUR and 1 x 340 EUR.",
    "questions": {
      "paid":      {"type": "noul", "instructions": "Is the invoice fully paid?"},
      "next_step": {"type": "choice", "instructions": "What should accounts receivable do next?",
                    "criteria": {"close": "Mark as paid and close", "remind": "Send a reminder for the remaining amount",
                                 "escalate": "Hand over to collections"}}
    }
  }'
```
```json
{"model": "super-decider",
 "answers": {
  "paid":      {"type": "noul", "noul": 0.02},
  "next_step": {"type": "choice", "choice": "remind",
                "probabilities": {"close": 0.01, "remind": 0.97, "escalate": 0.01}, "confidence": 0.97}},
 "usage": {"input_tokens": 394, "output_tokens": 240}}
```
The same request with `"thinking": "off"` answers in 0.27 s instead of 0.72 s, but less surely: `paid` 0.20 and
`remind` 0.80.

The same endpoint speaks OpenAI chat completions, so the OpenAI SDK works with a different base URL. See [`examples/`](examples/) for Python, the OpenAI SDK, batch calls and a confidence threshold that escalates to a person, and [`docs/API.md`](docs/API.md) for the full reference: question types, thinking modes, images and errors.

### Run it yourself

```bash
docker build -t super-decider runtime/
docker run --gpus all -p 8011:8011 -v $PWD/models:/data/models super-decider
curl localhost:8011/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state": "Order 8812 shipped Oct 1; the customer wants to cancel.", "questions": {"allowed": {"type": "noul", "instructions": "Is cancelling allowed?"}}}'
```

The container needs a GPU with 48 GB of memory and downloads the [weights](https://huggingface.co/Singularity-Compute/super-decider) (NVFP4, 18 GB) on first start. Set `-e API_KEY=...` to require a bearer token. Defaults live in [`runtime/Dockerfile`](runtime/Dockerfile); [`runtime/run.sh`](runtime/run.sh) runs the same setup without Docker.

## Where to use it

| Job | Questions you might ask |
|---|---|
| Support routing | Which team owns this ticket? How urgent is it? Is the customer about to churn? |
| Agent guardrails | Should the agent call this tool, ask a follow-up, or stop? Does this action need approval? |
| Moderation and safety | Does this message break the policy? Is this email phishing? |
| Document checks | Is this invoice complete? Does the total match the line items? Is this contract clause allowed? |
| Model routing | Can a small model handle this request, or does it need a large one? |

## How it works

- **Fine-tuned for decisions.** Google's DiffusionGemma 26B-A4B is fine-tuned on permissively licensed and synthetic decision data, then merged into NVIDIA's NVFP4 checkpoint.
- **One-step reads.** All answer slots of a request are filled in one denoising step, so several questions cost little more than one.
- **Bidirectional attention within each question.** The prompt block of every question is read in both directions, so the model sees the question and all its options at once. The model was trained this way, and the runtime applies the same attention through a small vLLM patch.
- **Adaptive thinking.** When the model is unsure of an answer, it writes a short thought and reads the answer again. A request can turn this off, keep it on under load, or think on every question.

## Limitations

- English is the primary language. Other languages work, but less well.
- Image questions are supported (up to 8 images per request), but the model was not fine-tuned on images.
- Thinking adds about 0.5 to 2 s to a request.
- Self-hosting needs one 48 GB GPU.

## Citation

```bibtex
@software{super_decider_2026,
  title  = {Super Decider: an open-weights decision model},
  author = {{Singularity Compute}},
  year   = {2026},
  url    = {https://github.com/Singularity-Compute/super-decider}
}
```

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). Built by [Singularity Compute](https://singularitycompute.com/) · [X](https://x.com/S_Compute) · [LinkedIn](https://www.linkedin.com/company/scompute/)

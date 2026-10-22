# super-decider API

Base URL `https://llm.a.singularitycompute.com`. Every request needs `Authorization: Bearer <key>` and `"model": "super-decider"` in the body.
A request carries a state (text, JSON or images) and typed questions; the reply has one calibrated answer per question.
All responses below are real outputs of the model, rounded.

| Question type | Ask with | Answer |
|---|---|---|
| `choice` | `criteria`: option name → description (or null) | `choice`, `probabilities`, `confidence` |
| `noul` | optional `criteria` `{"true": ..., "false": ...}` | `noul` = P(yes) |
| `score` | `criteria`: ordered list of levels | `score` = expected level (0-based), `legend`, `probabilities`, `confidence` |
| `span` / `spans` | optional `{"max_tokens": n}` | piece(s) of the state text with offsets |

Limits: 131,072 tokens and up to 64 questions per request, up to 8 images. `choice` questions can have more than 26
options (tested up to 151); they are read in chunks. Images go as data URLs inside the JSON body; multipart uploads are
not supported. Typical latency is about 0.25 s per request without thinking.

To control thinking per request, add `"thinking"` to the body: `"off"`, `"auto"` (the default: a short thought only on
uncertain questions, skipped when the server is busy), `"auto_forced"` (the same, but never skipped under load) or
`"always"` (every question gets a thought first). `"auto"` and `"auto_forced"` think only on text requests with at most
2 questions; with images or more questions, use `"always"` to get a thought. `"thinking_budget"` sets the thought length in tokens (16 to 4096; the
default is 512). Thinking raises accuracy on multi-step questions and costs about 0.5 to 2 s per request; on plain
classification it does not help. A too-long state returns 400 with the reason.

## 1. Jev API: `POST /v1/systemone`
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
 "model": "super-decider", "usage": {...}}
```

### Thinking
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

## 2. OpenAI-compatible: `POST /v1/chat/completions`
Exactly two messages: `system` holds the question schema as JSON, and `user` holds the state, also as JSON (wrap a plain string
with `json.dumps`; raw text returns 400).
```python
import json, os
from openai import OpenAI

client = OpenAI(base_url="https://llm.a.singularitycompute.com/v1", api_key=os.environ["API_KEY"])
schema = {"questions": [
    {"id": "next_step", "type": "choice", "instructions": "What should support do next?",
     "options": [{"name": "refund_now", "description": "Refund without a return"},
                 {"name": "start_return", "description": "Start a return"},
                 {"name": "escalate", "description": "Hand over to a human"}]},
    {"id": "allowed", "type": "noul", "instructions": "Is cancelling this order allowed by the policy?"},
    {"id": "urgency", "type": "score", "instructions": "How urgent is this ticket?", "levels": ["low", "medium", "high"]},
]}
state = "Order 8812 shipped on Oct 1. The customer writes: ..."
r = client.chat.completions.create(model="super-decider", messages=[
    {"role": "system", "content": json.dumps(schema)},
    {"role": "user", "content": json.dumps(state)},
])
answers = json.loads(r.choices[0].message.content)["answers"]
```
`content` is compact JSON with the answers only; every answer also carries a ready `label`:
```json
{"answers": {"next_step": {"type": "choice", "choice": "start_return", "probabilities": {...}, "confidence": 0.96, "label": "start_return"},
             "allowed": {"type": "noul", "noul": 0.04, "label": "no", "confidence": 0.96, "probabilities": {"yes": 0.04, "no": 0.96}},
             "urgency": {"type": "score", "score": 0.66, "legend": {...}, "probabilities": {...}, "confidence": 0.61, "label": "medium"}}}
```
`usage` has the real token counts. `output_tokens` includes the thought when the request thinks.

## 3. Images
Example image: [`invoice-demo.png`](invoice-demo.png). Images go as base64 data URLs (http(s) links
are untested). The model was not fine-tuned on images. The first image request after a restart takes
~5 s (warm-up).

**Jev API, JSON.** Put an `images` array next to `state`:
```python
import base64, json, os, requests

img = base64.b64encode(open("invoice-demo.png", "rb").read()).decode()
body = {
    "model": "super-decider",
    "state": "Scanned document from the accounts payable inbox.",
    "images": [f"data:image/png;base64,{img}"],
    "questions": {
        "doc_type": {"type": "choice", "instructions": "What kind of document is this?",
                     "criteria": {"invoice": "An invoice", "receipt": "A store receipt",
                                  "contract": "A contract", "other": "Something else"}},
        "biggest_line": {"type": "choice", "instructions": "Which line item has the largest amount?",
                         "criteria": {"paper": "Paper A4 (box)", "toner": "Toner cartridge", "lamp": "Desk lamp"}},
        "total_ok": {"type": "noul", "instructions": "Does the TOTAL equal the subtotal plus tax?"},
        "vendor": {"type": "choice", "instructions": "Who issued the document?",
                   "criteria": {"acme": "ACME Office Supplies", "northwind": "Northwind Ltd", "unknown": "Not shown"}},
    },
}
r = requests.post("https://llm.a.singularitycompute.com/v1/systemone",
                  headers={"Authorization": f"Bearer {os.environ['API_KEY']}"}, json=body, timeout=60)
print(json.dumps(r.json()["answers"], indent=2))
```
```json
{"doc_type": {"choice": "invoice", "confidence": 0.95}, "biggest_line": {"choice": "toner", "confidence": 0.96},
 "total_ok": {"noul": 0.93}, "vendor": {"choice": "acme", "confidence": 0.95}}
```

**Chat completions.** The `user` content is a list of `image_url` parts plus optional plain-text context:
```python
r = client.chat.completions.create(model="super-decider", messages=[
    {"role": "system", "content": json.dumps({"questions": [
        {"id": "doc_type", "type": "choice", "instructions": "What kind of document is this?",
         "options": [{"name": "invoice"}, {"name": "receipt"}, {"name": "contract"}, {"name": "other"}]},
        {"id": "total_ok", "type": "noul", "instructions": "Does the TOTAL equal the subtotal plus tax?"},
        {"id": "amount_band", "type": "score", "instructions": "How large is the invoice total?",
         "levels": ["under $100", "$100-$1,000", "over $1,000"]}]})},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        {"type": "text", "text": "Scanned document from the accounts payable inbox."},
    ]},
])
```
→ labels `invoice` (0.96), `yes` (P(yes) 0.91), `$100-$1,000` (0.92); usage 451 prompt / 23 completion tokens.

## 4. Errors
| Status | When | Body |
|---|---|---|
| 400 | `model` missing | `{"error": "Model name not provided"}` |
| 400 | unknown `model` name (for example `jev-latest`) | `{"error": "Model not found"}` |
| 400 / 422 | invalid request: unknown `thinking` value, bad question shape, state too long | `{"error": {"code": ..., "message": "<the model server's JSON error, as a string>"}}` |
| 401 | no `Authorization` header, or not in the form `Bearer sk-...` | `{"error": {"code": "authentication_error", "message": "Authorization header missing or invalid format. ..."}}` |
| 403 | key not valid | `{"error": {"code": "authentication_error", "message": "Access forbidden"}}` |
| 429 | rate limit for your key | no `Retry-After` header; wait a second or two and retry |

The `message` of a validation error is itself JSON; parse it to get the reason, for example
`thinking: one of "off", "auto", "auto_forced", "always"`.

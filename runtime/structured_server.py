"""Structured decisions in front of a vLLM DiffusionGemma server.

Derived from mmastrac/djev (Apache-2.0); modified by Singularity Compute.

POST /v1/systemone takes Jev's request body: {"model", "state", "questions"}.
"questions" maps an id to {"type", "instructions", "criteria"}, where "type"
is "noul", "choice", "score", "span" or "spans" and the criteria shape
follows the type:
  noul:   optional {"true": ..., "false": ...} descriptions
  choice: option name -> description or null
  score:  ordered list of levels
  span:   optional {"max_tokens": n}; the answer is a piece of the text
  spans:  optional {"max_tokens": n, "max_items": n}; every such piece
Answers take Jev's shapes, with this server's diagnostics alongside:
  noul:   {"noul": p}
  choice: {"choice", "probabilities", "confidence"}
  score:  {"score", "legend", "probabilities", "confidence"}
  span:   {"found", "text", "start", "end", "confidence", "coverage"}
  spans:  {"found", "items": [{"text", "start", "end", "confidence"}]}
A span is a substring of the state's text, by construction: the model fills
a pinned blank by copying, only the text's own token ids are read, and the
decode walks the text with those tokens. "start" and "end" are character
offsets into the state when it is a string, else into its "text" field.
The request body may also carry the schema keys "instructions", "samples",
"auto_max", "auto_threshold", "steps", "think", "ask", "chunk_rows",
"chunk_prompt" and "sequential" as extensions. Images go ahead of the
state, either as multipart/form-data with the JSON body in a part named
"request" and each image as a file part, or as an "images" array of data
URLs in the JSON body.

POST /v1/chat/completions makes the same decision from an OpenAI-shaped
call. The system message is the schema JSON below and the user message is
the state. The reply's `content` is the JSON answer set, {"answers": ...};
Read diagnostics are returned only when the server runs with DJEV_EXPOSE_DIAGNOSTICS=1.

POST /v1/raw/chat/completions passes the body to vLLM's chat completions
unchanged, for plain generation through this port.

With API_KEY set in the environment, every POST needs "Authorization:
Bearer <key>". --tls-port adds an HTTPS listener with a self-signed
certificate kept in --cert-dir, for clients that need a secure origin.

Each answer is one distribution per question, from one denoise step over a
seeded canvas, averaged over a few noise draws. This server handles the
canvas, tokenizer, slot resolution, noise draws and averaging.

Schema (system message):
  {"questions": [
     {"id": "urgent", "type": "noul", "instructions": "..."},
     {"id": "bucket", "type": "choice", "instructions": "...",
      "options": [{"name": "billing", "description": "..."}, ...]},
     {"id": "tone", "type": "score", "instructions": "...",
      "levels": ["calm", "annoyed", "furious"]}],
   "instructions": "optional context",
   "samples": "auto" | N, "auto_threshold": 0.1, "auto_max": 4,
   "steps": 1, "think": 0}

A question may also declare:
  "depends_on": [ids]        answered after those, with their answers in
                             its prompt
  "ask_if": {id: [answers]}  asked only when that question's answer is
                             among them (a skipped answer is null)
  "alone": true              a read of its own
Questions run in stages by these dependencies. Each stage is one joint
read. Later stages continue the earlier answers, prefilled for a text
state and restated for an image.

Up to ten questions answer as "id: label" lines. Past that the id runs
straight into the label, space separated, one row fewer per question. The
server splits a schema whose answer template does not fit the canvas into
chunks that run together, each with its own question list. "chunk_rows"
sets the rows per chunk, "ask" picks a subset of question ids for one
read, "sequential": true runs the chunks in order with the earlier answers
prefilled, and "chunk_prompt": "shared" lists every question in each
chunk's prompt. "think": N lets the model write up to N tokens in its
thought channel, as an ordinary generation, and the read then runs with
that thought in its prompt. With images the model writes the thought with
the image in view and the server seeds it into the canvas ahead of the
answer, so the canvas bounds it. The noise draws of a decision share one
thought.

Serve the model with a canvas that holds the answer template, for example:
  vllm serve google/diffusiongemma-26B-A4B-it \
      --diffusion-config '{"canvas_length": 64}' --max-logprobs 32 \
      --enable-prefix-caching
then run this in front of it:
  python structured_server.py --upstream http://127.0.0.1:8000 \
      --tokenizer google/diffusiongemma-26B-A4B-it --canvas 64 --port 8011

Two flags use engine features that are not in vLLM main yet, so both are off
by default. --constrained reads over the labels only (vllm-project/vllm#58216):
the same argmax for about a quarter less GPU time, with logprobs normalized
over the labels, so label mass is always 1. --engine-samples sends a fixed
sample count as one diffusion_samples request (vllm-project/vllm#58438)
instead of one request per draw.
"""

import argparse
import json
import math
import os
import random
import re
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pybase64 as base64

# samples when a request omits it: "auto" (upstream default) or a count
_ds = os.environ.get("DEFAULT_SAMPLES", "auto")
DEFAULT_SAMPLES = int(_ds) if _ds.isdigit() else _ds
# per-type temperature on the final label distribution (p ** (1/T), renormalized), e.g.
# DJEV_TEMPS="noul=0.6"; unset types stay at 1
EXPOSE_DIAGNOSTICS = os.environ.get("DJEV_EXPOSE_DIAGNOSTICS", "0") == "1"  # read diagnostics in responses (debug only)
TEMPS = {k.strip(): float(v) for k, v in (kv.split("=") for kv in os.environ.get("DJEV_TEMPS", "").split(",") if "=" in kv)}
# noul commit band: P(yes) inside (1-hi, hi) goes to hi+0.001 / 1-hi-0.001; unset = off
NOUL_COMMIT = float(os.environ.get("DJEV_NOUL_COMMIT", "0") or 0)
DOC_FIRST = os.environ.get("DJEV_DOC_FIRST", "0") == "1"
SANDWICH = os.environ.get("DJEV_DOC_FIRST", "0") == "2"  # questions first AND their ids restated after the state
JSON_INDEX = int(os.environ.get("DJEV_JSON_INDEX", "0") or 0)
_AT = {k.strip(): float(v) for k, v in (kv.split("=") for kv in os.environ.get("DJEV_ADAPTIVE_THINK", "").split(",") if "=" in kv)}
ADAPT_BUDGET = int(_AT.pop("budget", 1024)) if _AT else 0
ADAPT_MIN_TOKENS = int(_AT.pop("min_tokens", 0)) if _AT else 0
ADAPT_MAX_INFLIGHT = int(_AT.pop("max_inflight", 0)) if _AT else 0
ADAPT_MAX_Q = int(_AT.pop("max_questions", 0)) if _AT else 0
THINK_XARGS = {k: int(os.environ[e]) for k, e in (("diffusion_max_steps", "DJEV_THINK_STEPS"),
               ("diffusion_canvas_length", "DJEV_THINK_CANVAS")) if os.environ.get(e)}
INFLIGHT = [0]
MANY_OPTIONS = {k: (float(v) if k == 'eps' else int(v)) for k, v in (p.split('=') for p in os.environ.get('DJEV_MANY_OPTIONS', '').split(',') if '=' in p)}

from prometheus_client import CollectorRegistry as _PReg, Counter as _PCounter, Gauge as _PGauge, Histogram as _PHist, generate_latest as _p_latest
_P_REG = _PReg()  # own registry: no process_/python_ collectors that would clash with vLLM's /metrics
M_REQ = _PCounter("djev_requests", "djev HTTP requests by endpoint and status", ["endpoint", "status"], registry=_P_REG)
M_LAT = _PHist("djev_request_seconds", "djev request latency (POST, whole request)", ["endpoint"], registry=_P_REG,
               buckets=(0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0))
M_THINK = _PCounter("djev_thoughts", "adaptive thinking passes", registry=_P_REG)
M_MANY = _PCounter("djev_many_options", "choice questions with more than 26 options answered via chunks", registry=_P_REG)
M_INFL = _PGauge("djev_inflight", "decisions in flight in this server", registry=_P_REG)
M_INFL.set_function(lambda: INFLIGHT[0])
PLAIN_CHAT_MESSAGE = (
    "This model answers typed decision questions, not free-form chat. Send exactly two messages: a system message whose "
    "content is a JSON schema with a \"questions\" array, and a user message with the state as JSON. Or POST /v1/systemone "
    "with {\"state\": ..., \"questions\": {...}}."
)
_EP = ("/v1/systemone", "/v1/chat/completions", "/v1/raw/chat/completions", "/v1/models", "/health")
INFLIGHT_LOCK = threading.Lock()
ADAPT_GATES = _AT  # type -> raw top-probability gate
from transformers import AutoTokenizer

ARGS = None
TOK = None
API_KEY = os.environ.get(
    "API_KEY", ""
)  # when set, POST routes need "Authorization: Bearer <key>"
CANVAS_LEN = 64  # the served canvas length. A request may be narrower.
CANVAS_STEP = 16  # request widths are multiples of this
VOCAB = 262144
TURN_CLOSE = 106
PAD = 0
TOPK = int(os.environ.get("DJEV_TOPK", "20"))  # top logprobs per canvas position; labels come from logprob_token_ids
MAX_QUESTIONS = 64  # per request
MAX_SAMPLES = 32  # cap on reads per question, set by --max-samples
MAX_PARALLEL = 16  # question groups read at once
SPAN_STEPS = (
    4  # denoise steps for a span blank; one step spells it position by position
)
SPAN_REGION = 24  # default blank for one span, in tokens
SPAN_LIST_REGION = 64  # default blank for a list of spans
SPAN_ID_CAP = 128  # vLLM's logprob_token_ids cap: the text's ids must fit
SPAN_MIN_CONF = 0.3  # below this a span read is repeated with the next seed
SPAN_MAX_READS = 3
SPAN_PASSES = 2  # list reads per window, merged by offset
SPAN_FLOOR = -30.0
SPAN_BOUNDARY = set(" \t\n,;:.!?()[]{}\"'#$")
EOS = 1
NL = None
SPAN_SYSTEM = (
    "Answer with the exact value copied from the text, character for character, "
    "or none if the text has no such value.{context}\n\nQuestion answer: {q}\n"
    "  (the value only)\n\n"
    'Reply with one line per question, in this order, formatted as "id: label".'
)
SPAN_LIST_SYSTEM = (
    "List every value of the requested kind found in the text, one per line, "
    "each copied exactly as it appears, character for character, in the order "
    "they appear. Write none if there is none.{context}\n\nQuestion answer: {q}\n"
    "  (one value per line)\n\n"
    'Reply with the answer lines, formatted as "id: value".'
)
SPAN_CHOOSE_SYSTEM = (
    "Several candidate answers were copied from the text. Choose the one that "
    "is exactly the right answer, no more and no less.{context}\n\n"
    "Question answer: {q}\nCandidates:\n{options}\n\n"
    'Reply with one line, formatted as "id: label".'
)
# the empty thought block the chat template leaves to the model
SCAFFOLD_TEXT = "<|channel>thought\n<channel|>"
SCAFFOLD = None
THOUGHT_OPEN = None
THOUGHT_CLOSE = None


# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------


class SchemaError(ValueError):
    pass


def parse_schema(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > MAX_QUESTIONS:
        raise SchemaError(f"schema: at most {MAX_QUESTIONS} questions")
    qs = []
    seen = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, no ':' or newline"
            )
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [
                (o["name"], o.get("description"))
                if isinstance(o, dict)
                else (str(o), None)
                for o in opts
            ]
            labels = [chr(ord("A") + i) for i in range(len(choices))]
        elif kind == "score":
            choices = [(str(level), None) for level in (q.get("levels") or [])]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else [chr(ord("A") + i) for i in range(len(choices))]
            )
        elif kind in ("span", "spans"):
            default = SPAN_REGION if kind == "span" else SPAN_LIST_REGION
            span_cfg = {
                "max_tokens": q.get("max_tokens", default),
                "max_items": q.get("max_items", 16),
            }
            for key, hi in (("max_tokens", 96), ("max_items", 64)):
                v = span_cfg[key]
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= hi:
                    raise SchemaError(f"question {qid!r}: {key} must be 1 to {hi}")
            choices, labels = [], []
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if kind not in ("span", "spans") and len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        if len(choices) > 26:
            raise SchemaError(f"question {qid!r}: at most 26 alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a "
                "non-empty list of its answers"
            )
        if kind in ("span", "spans") and (deps or ask_if):
            raise SchemaError(
                f"question {qid!r}: a span question cannot depend on another question"
            )
        qs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": str(q.get("instructions", "")),
                "choices": choices,
                "labels": labels,
                "depends_on": list(dict.fromkeys(list(deps) + list(ask_if))),
                "ask_if": ask_if,
                "alone": bool(q.get("alone", False)),
                "span": span_cfg if kind in ("span", "spans") else None,
            }
        )
    by_id = {q["id"]: q for q in qs}
    for q in qs:
        for dep in q["depends_on"]:
            if dep not in by_id or dep == q["id"]:
                raise SchemaError(
                    f"question {q['id']!r}: depends on unknown question {dep!r}"
                )
            if by_id[dep]["span"]:
                raise SchemaError(
                    f"question {q['id']!r}: cannot depend on the span question {dep!r}"
                )
        for dep, vals in q["ask_if"].items():
            names = [c[0] for c in by_id[dep]["choices"]]
            if any(v not in names for v in vals):
                raise SchemaError(
                    f"question {q['id']!r}: ask_if values for {dep!r} "
                    f"must be among {names}"
                )
    schedule(qs)  # refuses a cycle
    samples = value.get("samples", DEFAULT_SAMPLES)
    if samples == "auto":
        policy = {
            "mode": "auto",
            "max": max(1, min(int(value.get("auto_max", 4)), MAX_SAMPLES)),
            "threshold": float(value.get("auto_threshold", 0.1)),
        }
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, MAX_SAMPLES)}
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
        for q in qs:
            if q["id"] in ask and any(d not in ask for d in q["depends_on"]):
                raise SchemaError(
                    f"schema: ask names {q['id']!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    sequential = bool(value.get("sequential", False))
    think = value.get("think", 0)
    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4096:
        raise SchemaError("schema: think must be a thought budget in tokens, 0 to 4096")
    adaptive = value.get("adaptive", True)
    if not isinstance(adaptive, bool):
        raise SchemaError("schema: adaptive must be true or false")
    labeled = [q for q in qs if not q["span"]]
    fmt = "lines" if len(labeled) <= 10 else "indexed"
    if fmt == "indexed" and not all(fits_indexed(q) for q in labeled):
        fmt = "lines"  # an id ran into its label; "id: label" keeps them apart
    return {
        "questions": qs,
        "instructions": value.get("instructions"),
        "policy": policy,
        "steps": max(1, min(int(value.get("steps", 1)), 8)),
        "think": think,
        "adaptive": adaptive,
        "ask": ask,
        "chunk_rows": chunk_rows,
        "chunk_prompt": chunk_prompt,
        "sequential": sequential,
        "format": fmt,
    }


_indexed_fit_cache = {}


def fits_indexed(q):
    """Whether each of the question's labels stays one token when written
    straight after its id, as "indexed" does ("toxicno" merges, "q0no" not)."""
    key = (q["id"], tuple(q["labels"]))
    if key not in _indexed_fit_cache:
        if len(_indexed_fit_cache) > 4096:
            _indexed_fit_cache.clear()
        try:
            resolve_template([q], [], FORMATS["indexed"][0], "indexed")
            _indexed_fit_cache[key] = True
        except SchemaError:
            _indexed_fit_cache[key] = False
    return _indexed_fit_cache[key]


# Answer template shape: (join between questions, what precedes the label,
# reply instruction). A small schema gets "lines", which is readable.
# "indexed" ("0yes 1no") costs three tokens a question against four or five
# and agreed with "lines" on every set tried: 42 booleans, ten 26-way
# choices, twenty 5-level scores. Past ten questions the saved rows keep a
# schema in one read. Two tokens a question, or no id at all, loses
# alignment beyond about twenty questions, because the id ties a label to
# its question.
FORMATS = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}


def system_text(schema, chunked=False):
    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"]):
            if q["type"] == "noul":
                s += f"  {label}: {str(desc).strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += "\n" + FORMATS[schema.get("format", "lines")][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line "
            "that is present."
        )
    return s


DECISION_PREAMBLE = (
    "Answer a fixed set of questions about the state the user provides. "
    "Each question lists its allowed answers; reply with exactly one label "
    "per question.\n"
)


def chat_messages(sys_text, state_content):
    """The system and user messages of a read. With DJEV_DOC_FIRST=1 a decision
    read keeps only the fixed preamble in the system message and moves the
    questions after the state in the user turn, so every request about the
    same state starts with the same tokens and vLLM's prefix cache reuses the
    state's prefill. Other system texts (spans) keep their layout."""
    if SANDWICH and sys_text.startswith(DECISION_PREAMBLE):
        ids = re.findall(r"^Question (\S+): ", sys_text[len(DECISION_PREAMBLE):], re.M)
        tail = "\n\nQuestions to answer: " + ", ".join(ids)
        user = state_content + tail if isinstance(state_content, str) else list(state_content) + [{"type": "text", "text": tail}]
        return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]
    if not (DOC_FIRST and sys_text.startswith(DECISION_PREAMBLE)):
        return [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": state_content},
        ]
    rest = sys_text[len(DECISION_PREAMBLE):].lstrip("\n")
    if isinstance(state_content, str):
        user = state_content + "\n\n" + rest
    else:  # multimodal parts: the questions follow as one more text part
        user = list(state_content) + [{"type": "text", "text": "\n\n" + rest}]
    return [
        {"role": "system", "content": DECISION_PREAMBLE.rstrip("\n")},
        {"role": "user", "content": user},
    ]


def answer_text(qs, labels, fmt="lines"):
    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i] for q, i in zip(qs, labels)
    )


def enc(text):
    return TOK.encode(text, add_special_tokens=False)


def init_tokenizer(tok):
    global TOK, SCAFFOLD, THOUGHT_OPEN, THOUGHT_CLOSE, EOS, NL
    TOK = tok
    if tok.eos_token_id is not None:
        EOS = int(tok.eos_token_id)
    NL = enc("\n")[0]
    THOUGHT_OPEN = enc("<|channel>thought\n")
    THOUGHT_CLOSE = enc("<channel|>")
    SCAFFOLD = enc(SCAFFOLD_TEXT)
    assert THOUGHT_OPEN + THOUGHT_CLOSE == SCAFFOLD, (
        "the thought tags must tokenize apart"
    )


def resolve_template(qs, head, lead, fmt):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or this raises SchemaError. ``head`` is the token run the canvas
    starts with: the empty thought block for a plain read, and empty when the
    prompt already ends the thought channel. ``lead`` is the text before the
    first answer: the join when earlier answers are in the prompt, so the
    tokens match one joint template."""
    base_labels = [0] * len(qs)
    base = head + enc(lead + answer_text(qs, base_labels, fmt))
    if len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"answer template is {len(base)} tokens; the canvas holds {CANVAS_LEN - 1}"
        )
    if len(qs) == 1 and len(base) + 1 > CANVAS_LEN:
        raise SchemaError(
            f"question {qs[0]['id']!r} alone needs {len(base) + 1} canvas rows"
        )
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + enc(lead + answer_text(qs, labels, fmt))
            if len(e) != len(base):
                raise SchemaError(
                    f"question {q['id']!r}: label {q['labels'][li]!r} is not a "
                    "single token"
                )
            diffs = [i for i in range(len(e)) if e[i] != base[i]]
            if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(
                    f"question {q['id']!r}: labels do not share one template slot"
                )
            pos = diffs[0]
            ids[li] = e[pos]
        ids[0] = base[pos]
        if len(set(ids)) != len(ids):
            raise SchemaError(
                f"question {q['id']!r}: two labels tokenize to the same id"
            )
        slots.append({"pos": pos, "label_ids": ids})
    return base, slots


_template_cache = {}


def template_for(schema, head, lead):
    fmt = schema.get("format", "lines")
    key = json.dumps(
        [head, lead, fmt] + [(q["id"], q["labels"]) for q in schema["questions"]]
    )
    if key not in _template_cache:
        if len(_template_cache) > 4096:  # bounded like the other caches (one entry per distinct question set)
            _template_cache.clear()
        _template_cache[key] = resolve_template(schema["questions"], head, lead, fmt)
    return _template_cache[key]


# ----------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------


def canvas_width(template):
    """Smallest multiple of CANVAS_STEP that holds the template and the turn close."""
    need = len(template) + 1
    return min(CANVAS_LEN, -(-need // CANVAS_STEP) * CANVAS_STEP)


def constrained_xargs():
    """Read over the labels only. The engine runs the unembedding, sampler and
    self-conditioning over the request's logprob_token_ids instead of the whole
    vocabulary. Same argmax, about a quarter less GPU time per read. Logprobs
    come back normalized over the labels."""
    return {"diffusion_constrained": True} if ARGS.constrained else {}


def pin_xargs(template, slots, steps, samples=1):
    """Past one denoise step the template must be held, or accept/renoise
    rewrites it: pin every canvas position that is not an answer slot. A
    multi-sample request pins the same positions, which tells the engine
    what to re-noise per sample."""
    if steps <= 1 and samples <= 1:
        return {}
    free = {s["pos"] for s in slots}
    return {
        "diffusion_pinned": [p for p in range(canvas_width(template)) if p not in free]
    }


def build_canvas(template, slots, seed):
    rng = random.Random(seed)
    canvas = list(template) + [TURN_CLOSE]
    canvas += [PAD] * (canvas_width(template) - len(canvas))
    for s in slots:
        canvas[s["pos"]] = rng.randrange(VOCAB)
    return canvas


def label_id_union(slots):
    ids = sorted({i for s in slots for i in s["label_ids"]})
    return ids[:128]  # vLLM's cap per request. A schema needs far fewer.


def upstream_chat(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def upstream_completions(body, timeout=600):
    req = urllib.request.Request(
        ARGS.upstream.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat_prompt_ids(sys_text, state_text, thinking=False):
    """The prompt the chat endpoint would build, as token ids, ending after
    the model turn marker. Text states only. ``thinking`` turns the chat
    template's thinking marker on."""
    messages = chat_messages(sys_text, state_text)
    out = TOK.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking
    )
    ids = (
        out["input_ids"] if hasattr(out, "keys") else out
    )  # newer transformers return a dict
    return [int(t) for t in ids]


def think(sys_text, state_text, budget):
    """A read prefix that ends with a thought the model wrote: the chat prompt
    with thinking on, the open tag, up to ``budget`` generated tokens, the
    close tag. Returns the prefix and a diagnostics dict for the thought."""
    prompt = chat_prompt_ids(sys_text, state_text, thinking=True) + THOUGHT_OPEN
    started = time.time()
    d = upstream_completions(
        {
            "model": ARGS.model,
            "prompt": prompt,
            "max_tokens": budget,
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "stop_token_ids": THOUGHT_CLOSE,
            **({"vllm_xargs": THINK_XARGS} if THINK_XARGS else {}),
        }
    )
    ids = [int(t.split(":")[1]) for t in d["choices"][0]["logprobs"]["tokens"]]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    info = {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }
    return prompt + ids + THOUGHT_CLOSE, info


def think_chat(sys_text, state_content, budget):
    """A thought written with the image in view: the chat endpoint with
    thinking on, capped at ``budget`` tokens and cut at the close tag.
    Returns the thought's token ids and a diagnostics dict."""
    body = {
        "model": ARGS.model,
        "messages": chat_messages(sys_text, state_content),
        "max_tokens": budget,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
        "stop_token_ids": THOUGHT_CLOSE,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    started = time.time()
    d = upstream_chat(body)
    choice = d["choices"][0]
    ids = [
        int(t["token"].split(":")[1])
        for t in (choice.get("logprobs") or {}).get("content") or []
    ]
    if ids[: len(THOUGHT_OPEN)] == THOUGHT_OPEN:  # the model opens the channel itself
        ids = ids[len(THOUGHT_OPEN) :]
    closed = THOUGHT_CLOSE[0] in ids
    if closed:
        ids = ids[: ids.index(THOUGHT_CLOSE[0])]
    return ids, {
        "tokens": len(ids),
        "closed": closed,
        "ms": (time.time() - started) * 1e3,
        "text": TOK.decode(ids),
    }


def read_xargs(schema, template, slots, seed, samples):
    x = {
        "diffusion_seed_canvas": build_canvas(template, slots, seed),
        "diffusion_canvas_length": canvas_width(template),
        "diffusion_max_steps": schema["steps"],
        "diffusion_read_only": True,
        **pin_xargs(template, slots, schema["steps"], samples),
        **constrained_xargs(),
    }
    if samples > 1:
        x["diffusion_samples"] = samples
    return x


def samples_body(seed, samples):
    """The engine draws the noise of a multi-sample request from the request
    seed, so the draws reproduce for a schema seed."""
    return {"seed": seed} if samples > 1 else {}


def one_read(
    schema,
    template,
    slots,
    sys_text,
    state_content,
    seed,
    prefix=None,
    thinking=False,
    samples=1,
):
    """One request, ``samples`` noise draws of the canvas: a list of one
    distribution per question for each draw, and the usage."""
    if prefix is not None:
        return one_read_continuation(schema, template, slots, prefix, seed, samples)
    messages = chat_messages(sys_text, state_content)
    body = {
        "model": ARGS.model,
        "messages": messages,
        "max_tokens": len(template) + 1,
        "logprobs": True,
        "top_logprobs": TOPK,
        # Exact logprobs for every label at every position. With a long
        # option list most labels never rank in the top-k, and the model's
        # mass sits on tokens that spell the option name instead.
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "vllm_xargs": read_xargs(schema, template, slots, seed, samples),
        **samples_body(seed, samples),
    }
    d = upstream_chat(body)
    draws = []
    for choice in d["choices"]:
        content = choice["logprobs"]["content"]
        out = []
        for q, s in zip(schema["questions"], slots):
            top = {
                int(t["token"].split(":")[1]): t["logprob"]
                for t in content[s["pos"]]["top_logprobs"]
            }
            out.append(slot_distribution(top, s["label_ids"]))
        draws.append(out)
    return draws, d.get("usage", {})


def slot_distribution(top, label_ids):
    """Label probabilities at one slot from the returned logprobs: every
    label's own value plus the argmax token. Read-only logprobs are at
    temperature 1, so the label softmax uses them directly. The entropy is
    over that returned set."""
    floor = min(top.values()) - 5.0
    lp_t = [top.get(i, floor) for i in label_ids]
    mx = max(lp_t)
    ex = [math.exp(x - mx) for x in lp_t]
    probs = [e / sum(ex) for e in ex]
    top_p = [math.exp(v) for v in top.values()]
    return {
        "probs": probs,
        "label_mass": sum(math.exp(x) for x in lp_t),
        "entropy": -sum(p * math.log(p) for p in top_p if p > 0),
        "argmax_is_label": max(top, key=top.get) in label_ids,
    }


def one_read_continuation(schema, template, slots, prompt_ids, seed, samples=1):
    """A read whose prompt already holds the thought scaffold and earlier
    answer lines, sent as token ids so the chat template cannot alter it."""
    body = {
        "model": ARGS.model,
        "prompt": prompt_ids,
        "max_tokens": len(template) + 1,
        "logprobs": TOPK,
        "logprob_token_ids": label_id_union(slots),
        "return_tokens_as_token_ids": True,
        "vllm_xargs": read_xargs(schema, template, slots, seed, samples),
        **samples_body(seed, samples),
    }
    d = upstream_completions(body)
    draws = []
    for choice in d["choices"]:
        rows = choice["logprobs"]["top_logprobs"]
        out = []
        for q, sl in zip(schema["questions"], slots):
            top = {int(k.split(":")[1]): v for k, v in rows[sl["pos"]].items()}
            out.append(slot_distribution(top, sl["label_ids"]))
        draws.append(out)
    return draws, d.get("usage", {})


def read_many(
    schema,
    template,
    slots,
    sys_text,
    state_content,
    seed,
    n,
    prefix=None,
    thinking=False,
    engine=False,
):
    """``n`` noise draws of the read: one diffusion_samples request when
    ``engine`` is set, otherwise one request per draw."""
    if n > 1 and engine and ARGS.engine_samples:
        draws, usage = one_read(
            schema, template, slots, sys_text, state_content, seed, prefix, thinking, n
        )
        return draws, [usage] + [None] * (len(draws) - 1)
    results = [None] * n
    errors = [None] * n
    usages = [None] * n

    def run(k):
        try:
            (results[k],), usages[k] = one_read(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + k * 7919,
                prefix,
                thinking,
            )
        except Exception as e:  # raised again below
            errors[k] = e

    threads = [threading.Thread(target=run, args=(k,)) for k in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results, usages


def schedule(qs):
    """Questions in stages: a question's stage comes after the stages of
    everything it depends on. Declaration order is kept within a stage."""
    ids = {q["id"] for q in qs}
    pending = list(qs)
    done: set = set()
    levels = []
    while pending:
        level = [
            q
            for q in pending
            if all(d in done or d not in ids for d in q["depends_on"])
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among " + ", ".join(q["id"] for q in pending)
            )
        levels.append(level)
        done |= {q["id"] for q in level}
        pending = [q for q in pending if q["id"] not in done]
    return levels


def chunk_groups(schema, qs):
    """``qs`` split, in order, into the fewest groups whose answer templates
    fit ``chunk_rows`` (the canvas by default). A question marked alone gets
    its own group."""
    limit = schema.get("chunk_rows") or CANVAS_LEN
    groups, group = [], []
    for q in qs:
        if q["alone"]:
            if group:
                groups.append(group)
                group = []
            groups.append([q])
            continue
        trial = group + [q]
        rows = (
            len(SCAFFOLD)
            + len(
                enc(answer_text(trial, [0] * len(trial), schema.get("format", "lines")))
            )
            + 1
        )
        if rows > limit and group:
            groups.append(group)
            group = [q]
        else:
            group = trial
    if group:
        groups.append(group)
    return groups


def answer_name(q, a):
    """The answer as the name ask_if compares against: yes or no, an option
    name, or a level name."""
    if a is None:
        return None
    return a["label"] if q["type"] == "noul" else a.get("choice", a.get("level"))


def decide(schema, state_content, seed):
    """One decision. Questions run in stages by their dependencies. A stage
    is one joint read, chunked by the canvas, with a question marked alone
    in its own read. Later stages condition on every earlier answer: a
    prefilled continuation for a text state, or the answers restated in the
    state for an image. A question whose ask_if condition failed is skipped
    and its answer is null."""
    started = time.time()
    qs = [
        q
        for q in schema["questions"]
        if not schema.get("ask") or q["id"] in schema["ask"]
    ]
    span_qs = [q for q in qs if q["span"]]
    qs = [q for q in qs if not q["span"]]
    # Span questions are reads of their own, run beside the label reads.
    span_pool = ThreadPoolExecutor(max_workers=1) if span_qs else None
    span_future = (
        span_pool.submit(decide_spans, schema, span_qs, state_content, seed)
        if span_qs
        else None
    )
    schema = dict(schema, questions=[q for q in schema["questions"] if not q["span"]])
    if not qs:
        span_answers, span_diag = span_future.result()
        span_pool.shutdown(wait=False)
        diagnostics = {
            "steps": schema["steps"],
            "stages": [],
            "skipped": {},
            "chunks": [],
            "spans": span_diag,
            "timing": {
                "total_ms": (time.time() - started) * 1e3,
                "reads": span_diag["reads"],
            },
            "questions": {},
            "engine": "vllm",
        }
        return {"answers": span_answers, "diagnostics": diagnostics}, span_diag["rows"]
    levels = schedule(qs)
    text_state = isinstance(state_content, str)
    fmt = schema["format"]
    join = FORMATS[fmt][0]
    # More than one read in sequence needs the full question list in every
    # prompt, so that later reads continue one answer.
    chained = len(levels) > 1 or schema["sequential"]
    sys_full = system_text(schema, chunked=not text_state and chained)
    base_ids, thought = None, None
    if chained and text_state:
        if schema["think"]:
            base_ids, thought = think(sys_full, state_content, schema["think"])
        else:
            base_ids = chat_prompt_ids(sys_full, state_content) + SCAFFOLD
    shared = schema["chunk_prompt"] == "shared"
    answered, lines, earlier = {}, [], []
    parts, stages, chunks, skipped = [], [], [], {}
    by_id = {q["id"]: q for q in schema["questions"]}

    def run(group, k, conditioned):
        sub = dict(schema, questions=group)
        # The thought is written once: in the prefix of a chained text decision,
        # or in the first read of anything else.
        sub["think"] = 0 if (chained and text_state) or conditioned else schema["think"]
        if text_state:
            if conditioned:
                prefix, lead, sys_text = (
                    base_ids + enc(join.join(lines)),
                    join,
                    sys_full,
                )
            elif chained:
                prefix, lead, sys_text = (base_ids if thought else None), "", sys_full
            else:
                prefix, lead, sys_text = (
                    None,
                    "",
                    (system_text(schema, chunked=True) if shared else system_text(sub)),
                )
            return decide_group(
                sub, sys_text, state_content, seed + 104729 * k, prefix, lead
            )
        state = state_content
        if conditioned:
            text = next(
                (p["text"] for p in state_content if p.get("type") == "text"), ""
            )
            text += "\n\nAnswers so far:\n" + "\n".join(earlier)
            state = [p for p in state_content if p.get("type") != "text"] + [
                {"type": "text", "text": text}
            ]
            sys_text = sys_full
        else:
            sys_text = sys_full if (chained or shared) else system_text(sub)
        return decide_group(sub, sys_text, state, seed + 104729 * k)

    def absorb(group, body, rows):
        parts.append((body, rows))
        chunks.append([q["id"] for q in group])
        answered.update(body["answers"])
        lines.append(
            answer_text(
                group,
                [q["labels"].index(body["answers"][q["id"]]["label"]) for q in group],
                fmt,
            )
        )
        earlier.extend(
            f"{q['id']}: {answer_name(q, body['answers'][q['id']])}" for q in group
        )

    k = 0
    for level in levels:
        asked = []
        for q in level:
            failed = next(
                (
                    (dep, vals)
                    for dep, vals in q["ask_if"].items()
                    if answer_name(by_id[dep], answered.get(dep)) not in vals
                ),
                None,
            )
            if failed:
                answered[q["id"]] = None
                skipped[q["id"]] = {
                    "because": failed[0],
                    "was": answer_name(by_id[failed[0]], answered.get(failed[0])),
                    "wanted": failed[1],
                }
                continue
            asked.append(q)
        if not asked:
            continue
        stages.append([q["id"] for q in asked])
        groups = chunk_groups(schema, asked)
        conditioned = bool(lines)
        if schema["sequential"] or len(groups) == 1:
            for group in groups:
                body, rows = run(
                    group, k, conditioned or (schema["sequential"] and bool(lines))
                )
                absorb(group, body, rows)
                k += 1
        else:
            with ThreadPoolExecutor(max_workers=min(len(groups), MAX_PARALLEL)) as ex:
                results = list(
                    ex.map(
                        lambda gk, conditioned=conditioned: run(
                            gk[1], gk[0], conditioned
                        ),
                        [(k + i, g) for i, g in enumerate(groups)],
                    )
                )
            for group, (body, rows) in zip(groups, results):
                absorb(group, body, rows)
            k += len(groups)

    answers = {q["id"]: answered.get(q["id"]) for q in qs}
    diag_q = {}
    for body, _ in parts:
        diag_q.update(body["diagnostics"]["questions"])
    # A thought written here (a chained text decision) is outside every
    # group's row count. One written inside a group is already counted there.
    extra_rows = thought["tokens"] if thought else 0
    if thought is None:
        thoughts = [b["diagnostics"].get("thought") for b, _ in parts]
        thought = (
            thoughts[0] if len(parts) == 1 else ([t for t in thoughts if t] or None)
        )
    one = len(parts) == 1 and not skipped
    diagnostics = {
        "steps": schema["steps"],
        "stages": stages,
        "skipped": skipped,
        "chunks": chunks,
        "chunk_prompt": "full" if chained else schema["chunk_prompt"],
        "sequential": schema["sequential"],
        "conditioning": (
            None
            if len(stages) <= 1 and not schema["sequential"]
            else ("prefill" if text_state else "restated")
        ),
        "thought": thought,
        "samples": (
            parts[0][0]["diagnostics"]["samples"]
            if one
            else {
                "n": [b["diagnostics"]["samples"]["n"] for b, _ in parts],
                "tops": [b["diagnostics"]["samples"]["tops"] for b, _ in parts],
                "policy": [b["diagnostics"]["samples"]["policy"] for b, _ in parts],
            }
        ),
        "timing": {
            "total_ms": (time.time() - started) * 1e3,
            "reads": sum(b["diagnostics"]["timing"]["reads"] for b, _ in parts),
            "requests": sum(b["diagnostics"]["timing"]["requests"] for b, _ in parts),
        },
        "prompt_tokens": max(
            (b["diagnostics"].get("prompt_tokens") or 0) for b, _ in parts
        )
        or None,
        "questions": diag_q,
        "engine": "vllm",
    }
    total_rows = sum(rows for _, rows in parts) + extra_rows
    if span_future is not None:
        try:
            span_answers, span_diag = span_future.result()
        finally:
            span_pool.shutdown(wait=False)
        answers.update(span_answers)
        diagnostics["spans"] = span_diag
        diagnostics["timing"]["reads"] += span_diag["reads"]
        diagnostics["timing"]["total_ms"] = (time.time() - started) * 1e3
        total_rows += span_diag["rows"]
    order = [q["id"] for q in qs] + [q["id"] for q in span_qs]
    answers = {qid: answers.get(qid) for qid in order}
    return {"answers": answers, "diagnostics": diagnostics}, total_rows


# ----------------------------------------------------------------------------
# Spans
# ----------------------------------------------------------------------------


def span_text_of(state):
    """The text a span's offsets index for a state as the client sent it: the
    state itself when it is a string, its "text" field when it is an object
    holding one as a string, else None (spans then ground in the rendered
    state, see span_source)."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict) and isinstance(state.get("text"), str):
        return state["text"]
    return None


def span_source(state_content, schema=None):
    """The text a span is grounded in and its offsets refer to. The request's
    span_text when it has one (a string state, or an object state's "text"
    field); else the state as the model reads it: the message when it is a
    string, the text part of an image state. The model reads an object state
    as JSON, so grounding in that JSON would put offsets into the JSON and
    make any escaped character (a newline, a quote) impossible to copy."""
    if schema is not None and schema.get("span_text") is not None:
        return schema["span_text"]
    if isinstance(state_content, str):
        return state_content
    return next((p["text"] for p in state_content if p.get("type") == "text"), "")


_piece_cache = {}
_piece_lock = threading.Lock()


def span_pieces(text):
    """id -> the string that token contributes: every token of the text in
    both spacings, plus each word's own first token with and without a
    leading space, since the model opens a value like "A-1042" with " A"
    while the text only holds "#A"."""
    with _piece_lock:
        if text in _piece_cache:
            return _piece_cache[text]
    variants = [" " + text, text]
    for w in set(text.split()):
        variants += [" " + w, w]
        bare = w.lstrip("#$([\"'")
        if bare and bare != w:
            variants += [" " + bare, bare]
    pieces = {}
    for v in variants:
        ids = enc(v)
        for tid, piece in zip(ids, TOK.convert_ids_to_tokens(ids)):
            if piece and not piece.startswith("<"):
                pieces[tid] = piece.replace("\u2581", " ")
    with _piece_lock:
        if len(_piece_cache) > 256:
            _piece_cache.clear()
        _piece_cache[text] = pieces
    return pieces


def span_read_canvas(sys_text, state_content, body_ids, free, allowed, seed, steps):
    """One pinned read of a canvas whose body follows the scaffold, with the
    positions in ``free`` left as noise. Returns per body position the
    logprobs of the ``allowed`` ids, and the emitted ids."""
    rng = random.Random(seed)
    shown = [rng.randrange(VOCAB) if i in free else t for i, t in enumerate(body_ids)]
    template = SCAFFOLD + shown
    width = canvas_width(template)
    if len(template) + 1 > width:
        raise SchemaError(
            f"a span blank of {len(body_ids)} rows does not fit the canvas"
        )
    canvas = template + [TURN_CLOSE]
    canvas += [PAD] * (width - len(canvas))
    want = sorted(allowed)
    if len(want) > SPAN_ID_CAP:
        raise SchemaError(
            f"{len(want)} distinct token ids; a read allows {SPAN_ID_CAP}"
        )
    pinned = [p for p in range(width) if (p - len(SCAFFOLD)) not in free]
    d = upstream_chat(
        {
            "model": ARGS.model,
            "messages": chat_messages(sys_text, state_content),
            "max_tokens": width,
            "logprobs": True,
            "top_logprobs": 1,
            "logprob_token_ids": want,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": width,
                "diffusion_max_steps": steps,
                "diffusion_read_only": True,
                "diffusion_pinned": pinned,
            },
        }
    )
    content = d["choices"][0]["logprobs"]["content"]
    rows, emitted = [], []
    for c in content[len(SCAFFOLD) : len(SCAFFOLD) + len(body_ids)]:
        rows.append(
            {int(t["token"].split(":")[1]): t["logprob"] for t in c["top_logprobs"]}
        )
        emitted.append(int(c["token"].split(":")[1]))
    return rows, emitted, width


def span_decode(text, rows, pieces, none_id, max_slop=2, slop_cost=-1.5, min_start=0):
    """The best contiguous substring under the blank's logprobs. Each token
    scores as regret against the model's own top pick at its position, so a
    long span pays only where it departs from what the model wanted to
    write. Up to ``max_slop`` blank positions may be skipped at ``slop_cost``
    each without advancing in the text, so an inserted token (a fix, a
    closing quote) does not break the walk. Returns (score, start, end,
    confidence, coverage); start is None for none. Confidence is the weakest
    normalised probability along the span and its end; coverage is the mass
    the allowed ids held there."""
    n = len(rows)
    L = len(text)
    end_ids = {TURN_CLOSE, EOS, NL}
    matches = [[] for _ in range(L + 1)]
    starts = [[] for _ in range(L + 1)]
    for tid, pc in pieces.items():
        c = text.find(pc)
        while c != -1:
            matches[c].append((tid, len(pc)))
            c = text.find(pc, c + 1)
        if pc.startswith(" ") and len(pc) > 1:
            q = pc[1:]
            c = text.find(q)
            while c != -1:
                starts[c].append((tid, len(q)))
                # one space from the model where the text has a run of whitespace
                w = c
                while w > 0 and text[w - 1].isspace():
                    w -= 1
                if c - w > 1:
                    matches[w].append((tid, c - w + len(q)))
                c = text.find(q, c + 1)
    space_ids = [tid for tid, pc in pieces.items() if pc == " "]
    z = [sum(math.exp(v) for v in r.values()) or 1e-9 for r in rows]
    top = [max(r.values()) if r else SPAN_FLOOR for r in rows]
    end_lp = [
        max(r.get(e, SPAN_FLOOR) for e in end_ids) - top[k] for k, r in enumerate(rows)
    ]
    punct = {
        tid
        for tid, pc in pieces.items()
        if pc.strip() and all(ch in SPAN_BOUNDARY for ch in pc.strip())
    }
    end_conf = [
        sum(math.exp(r[t]) for t in r if t in end_ids or t in punct) / z[k]
        for k, r in enumerate(rows)
    ]
    cov = [min(1.0, zk) for zk in z]
    best = None
    for a in range(min_start, L):
        if text[a].isspace() or not (a == 0 or text[a - 1] in SPAN_BOUNDARY):
            continue
        dp = {(a, 0): (0.0, 1.0)}
        for k in range(n - 1):
            nxt = {}
            for (c, used), (sc, mn) in dp.items():
                opts = matches[c]
                if k == 0:  # a value may open with a lone space token
                    opts = opts + starts[c] + [(t, 0) for t in space_ids]
                for tid, ln in opts:
                    lp = rows[k].get(tid, SPAN_FLOOR)
                    if lp <= SPAN_FLOOR:
                        continue
                    cand = (sc + lp - top[k], min(mn, math.exp(lp) / z[k]))
                    key = (c + ln, used)
                    if key not in nxt or cand[0] > nxt[key][0]:
                        nxt[key] = cand
                if used < max_slop and c > a:
                    key = (c, used + 1)
                    cand = (sc + slop_cost, mn)
                    if key not in nxt or cand[0] > nxt[key][0]:
                        nxt[key] = cand
            dp = nxt
            if not dp:
                break
            for (c, used), (sc, mn) in dp.items():
                if c == a or not (c == L or text[c] in SPAN_BOUNDARY):
                    continue
                total = sc + end_lp[k + 1]
                if best is None or total > best[0]:
                    best = (
                        total,
                        a,
                        c,
                        min(mn, end_conf[k + 1]),
                        sum(cov[: k + 2]) / (k + 2),
                    )
    none = rows[0].get(none_id, SPAN_FLOOR) - top[0] + (end_lp[1] if n > 1 else 0.0)
    if best is None or none > best[0]:
        return (
            none,
            None,
            None,
            math.exp(rows[0].get(none_id, SPAN_FLOOR)) / z[0],
            cov[0],
        )
    return best


def span_trim(text, a, b):
    """Trailing sentence punctuation the model copied along ("No!" -> "No")."""
    while b - a > 1 and text[b - 1] in ".,;:!?" and not text[b - 2].isdigit():
        b -= 1
    return b


def span_windows(text, reserve=0):
    """(start, end) windows whose token ids fit one read, split at sentence
    ends and newlines with one sentence of overlap. The whole text when it
    fits."""
    budget = SPAN_ID_CAP - 4 - reserve
    if len(span_pieces(text)) <= budget:
        return [(0, len(text))]
    units = [
        (m.start(), m.end())
        for m in re.finditer(r"[^.!?\n]*[.!?\n]+\s*|[^.!?\n]+$", text)
        if text[m.start() : m.end()].strip()
    ]
    split = []
    for s_, e in units:
        if len(span_pieces(text[s_:e])) <= budget:
            split.append((s_, e))
            continue
        cur = s_
        for m in re.finditer(r"\S+\s*", text[s_:e]):
            if (
                len(span_pieces(text[cur : s_ + m.end()])) > budget
                and s_ + m.start() > cur
            ):
                split.append((cur, s_ + m.start()))
                cur = s_ + m.start()
        split.append((cur, e))
    windows, cur = [], []
    for s_, e in split:
        if cur and len(span_pieces(text[cur[0][0] : e])) > budget:
            windows.append((cur[0][0], cur[-1][1]))
            cur = [cur[-1]] if len(span_pieces(text[cur[-1][0] : e])) <= budget else []
        cur.append((s_, e))
    if cur:
        windows.append((cur[0][0], cur[-1][1]))
    return windows


def span_context(schema):
    ins = schema.get("instructions")
    return ("\n\n" + str(ins).strip()) if ins else ""


def span_choose(schema, q, state_content, candidates, seed):
    """A one-step choice over candidate strings with the whole state in
    view: the verify step when windows disagree. Returns (index, probs)."""
    letters = [chr(ord("A") + i) for i in range(len(candidates))]
    options = "\n".join(f"  {ltr}: {c}" for ltr, c in zip(letters, candidates))
    sys_text = SPAN_CHOOSE_SYSTEM.format(
        q=q["instructions"], context=span_context(schema), options=options
    )
    ids = [enc(" " + ltr)[0] for ltr in letters]
    body = enc("answer:") + [PAD]
    rows, _, _ = span_read_canvas(
        sys_text,
        state_content,
        body,
        {len(body) - 1},
        set(ids) | {TURN_CLOSE, EOS, NL},
        seed,
        1,
    )
    dist = slot_distribution(rows[-1], ids)
    probs = dict(zip(candidates, dist["probs"]))
    return max(range(len(candidates)), key=lambda i: dist["probs"][i]), probs


def read_span_window(schema, q, sys_text, state_content, text, window, seed):
    """One span question in one window: up to SPAN_MAX_READS reads with
    successive seeds until the answer clears SPAN_MIN_CONF. A rare bad read
    places the end early and shows up as low confidence."""
    wtext = text[window[0] : window[1]]
    pieces = span_pieces(wtext)
    none_id = enc(" none")[0]
    allowed = set(pieces) | {TURN_CLOSE, EOS, NL, none_id}
    head = enc("answer:")
    region = max(
        1, min(q["span"]["max_tokens"], CANVAS_LEN - len(SCAFFOLD) - len(head) - 1)
    )
    body = head + [PAD] * region
    free = set(range(len(head), len(body)))
    steps = max(SPAN_STEPS, schema["steps"])
    best, reads, rows_used = None, 0, 0
    for i in range(SPAN_MAX_READS):
        rows, _, width = span_read_canvas(
            sys_text, state_content, body, free, allowed, seed + i, steps
        )
        reads += 1
        rows_used += width
        _, a, b, conf, cov = span_decode(wtext, rows[len(head) :], pieces, none_id)
        if a is None:
            ans = {
                "type": "span",
                "found": False,
                "text": None,
                "start": None,
                "end": None,
            }
        else:
            b = span_trim(wtext, a, b)
            ans = {
                "type": "span",
                "found": True,
                "text": wtext[a:b],
                "start": a + window[0],
                "end": b + window[0],
            }
        ans["confidence"] = round(conf, 4)
        ans["coverage"] = round(cov, 4)
        if best is None or ans["confidence"] > best["confidence"]:
            best = ans
        if ans["confidence"] >= SPAN_MIN_CONF:
            break
    best["reads"] = reads
    return best, rows_used


def read_span(schema, q, state_content, seed):
    """One span question: the whole text in the prompt, the read restricted
    to each window's ids, windows in parallel, and one verify choice when
    they disagree."""
    text = span_source(state_content, schema)
    sys_text = SPAN_SYSTEM.format(q=q["instructions"], context=span_context(schema))
    windows = span_windows(text)
    with ThreadPoolExecutor(max_workers=min(8, len(windows))) as ex:
        parts = list(
            ex.map(
                lambda w: read_span_window(
                    schema, q, sys_text, state_content, text, w, seed
                ),
                windows,
            )
        )
    rows = sum(r for _, r in parts)
    found = [a for a, _ in parts if a["found"]]
    if not found:
        a = max((a for a, _ in parts), key=lambda a: a["confidence"] * a["coverage"])
    else:
        found.sort(key=lambda a: -a["confidence"] * a["coverage"])
        distinct = []
        for a in found:
            if a["text"] not in [d["text"] for d in distinct]:
                distinct.append(a)
        a = distinct[0]
        if len(distinct) > 1:
            i, probs = span_choose(
                schema, q, state_content, [d["text"] for d in distinct[:8]], seed
            )
            a = distinct[i]
            a["resolved"] = probs
            a["reads"] += 1
    a["reads"] = sum(p["reads"] for p, _ in parts) + (1 if "resolved" in a else 0)
    if len(windows) > 1:
        a["windows"] = len(windows)
    return a, rows


def read_spans_window(schema, q, sys_text, state_content, text, window, seed):
    """A list of spans in one window: one read writes the values one per
    line, each line is decoded as a span whose start follows the previous
    item, so repeated mentions map to successive occurrences."""
    wtext = text[window[0] : window[1]]
    pieces = span_pieces(wtext)
    none_id = enc(" none")[0]
    allowed = set(pieces) | {TURN_CLOSE, EOS, NL, none_id}
    head = enc("answer:")
    region = max(
        1, min(q["span"]["max_tokens"], CANVAS_LEN - len(SCAFFOLD) - len(head) - 1)
    )
    body = head + [PAD] * region
    free = set(range(len(head), len(body)))
    steps = max(SPAN_STEPS, schema["steps"])
    rows, emitted, width = span_read_canvas(
        sys_text, state_content, body, free, allowed, seed, steps
    )
    region_rows, region_em = rows[len(head) :], emitted[len(head) :]
    items, seg_start, min_start = [], 0, 0
    for k, t in enumerate(region_em + [TURN_CLOSE]):
        if t not in (TURN_CLOSE, EOS, NL) and t != PAD:
            continue
        # the model repeats an "answer:" label on later lines: its word is
        # outside the allowed set, its colon may not be (the text can hold
        # "02:14"), so skip unreadable tokens and a colon after them
        if seg_start < k and region_em[seg_start] not in allowed:
            while seg_start < k and region_em[seg_start] not in allowed:
                seg_start += 1
            if seg_start < k and pieces.get(region_em[seg_start], "").strip() == ":":
                seg_start += 1
        seg = region_rows[seg_start : k + 1]
        if k > seg_start and seg:
            _, a, b, conf, cov = span_decode(wtext, seg, pieces, none_id)
            dup = False
            if a is not None and a < min_start:
                if wtext.find(wtext[a:b], min_start) != -1:
                    _, a, b, conf, cov = span_decode(
                        wtext, seg, pieces, none_id, min_start=min_start
                    )
                else:
                    a, dup = None, True  # the model repeated a line it wrote
            if a is not None:
                b = span_trim(wtext, a, b)
                items.append(
                    {
                        "text": wtext[a:b],
                        "start": a + window[0],
                        "end": b + window[0],
                        "confidence": round(conf, 4),
                        "coverage": round(cov, 4),
                    }
                )
                min_start = b
            elif not items and not dup:
                break  # the first line is none
        seg_start = k + 1
        if t != NL or len(items) >= q["span"]["max_items"]:
            break
    return items, width


def read_spans(schema, q, state_content, seed):
    """Every span of a kind: windows and SPAN_PASSES seeds in parallel,
    merged by offset, low-confidence lines dropped."""
    text = span_source(state_content, schema)
    sys_text = SPAN_LIST_SYSTEM.format(
        q=q["instructions"], context=span_context(schema)
    )
    windows = span_windows(text)
    jobs = [(w, seed + p) for p in range(SPAN_PASSES) for w in windows]
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        parts = list(
            ex.map(
                lambda j: read_spans_window(
                    schema, q, sys_text, state_content, text, j[0], j[1]
                ),
                jobs,
            )
        )
    items = []
    for part, _ in parts:
        for it in part:
            if it["confidence"] < SPAN_MIN_CONF:
                continue
            same = [
                o for o in items if it["start"] < o["end"] and it["end"] > o["start"]
            ]
            if same:
                if it["confidence"] > same[0]["confidence"]:
                    same[0].update(it)
                continue
            items.append(it)
    items.sort(key=lambda i: i["start"])
    a = {"type": "spans", "found": bool(items), "items": items, "reads": len(jobs)}
    if len(windows) > 1:
        a["windows"] = len(windows)
    return a, sum(r for _, r in parts)


def decide_spans(schema, span_qs, state_content, seed):
    """Every span question of a decision, in parallel."""
    started = time.time()
    if not span_source(state_content, schema).strip():
        raise SchemaError("a span question needs a text state to point into")

    def run(iq):
        i, q = iq
        fn = read_spans if q["type"] == "spans" else read_span
        return fn(schema, q, state_content, seed + 104729 * (i + 1))

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(span_qs))) as ex:
        results = list(ex.map(run, enumerate(span_qs)))
    answers = {q["id"]: a for q, (a, _) in zip(span_qs, results)}
    diag = {
        "steps": max(SPAN_STEPS, schema["steps"]),
        "reads": sum(a["reads"] for a, _ in results),
        "rows": sum(r for _, r in results),
        "source": "state" if isinstance(state_content, str) else "text part",
        "ms": (time.time() - started) * 1e3,
    }
    return answers, diag


def decide_group(schema, sys_text, state_content, seed, prefix=None, lead=""):
    """Single-pass read; with DJEV_ADAPTIVE_THINK, uncertain questions are re-answered after a thought."""
    out, n_tok = _decide_group(schema, sys_text, state_content, seed, prefix, lead)
    if not (ADAPT_GATES and schema.get("adaptive", True) and not schema["think"] and prefix is None
            and isinstance(state_content, str)):
        return out, n_tok
    if ADAPT_MIN_TOKENS and (out["diagnostics"].get("prompt_tokens") or 0) < ADAPT_MIN_TOKENS:
        return out, n_tok
    if ADAPT_MAX_INFLIGHT and INFLIGHT[0] > ADAPT_MAX_INFLIGHT and not schema.get("adaptive_force"):  # djev-thinking-mode
        return out, n_tok
    if ADAPT_MAX_Q and len(schema["questions"]) > ADAPT_MAX_Q:
        return out, n_tok
    raw = out["diagnostics"]["raw_top"]
    low = [q["id"] for q in schema["questions"] if raw[q["id"]] < ADAPT_GATES.get(q["type"], 0.0)]
    if not low:
        return out, n_tok
    started = time.time()
    M_THINK.inc()
    try:
        out2, n2 = _decide_group(dict(schema, think=schema.get("adaptive_budget") or ADAPT_BUDGET), sys_text, state_content, seed, None, lead)  # djev-thinking-mode
    except SchemaError:
        return out, n_tok  # no room for a thought: keep the single-pass answers
    for qid in low:
        out["answers"][qid] = out2["answers"][qid]
    out["diagnostics"]["adaptive_think"] = {
        "questions": low,
        "ms": (time.time() - started) * 1e3,
        "thought_tokens": (out2["diagnostics"].get("thought") or {}).get("tokens"),
    }
    return out, n_tok + n2


def _decide_group(schema, sys_text, state_content, seed, prefix=None, lead=""):
    started = time.time()
    thought = None
    head = SCAFFOLD if prefix is None else []
    thinking = False
    if prefix is None and schema["think"]:
        if isinstance(state_content, str):
            prefix, thought = think(sys_text, state_content, schema["think"])
            head = []
        else:
            # With images the model writes the thought with the image in view, and
            # the server seeds it into the canvas ahead of the answer, so the read
            # keeps the image. The canvas bounds the thought.
            answer = enc(
                answer_text(
                    schema["questions"],
                    [0] * len(schema["questions"]),
                    schema.get("format", "lines"),
                )
            )
            fits = CANVAS_LEN - 1 - len(THOUGHT_OPEN) - len(THOUGHT_CLOSE) - len(answer)
            if fits < 8:
                raise SchemaError(
                    f"think: the canvas leaves {fits} rows for a thought "
                    "beside this template"
                )
            ids, thought = think_chat(
                sys_text, state_content, min(schema["think"], fits)
            )
            thought["budget"] = min(schema["think"], fits)
            head = THOUGHT_OPEN + ids + THOUGHT_CLOSE
            thinking = True
    template, slots = template_for(schema, head, lead)
    policy = schema["policy"]
    if policy["mode"] == "fixed":
        reads, usages = read_many(
            schema,
            template,
            slots,
            sys_text,
            state_content,
            seed,
            policy["n"],
            prefix,
            thinking,
            engine=True,
        )
        extended = None
        first_entropy = None
    else:
        reads, usages = read_many(
            schema, template, slots, sys_text, state_content, seed, 1, prefix, thinking
        )
        first_entropy = {
            q["id"]: r["entropy"] for q, r in zip(schema["questions"], reads[0])
        }
        extended = (
            max(first_entropy.values()) > policy["threshold"] and policy["max"] > 1
        )
        if extended:
            more, more_usages = read_many(
                schema,
                template,
                slots,
                sys_text,
                state_content,
                seed + 1,
                policy["max"] - 1,
                prefix,
                thinking,
            )
            reads += more
            usages += more_usages
    prompt_tokens = next(
        (u["prompt_tokens"] for u in usages if u and u.get("prompt_tokens")), None
    )
    elapsed_ms = (time.time() - started) * 1e3

    answers = {}
    diag_q = {}
    raw_top = {}
    n = len(reads)
    for qi, q in enumerate(schema["questions"]):
        per = [r[qi]["probs"] for r in reads]
        mean = [sum(p[i] for p in per) / n for i in range(len(q["labels"]))]
        raw_top[q["id"]] = max(mean)
        t = TEMPS.get(q["type"], 1.0)
        if t != 1.0:
            w = [max(m, 1e-12) ** (1.0 / t) for m in mean]
            mean = [x / sum(w) for x in w]
        if NOUL_COMMIT and q["type"] == "noul" and 1 - NOUL_COMMIT < mean[0] < NOUL_COMMIT:
            y = NOUL_COMMIT + 0.001 if mean[0] >= 0.5 else 1 - NOUL_COMMIT - 0.001
            mean = [y, 1 - y]
        top = max(range(len(mean)), key=lambda i: mean[i])
        a = {
            "type": q["type"],
            "label": q["labels"][top],
            "confidence": mean[top],
            "probabilities": {c[0]: m for c, m in zip(q["choices"], mean)},
        }
        if q["type"] == "noul":
            a["noul"] = mean[0]
        elif q["type"] == "choice":
            a["choice"] = q["choices"][top][0]
        else:
            a["score"] = sum((i + 1) * m for i, m in enumerate(mean))
            a["level"] = q["choices"][top][0]
        if n > 1:
            var = sum((p[top] - mean[top]) ** 2 for p in per) / (n - 1)
            a["stderr"] = (var / n) ** 0.5
            a["agreement"] = (
                sum(1 for p in per if max(range(len(p)), key=lambda i: p[i]) == top) / n
            )
        answers[q["id"]] = a
        diag_q[q["id"]] = {
            "pos": slots[qi]["pos"],
            "entropy": [r[qi]["entropy"] for r in reads],
            "label_mass": reads[0][qi]["label_mass"],
            "argmax_is_label": reads[0][qi]["argmax_is_label"],
        }
    tops = [
        {
            q["id"]: [
                q["labels"][
                    max(range(len(r[qi]["probs"])), key=lambda i: r[qi]["probs"][i])
                ],
                max(r[qi]["probs"]),
                r[qi]["entropy"],
            ]
            for qi, q in enumerate(schema["questions"])
        }
        for r in reads
    ]
    return {
        "answers": answers,
        "diagnostics": {
            "steps": schema["steps"],
            "samples": {
                "n": n,
                "tops": tops,
                "policy": dict(
                    policy, extended=extended, first_read_entropy=first_entropy
                ),
            },
            "timing": {
                "total_ms": elapsed_ms,
                "reads": n,
                "requests": sum(1 for u in usages if u is not None),
            },
            "thought": thought,
            "raw_top": raw_top,
            "prompt_tokens": prompt_tokens,
            "questions": diag_q,
            "engine": "vllm",
        },
    }, len(template) + 1 + (thought["tokens"] if thought else 0)


# ----------------------------------------------------------------------------
# Jev's contract
# ----------------------------------------------------------------------------

JEV_EXTENSIONS = (
    "instructions",
    "samples",
    "auto_max",
    "auto_threshold",
    "steps",
    "think",
    "ask",
    "chunk_rows",
    "chunk_prompt",
    "sequential",
    "adaptive",
)


def jev_schema(body):
    """This server's schema from a Jev request body."""
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise SchemaError("questions: needs a non-empty map of id -> question")
    out = []
    for qid, q in qs.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question {qid!r}: must be an object")
        kind, crit, ins = q.get("type"), q.get("criteria"), q.get("instructions", "")
        item = {
            "id": qid,
            "type": kind,
            "instructions": ins if isinstance(ins, str) else json.dumps(ins),
        }
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: noul criteria must be an object with "
                    "true and false"
                )
            item["criteria"] = crit
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise SchemaError(
                    f"question {qid!r}: choice criteria must map option names "
                    "to descriptions"
                )
            item["options"] = [
                {"name": str(n), "description": d} for n, d in crit.items()
            ]
        elif kind == "score":
            if not isinstance(crit, list):
                raise SchemaError(
                    f"question {qid!r}: score criteria must be an ordered list "
                    "of levels"
                )
            item["levels"] = crit
        elif kind in ("span", "spans"):
            if crit is not None and not isinstance(crit, dict):
                raise SchemaError(
                    f"question {qid!r}: {kind} criteria must be an object, "
                    'e.g. {"max_tokens": 24}'
                )
            for key in ("max_tokens", "max_items"):
                if crit and key in crit:
                    item[key] = crit[key]
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        for key in ("depends_on", "ask_if", "alone"):
            if key in q:
                item[key] = q[key]
        out.append(item)
    mode, budget = body.get("thinking"), body.get("thinking_budget")  # djev-thinking-mode
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or not 16 <= budget <= 4096):
        raise SchemaError("thinking_budget: thought tokens, 16 to 4096")
    if mode is not None:
        if mode not in ("off", "auto", "auto_forced", "always"):
            raise SchemaError('thinking: one of "off", "auto", "auto_forced", "always"')
        body = dict(body)
        body["adaptive"] = mode in ("auto", "auto_forced")
        body["think"] = (budget or ADAPT_BUDGET or 512) if mode == "always" else 0
    schema = {k: body[k] for k in JEV_EXTENSIONS if k in body}
    schema["questions"] = out
    parsed = parse_schema(schema)
    parsed["adaptive_force"] = mode == "auto_forced"
    parsed["adaptive_budget"] = budget if mode in ("auto", "auto_forced") else None
    return parsed


def image_part(content_type, data):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{content_type};base64," + base64.b64encode(data).decode()
        },
    }


def jev_images(value):
    """Image parts from the body's "images": data URLs, or objects with
    content_type and base64."""
    parts = []
    for i, im in enumerate(value or []):
        if isinstance(im, str) and im.startswith("data:image/"):
            parts.append({"type": "image_url", "image_url": {"url": im}})
        elif (
            isinstance(im, dict)
            and str(im.get("content_type", "")).startswith("image/")
            and isinstance(im.get("base64"), str)
        ):
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{im['content_type']};base64,{im['base64']}"
                    },
                }
            )
        else:
            raise SchemaError(
                f"images[{i}]: a data:image/... URL or an object with "
                "content_type and base64"
            )
    return parts


def index_arrays(x):
    """Number the objects of long arrays ("_index": i, first key) so a question can point at one record."""
    if isinstance(x, list):
        items = [index_arrays(e) for e in x]
        if len(items) >= JSON_INDEX and all(isinstance(e, dict) for e in items):
            items = [{"_index": i, **e} for i, e in enumerate(items)]
        return items
    if isinstance(x, dict):
        return {k: index_arrays(v) for k, v in x.items()}
    return x

def upstream_client_error(detail):
    """A readable reason for an upstream 400."""
    low = detail.lower()
    if "context length" in low or "max_model_len" in low or "too long" in low or "maximum" in low and "tokens" in low:
        return "state too long for this deployment's context window (" + detail[:200] + ")"
    if "image" in low or "multimodal" in low or "mm_" in low:
        return "images are not supported by this deployment (" + detail[:200] + ")"
    return "invalid request: " + detail[:300]


def jev_state(body, image_parts=()):
    """The user message: the state as text (as given, or as JSON), with any
    images ahead of it."""
    state = body.get("state")
    if state is None:
        raise SchemaError("state: required")
    # ensure_ascii=False: the model reads "°" as itself, not as \u00b0, so a
    # span can copy it.
    if JSON_INDEX and not isinstance(state, str):
        state = index_arrays(state)
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if not image_parts:
        return text
    return list(image_parts) + [{"type": "text", "text": text}]


def jev_answer(q, a):
    if a is None:
        return None
    if q["type"] == "noul":
        return {"type": "noul", "noul": a["noul"]}
    if q["type"] == "choice":
        return {
            "type": "choice",
            "choice": a["choice"],
            "probabilities": a["probabilities"],
            "confidence": a["confidence"],
        }
    if q["type"] in ("span", "spans"):
        return a
    names = [c[0] for c in q["choices"]]
    probs = {str(i): a["probabilities"][n] for i, n in enumerate(names)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs.values())),
        "legend": {str(i): n for i, n in enumerate(names)},
        "probabilities": probs,
        "confidence": a["confidence"],
    }


def chat_answer(q, a):
    """The chat endpoint's answer: the Jev shape plus a readable label (and, for noul, its probabilities)."""
    j = jev_answer(q, a)
    if j is None or q["type"] not in ("noul", "choice", "score"):
        return j
    j = dict(j)
    if q["type"] == "noul":
        j["label"] = "yes" if a["noul"] >= 0.5 else "no"
        j["confidence"] = a["confidence"]
        j["probabilities"] = a["probabilities"]
    elif q["type"] == "choice":
        j["label"] = a["choice"]
    else:
        j["label"] = a["level"]
    return j


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------


def answer_brief(v):
    """One word of an answer for the log line."""
    if v is None:
        return "skipped"
    if "label" in v:
        return v["label"]
    if "items" in v:
        return f"{len(v['items'])}spans"
    return repr(v.get("text")) if v.get("found") else "none"


def message_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


def is_structured(req):
    """True when a chat request is a structured read: its first message is a
    system (or developer) message whose text is a JSON object with a
    "questions" key. Anything else is ordinary chat for vLLM."""
    msgs = req.get("messages") or []
    if not msgs or not isinstance(msgs[0], dict):
        return False
    if msgs[0].get("role") not in ("system", "developer"):
        return False
    try:
        value = json.loads(message_text(msgs[0]))
    except ValueError:
        return False
    return isinstance(value, dict) and "questions" in value


class Server(ThreadingHTTPServer):
    # The default backlog of 5 resets connections when a client opens as many
    # at once as the engine serves sequences.
    request_queue_size = 256
    daemon_threads = True

    def server_bind(self):
        if os.environ.get("DJEV_REUSE_PORT") == "1":  # several processes share the port; the kernel spreads connections
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_response(self, code, message=None):
        self._djev_status = code
        return super().send_response(code, message)

    def _observe(self, method, fn):
        t0 = time.time()
        self._djev_status = 0
        try:
            return fn()
        finally:
            p = self.path.split("?", 1)[0]
            ep = p if p in _EP else ("/v1/other" if p.startswith("/v1/") else "other")
            M_REQ.labels(ep, str(self._djev_status)).inc()
            if method == "POST":
                M_LAT.labels(ep).observe(time.time() - t0)

    def do_POST(self):
        return self._observe("POST", self._do_post)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/metrics":
            return self._metrics()
        return self._observe("GET", self._do_get)

    def _metrics(self):
        """Prometheus scrape: djev_* plus vLLM's metrics; not served through the public ingress."""
        if self.headers.get("x-forwarded-for") or self.headers.get("x-real-ip"):
            return self._json(404, {"error": {"message": "unknown route"}})
        parts = [_p_latest(_P_REG)]
        try:
            with urllib.request.urlopen(ARGS.upstream.rstrip("/") + "/metrics", timeout=5) as r:
                parts.append(r.read())
        except OSError:
            pass
        body = b"".join(parts)
        self.send_response(200)
        self.send_header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        if isinstance(obj, dict) and not EXPOSE_DIAGNOSTICS:  # no internal state in client responses
            obj = {k: v for k, v in obj.items() if k != "diagnostics"}
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _do_get(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/models":
            return self._models()
        if ARGS.pages:
            return self._page()
        return self._json(404, {"error": {"message": "unknown route"}})

    def _page(self):
        """With --pages DIR, GET /name serves DIR/name.html and / serves
        DIR/index.html, for test pages that post to /v1/systemone."""
        name = self.path.split("?", 1)[0].strip("/") or "index"
        path = os.path.join(ARGS.pages, name + ".html")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not os.path.isfile(path):
            return self._json(404, {"error": {"message": "unknown route"}})
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _models(self):
        """Pass GET /v1/models through to vLLM, so this port lists the same
        served name as the upstream. An OpenAI router that discovers models by
        probing this route can then send /v1/systemone here by the body's
        "model"."""
        try:
            with urllib.request.urlopen(
                ARGS.upstream.rstrip("/") + "/v1/models", timeout=10
            ) as r:
                code, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, body = e.code, e.read()
        except OSError as e:
            return self._json(
                503, {"error": {"message": f"upstream unavailable: {e}"}}
            )
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request(self):
        """-> (body, image parts): a JSON body, or multipart/form-data with the
        JSON in a part named request and each image as a file part, in order."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        self._raw = raw  # kept so a request that is not ours can be relayed as sent
        ctype = self.headers.get("content-type", "")
        if not ctype.lower().startswith("multipart/form-data"):
            return json.loads(raw), []
        from email.parser import BytesParser
        from email.policy import HTTP

        msg = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
        )
        body, images = None, []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            data = part.get_payload(decode=True)
            if name == "request":
                body = json.loads(data)
            elif part.get_content_type().startswith("image/"):
                images.append(image_part(part.get_content_type(), data))
            else:
                raise ValueError(
                    f"part {name!r}: neither the request JSON nor an image"
                )
        if body is None:
            raise ValueError(
                "multipart needs a part named request holding the JSON body"
            )
        return body, images

    def _do_post(self):
        if API_KEY and self.headers.get("authorization", "") != f"Bearer {API_KEY}":
            return self._json(
                401,
                {
                    "error": {
                        "message": "missing or wrong bearer token",
                        "type": "authentication_error",
                    }
                },
            )
        if self.path == "/v1/raw/chat/completions":
            return self._raw_chat()
        try:
            req, images = self._read_request()
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": f"invalid body: {e}",
                        "type": "invalid_request_error",
                    }
                },
            )
        if self.path == "/v1/systemone":
            return self._systemone(req, images)
        if self.path == "/v1/chat/completions":
            if is_structured(req):
                return self._chat(req)
            return self._json(400, {"error": {"message": PLAIN_CHAT_MESSAGE, "type": "invalid_request_error",
                                              "code": "structured_request_required"}})
        if self.path.startswith("/v1/"):
            return self._relay(self.path, self._raw)
        return self._json(404, {"error": {"message": "unknown route"}})

    def _raw_chat(self):
        """Pass the body and status through to vLLM's chat completions."""
        raw = self.rfile.read(int(self.headers.get("content-length", "0")))
        return self._relay("/v1/chat/completions", raw)

    def _relay(self, path, raw):
        """Send a request to vLLM as it arrived and return vLLM's status,
        content type and body unchanged. The body is copied as it arrives, so a
        streamed (SSE) response streams; this handler speaks HTTP/1.0, so a
        response without a length ends when the connection closes."""
        req = urllib.request.Request(
            ARGS.upstream.rstrip("/") + path,
            data=raw,
            headers={
                "content-type": self.headers.get("content-type", "application/json")
            },
        )
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            r = e
        except OSError as e:
            return self._json(
                503, {"error": {"message": f"upstream unavailable: {e}"}}
            )
        with r:
            self.send_response(r.code)
            self.send_header(
                "content-type", r.headers.get("content-type", "application/json")
            )
            if r.headers.get("content-length"):
                self.send_header("content-length", r.headers["content-length"])
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

    def _decide(self, schema, state, seed):
        """-> (status, body) with the error body already shaped."""
        with INFLIGHT_LOCK:
            INFLIGHT[0] += 1
        try:
            return self._decide_inner(schema, state, seed)
        finally:
            with INFLIGHT_LOCK:
                INFLIGHT[0] -= 1

    def _decide_inner(self, schema, state, seed):
        try:
            return 200, decide(schema, state, seed)
        except SchemaError as e:
            return 422, {"error": {"message": str(e), "type": "validation_error"}}
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode(errors="replace")
            if e.code == 400:
                return 400, {"error": {"message": upstream_client_error(detail), "type": "invalid_request_error"}}
            return 502, {
                "error": {
                    "message": f"upstream {e.code}: {detail[:300]}",
                    "type": "server_error",
                }
            }
        except Exception as e:
            return 500, {"error": {"message": repr(e), "type": "server_error"}}

    def _systemone(self, req, images):
        many = [k for k, q in (req.get("questions") or {}).items() if isinstance(q, dict) and q.get("type") == "choice"
                and isinstance(q.get("criteria"), dict) and len(q["criteria"]) > 26] if MANY_OPTIONS else []
        if many:
            return self._systemone_many(req, images, many)
        return self._systemone_one(req, images)

    def _systemone_core(self, req, images):
        """-> (code, body) for one /v1/systemone request, without writing it."""
        try:
            schema = jev_schema(req)
            state = jev_state(req, images + jev_images(req.get("images")))
            schema["span_text"] = span_text_of(req.get("state"))
        except SchemaError as e:
            return 422, {"error": {"message": str(e), "type": "validation_error"}}
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            return code, result
        body, completion_tokens = result
        answers = {q["id"]: jev_answer(q, body["answers"][q["id"]]) for q in schema["questions"]}
        usage = {"input_tokens": body["diagnostics"].get("prompt_tokens") or (
            len(chat_prompt_ids(system_text(schema), state)) if isinstance(state, str) else 0), "output_tokens": completion_tokens}
        return 200, {"model": ARGS.model, "answers": answers, "usage": usage, "diagnostics": body["diagnostics"]}

    def _systemone_many(self, req, images, many):
        chunk, top, eps = MANY_OPTIONS.get("chunk", 20), MANY_OPTIONS.get("top", 3), MANY_OPTIONS.get("eps", 0.02)
        t0 = time.time()
        qs1, parts = {}, {}
        for k, q in req["questions"].items():
            if k not in many:
                qs1[k] = q
                continue
            items = list(q["criteria"].items())
            m = -(-len(items) // chunk)
            size = -(-len(items) // m)
            parts[k] = []
            for j in range(m):
                sub = items[j * size:(j + 1) * size]
                if not sub:
                    continue
                if len(sub) == 1:  # a single leftover option rides with the previous chunk
                    parts[k][-1][1].update(sub)
                    qs1[parts[k][-1][0]]["criteria"] = parts[k][-1][1]
                    continue
                sid = f"{k}__p{j}"
                parts[k].append((sid, dict(sub)))
                qs1[sid] = {**q, "criteria": dict(sub)}
        req1 = {**req, "questions": qs1}
        if not MANY_OPTIONS.get("think1", 1):
            req1["adaptive"] = False
        code, r1 = self._systemone_core(req1, images)
        if code != 200:
            return self._json(code, r1)
        answers = {k: r1["answers"][k] for k in req["questions"] if k not in many}
        usage = dict(r1["usage"])
        diag = {"many_options": {}, "round1": r1["diagnostics"]}
        for k in many:
            p1 = {}
            finalists = []
            for sid, crit in parts[k]:
                pr = r1["answers"][sid]["probabilities"]
                p1.update(pr)
                finalists += sorted(crit, key=lambda n: -pr[n])[:top]
            if len(finalists) > 26:
                finalists = sorted(finalists, key=lambda n: -p1[n])[:26]
            order = [n for n in req["questions"][k]["criteria"] if n in set(finalists)]
            q2 = {**req["questions"][k], "criteria": {n: req["questions"][k]["criteria"][n] for n in order}}
            req2 = {**req, "questions": {k: q2}}
            if not MANY_OPTIONS.get("think2", 1):
                req2["adaptive"] = False
            code, r2 = self._systemone_core(req2, images)
            if code != 200:
                return self._json(code, r2)
            p2 = r2["answers"][k]["probabilities"]
            rest = [n for n in req["questions"][k]["criteria"] if n not in p2]
            rest_mass = sum(p1[n] for n in rest) or 1.0
            probs = {}
            for n in req["questions"][k]["criteria"]:
                probs[n] = (1 - eps) * p2[n] if n in p2 else eps * p1[n] / rest_mass
            z = sum(probs.values())
            probs = {n: v / z for n, v in probs.items()}
            best = max(probs, key=probs.get)
            answers[k] = {"type": "choice", "choice": best, "probabilities": probs, "confidence": probs[best]}
            usage["input_tokens"] += r2["usage"]["input_tokens"]
            usage["output_tokens"] += r2["usage"]["output_tokens"]
            M_MANY.inc()
            diag["many_options"][k] = {"options": len(probs), "chunks": len(parts[k]), "finalists": len(order)}
            diag.setdefault("round2", {})[k] = r2["diagnostics"]
        diag["timing"] = {"total_ms": (time.time() - t0) * 1000}
        brief = " ".join(str(k) + "=" + str(answers[k]["choice"]) for k in many)
        print(f"systemone-many: {brief} {diag['timing']['total_ms']:.0f}ms", flush=True)
        self._json(200, {"model": ARGS.model, "answers": answers, "usage": usage, "diagnostics": diag})

    def _systemone_one(self, req, images):
        try:
            schema = jev_schema(req)
            state = jev_state(req, images + jev_images(req.get("images")))
            schema["span_text"] = span_text_of(req.get("state"))
        except SchemaError as e:
            return self._json(
                422, {"error": {"message": str(e), "type": "validation_error"}}
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            return self._json(code, result)
        body, completion_tokens = result
        answers = {
            q["id"]: jev_answer(q, body["answers"][q["id"]])
            for q in schema["questions"]
        }
        labels = " ".join(f"{k}={answer_brief(v)}" for k, v in body["answers"].items())
        print(
            f"systemone: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "model": ARGS.model,
                "answers": answers,
                # vLLM's prompt count when the reads reported one, since it covers
                # images. Otherwise the tokenizer's count of the text prompt.
                "usage": {
                    "input_tokens": body["diagnostics"].get("prompt_tokens")
                    or (
                        len(chat_prompt_ids(system_text(schema), state))
                        if isinstance(state, str)
                        else 0
                    ),
                    "output_tokens": completion_tokens,
                },
                "diagnostics": body["diagnostics"],
            },
        )

    def _chat(self, req):
        msgs = req.get("messages") or []
        if (
            len(msgs) != 2
            or msgs[0].get("role") not in ("system", "developer")
            or msgs[1].get("role") != "user"
        ):
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "a structured request is exactly two messages: the "
                            "schema (system) and the state JSON (user)"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        try:
            schema_value = json.loads(message_text(msgs[0]))
            schema = parse_schema(schema_value)
            content = msgs[1].get("content", "")
            has_image = isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("image_url", "image")
                for p in content
            )
            if has_image:
                # image parts pass through to vLLM unchanged, with text parts as context
                state = content
            else:
                state = message_text(msgs[1]).strip()
                schema["span_text"] = span_text_of(json.loads(state))
        except SchemaError as e:
            return self._json(
                400, {"error": {"message": str(e), "type": "invalid_request_error"}}
            )
        except Exception as e:
            return self._json(
                400,
                {
                    "error": {
                        "message": (
                            "system must be a JSON question schema and user "
                            f"must be JSON state or image parts: {e}"
                        ),
                        "type": "invalid_request_error",
                    }
                },
            )
        code, result = self._decide(schema, state, int(req.get("seed", 42)))
        if code != 200:
            if code == 422:
                result["error"]["type"] = "invalid_request_error"
                code = 400
            return self._json(code, result)
        body, completion_tokens = result
        payload = {
            "answers": {
                q["id"]: chat_answer(q, body["answers"].get(q["id"]))
                for q in schema["questions"]
            }
        }
        if req.get("diagnostics") and EXPOSE_DIAGNOSTICS:  # only when the server opts in
            payload["diagnostics"] = body["diagnostics"]
        content = json.dumps(payload, ensure_ascii=False)
        prompt_tokens = body["diagnostics"].get("prompt_tokens") or (
            len(chat_prompt_ids(system_text(schema), state)) if isinstance(state, str) else 0
        )
        labels = " ".join(f"{k}={answer_brief(v)}" for k, v in body["answers"].items())
        print(
            f"structured: {labels} "
            f"reads={body['diagnostics']['timing']['reads']} "
            f"{body['diagnostics']['timing']['total_ms']:.0f}ms",
            flush=True,
        )
        self._json(
            200,
            {
                "id": f"chatcmpl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.get("model", "super-decider"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )


def self_signed(cert_dir):
    """Paths of a self-signed certificate and key in cert_dir, made with
    openssl on first use."""
    os.makedirs(cert_dir, exist_ok=True)
    cert, key = os.path.join(cert_dir, "djev.crt"), os.path.join(cert_dir, "djev.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "3650",
                "-subj",
                "/CN=djev",
                "-keyout",
                key,
                "-out",
                cert,
            ],
            check=True,
            capture_output=True,
        )
    return cert, key


def serve_tls(host, port, cert_dir):
    cert, key = self_signed(cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = Server((host, port), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    global ARGS, CANVAS_LEN, CANVAS_STEP, MAX_SAMPLES
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="http://127.0.0.1:8010")
    p.add_argument("--model", default="super-decider")
    p.add_argument("--tokenizer", default="/models/super-decider", help="HF id or local path")
    p.add_argument("--canvas", type=int, default=64, help="the served canvas length")
    p.add_argument(
        "--canvas-step",
        type=int,
        default=16,
        help="request widths round up to a multiple of this",
    )
    p.add_argument(
        "--constrained",
        action="store_true",
        help="read over the labels only (diffusion_constrained). Needs an "
        "engine with vllm-project/vllm#58216",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="cap on reads per question. A fixed count must also fit the "
        "engine's diffusion_config.max_samples",
    )
    p.add_argument(
        "--engine-samples",
        action="store_true",
        help="send a fixed sample count as one diffusion_samples request "
        "instead of one request per draw. Needs an engine with "
        "vllm-project/vllm#58438",
    )
    p.add_argument(
        "--tls-port", type=int, default=0, help="also listen with HTTPS here (0 = off)"
    )
    p.add_argument(
        "--pages",
        default="",
        help="serve DIR/name.html at GET /name and DIR/index.html at / "
        "(off when empty)",
    )
    p.add_argument(
        "--cert-dir",
        default=os.path.expanduser("~/.cache/djev"),
        help="directory for the self-signed certificate",
    )
    ARGS = p.parse_args()
    CANVAS_LEN = ARGS.canvas
    CANVAS_STEP = ARGS.canvas_step
    MAX_SAMPLES = max(1, ARGS.max_samples)
    init_tokenizer(AutoTokenizer.from_pretrained(ARGS.tokenizer))
    if ARGS.tls_port:
        serve_tls(ARGS.host, ARGS.tls_port, ARGS.cert_dir)
        print(
            f"structured server https on {ARGS.host}:{ARGS.tls_port} (self-signed)",
            flush=True,
        )
    print(
        f"structured server on {ARGS.host}:{ARGS.port} -> {ARGS.upstream} "
        f"(canvas {CANVAS_LEN})",
        flush=True,
    )
    Server((ARGS.host, ARGS.port), Handler).serve_forever()


# djev-bidir-server
BIDIR = os.environ.get("DJEV_BIDIR", "")
BIDIR_MAXTOK = int(os.environ.get("DJEV_BIDIR_MAXTOK", "0") or 0)
BIDIR_MAXPROMPT = int(os.environ.get("DJEV_BIDIR_MAXPROMPT", "8192") or 0)  # longer prompts are read causally: with ranges the
# attention kernel loses its causal tile skip, and on 30-40K-token prompts (BRIGHT/ToolRet) that is a 2.7x slowdown
BIDIR_SALT = "djev-bq-" + BIDIR
_BIDIR_REST = {}   # hash(user content) -> length of the questions block text at its end
_BIDIR_IDS = {}    # hash(tuple(chat prompt ids)) -> (a, b)
_BIDIR_STATS = {"chat": 0, "completions": 0, "none": 0, "capped": 0, "over_2496": 0}

_chat_messages_causal = chat_messages
def chat_messages(sys_text, state_content):
    msgs = _chat_messages_causal(sys_text, state_content)
    if BIDIR and isinstance(state_content, str) and len(msgs) == 2 and msgs[0]["content"] == DECISION_PREAMBLE.rstrip("\n"):
        user = msgs[1]["content"]
        if isinstance(user, str) and user.startswith(state_content + "\n\n"):
            if len(_BIDIR_REST) > 20000:
                _BIDIR_REST.clear()
            _BIDIR_REST[hash(user)] = len(user) - len(state_content) - 2
    return msgs

def _bidir_range(messages, thinking):
    user = messages[-1]["content"]
    n = _BIDIR_REST.get(hash(user)) if isinstance(user, str) else None
    if not n:
        return None, None
    text = TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    enc = TOK(text, add_special_tokens=False, return_offsets_mapping=True)
    if BIDIR_MAXPROMPT and len(enc["input_ids"]) > BIDIR_MAXPROMPT:
        _BIDIR_STATS["long_causal"] = _BIDIR_STATS.get("long_causal", 0) + 1
        return enc["input_ids"], None
    e = text.rfind(user) + len(user); a = e - n
    idx = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x >= a and y <= e and y > x]
    if not idx or idx[-1] - idx[0] + 1 != len(idx):
        return enc["input_ids"], None
    rg = (idx[0], idx[-1] + 1)
    if BIDIR_MAXTOK and rg[1] - rg[0] > BIDIR_MAXTOK:
        _BIDIR_STATS["capped"] += 1
        return enc["input_ids"], None
    if BIDIR == "all":
        return enc["input_ids"], [(0, len(enc["input_ids"]))]
    if BIDIR == "qsep":  # one range per question (question line + its options), causal across questions
        rest = user[-n:]
        starts = [m.start() for m in re.finditer(r"(?m)^Question \S+: ", rest)]
        endq = rest.rfind("\n\nReply with")
        endq = endq if endq > 0 else len(rest)
        out = []
        for k, s0 in enumerate(starts):
            c0, c1 = a + s0, a + (starts[k + 1] - 2 if k + 1 < len(starts) else endq)
            jdx = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x >= c0 and y <= c1 and y > x]
            if jdx and jdx[-1] - jdx[0] + 1 == len(jdx) and jdx[-1] > jdx[0]:
                out.append((jdx[0], jdx[-1] + 1))
        return enc["input_ids"], (out or None)
    if BIDIR == "qdoc":
        d0 = text.rfind(user)  # user = state + "\n\n" + questions block; the state is the document block
        d1 = d0 + len(user) - n - 2
        jdx = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x >= d0 and y <= d1 and y > x]
        if jdx and jdx[-1] - jdx[0] + 1 == len(jdx) and jdx[-1] < rg[0]:
            return enc["input_ids"], [(jdx[0], jdx[-1] + 1), rg]
    return enc["input_ids"], [rg]

def _bidir_attach(body, rg, kind):
    if rg is None:
        _BIDIR_STATS["none"] += 1
        return body
    _BIDIR_STATS[kind] += 1
    if max(r[1] - r[0] for r in rg) > 2496:  # vLLM's default prefill chunk here; longer blocks get split across chunks
        _BIDIR_STATS["over_2496"] += 1
    if (_BIDIR_STATS["chat"] + _BIDIR_STATS["completions"]) % 500 == 0:
        print("djev-bidir stats", _BIDIR_STATS, flush=True)
    return dict(body, cache_salt=BIDIR_SALT, vllm_xargs=dict(body.get("vllm_xargs") or {}, djev_bidir=[int(x) for a, b in rg for x in (a, b)]))  # flat [a1, b1, a2, b2, ...]: vLLM rejects nested lists

_chat_prompt_ids_causal = chat_prompt_ids
def chat_prompt_ids(sys_text, state_text, thinking=False):
    ids = _chat_prompt_ids_causal(sys_text, state_text, thinking)
    if BIDIR:
        enc_ids, rg = _bidir_range(chat_messages(sys_text, state_text), thinking)
        if rg is not None and list(enc_ids) == ids:
            if len(_BIDIR_IDS) > 20000:
                _BIDIR_IDS.clear()
            _BIDIR_IDS[hash(tuple(ids))] = rg
    return ids

_upstream_chat_causal = upstream_chat
def upstream_chat(body, timeout=600):
    if BIDIR and isinstance(body.get("messages"), list) and body["messages"]:
        thinking = bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", False))
        _, rg = _bidir_range(body["messages"], thinking)
        body = _bidir_attach(body, rg, "chat")
    return _upstream_chat_causal(body, timeout)

BIDIR_THINK = os.environ.get("DJEV_BIDIR_THINK", "bidir")  # "causal": thought generation reads the prompt causally
_upstream_completions_causal = upstream_completions
def upstream_completions(body, timeout=600):
    pr = body.get("prompt")
    if BIDIR and BIDIR_THINK == "causal" and body.get("stop_token_ids") and not body.get("logprob_token_ids"):
        _BIDIR_STATS["think_causal"] = _BIDIR_STATS.get("think_causal", 0) + 1
        return _upstream_completions_causal(dict(body, cache_salt=BIDIR_SALT + "-thinkcausal"), timeout)
    if BIDIR and isinstance(pr, list) and pr and isinstance(pr[0], int):
        rg = None
        cuts = [i for i, t in enumerate(pr) if THOUGHT_OPEN and t == THOUGHT_OPEN[0]] + [len(pr)]
        for c in cuts:
            rg = _BIDIR_IDS.get(hash(tuple(pr[:c])))
            if rg is not None:
                break
        body = _bidir_attach(body, rg, "completions")
    return _upstream_completions_causal(body, timeout)



if __name__ == "__main__":
    main()

"""Warm a fresh djev server before it takes traffic: concurrent batches of 1..32 requests with 1/3/9 questions and
short/medium/long states, plus one request that thinks, so vLLM compiles its kernels (Triton JIT, CUDA graphs) for
the batch shapes real traffic uses. usage: warmup.py DJEV_PORT"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
P = sys.argv[1]
words = ("the team confirmed that the validator rewards will be distributed after the next epoch while users reported "
         "delays on withdrawals and asked for an update on the bridge status and the new release binary ").split()
def state(n): return " ".join(words[i % len(words)] for i in range(n))
def qs(k):
    base = {"q_choice": {"type": "choice", "instructions": "Main topic?", "criteria": {"a": "Rewards", "b": "Withdrawals", "c": "Releases", "d": "Other"}},
            "q_noul": {"type": "noul", "instructions": "Is a reply needed today?"},
            "q_score": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}
    out = {}
    for i in range(k):
        name, q = list(base.items())[i % 3]
        out[f"{name}_{i}"] = q
    return out
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{P}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "authorization": "Bearer " + os.environ.get("API_KEY", "local")})
    urllib.request.urlopen(req, timeout=600).read()
t0 = time.time()
for conc in (1, 2, 4, 8, 16, 32):
    bodies = [{"model": "super-decider", "state": state((40, 300, 1200)[i % 3]), "questions": qs((1, 3, 9)[i % 3])} for i in range(conc)]
    list(ThreadPoolExecutor(conc).map(post, bodies))
# label-set sizes: single choice questions with 2..26 options and multi-question unions, alone and in batches
opts = lambda k: {f"opt{j}": f"Option number {j}" for j in range(k)}
singles = [{"model": "super-decider", "state": state(120), "questions": {"q": {"type": "choice", "instructions": "Pick one", "criteria": opts(k)}}} for k in range(2, 27)]
for b in singles: post(b)
list(ThreadPoolExecutor(8).map(post, singles))
unions = [{"model": "super-decider", "state": state(200), "questions": {**{f"c{i}": {"type": "choice", "instructions": "Pick one", "criteria": opts(k)} for i, k in enumerate(ks)}, **qs(n)}}
          for ks, n in (((3, 10), 7), ((5, 12, 4), 4), ((26, 8), 2), ((20, 20), 3), ((6,), 9))]
for b in unions: post(b)
list(ThreadPoolExecutor(8).map(post, unions * 2))
# uniform-schema batches: vLLM's label-only output path runs only when every request in a step shares one label set
for sch in [b["questions"] for b in singles[::3]] + [b["questions"] for b in unions] + [qs(1), qs(3), qs(9)]:
    for conc in (4, 16):
        list(ThreadPoolExecutor(conc).map(post, [{"model": "super-decider", "state": state((60, 250, 600)[i % 3] + i), "questions": sch} for i in range(conc)]))
post({"model": "super-decider", "state": state(400), "questions": qs(1), "think": 128})
print(f"warmup done in {time.time() - t0:.1f}s", flush=True)

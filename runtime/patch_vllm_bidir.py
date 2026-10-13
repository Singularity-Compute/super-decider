"""Serve a bidirectional questions block in vLLM's DiffusionGemma (encoder prefill).
A request carries vllm_xargs {"djev_bidir": [a, b]} (several ranges flat: [a1, b1, a2, b2, ...]): prompt tokens [a, b) attend to each other both ways during
prefill ((causal AND window) OR same-range, unclamped by the sliding window, as in the HF training mask).
Everything is gated by env DJEV_BIDIR_SERVE=1 in the vLLM process; without it the patched code is a no-op.
  - model state: per-slot range -> mm_req_doc_ranges for prefilling requests (Triton unified attention supports it)
  - scheduler: a prefill chunk never ends inside [a, b) when the block fits the token budget
  - KV cache manager: a prefix-cache hit that ends inside [a, b) is cut back to a (KV there depends on later
    block tokens); a hit covering all of [a, b) is the same block and is kept (think continuations reuse it)
usage: python3 patch_vllm_bidir.py [vllm_dir]"""
import sys, os
V = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages/vllm"
M = "# djev-bidir"

def pristine(path):
    if os.path.exists(path + ".pre_bidir"):
        return open(path + ".pre_bidir").read()
    s = open(path).read()
    open(path + ".pre_bidir", "w").write(s)
    return s

def patch(path, edits):
    s = pristine(path)
    for old, new in edits:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)

# ---- model state
patch(f"{V}/model_executor/models/diffusion_gemma.py", [
    ("        self._inputs_embeds_buf = torch.zeros(\n",
     "        " + M + ": per-slot bidirectional prompt range, inclusive end\n"
     "        import os as _os\n"
     "        self._djev_on = bool(_os.environ.get(\"DJEV_BIDIR_SERVE\"))\n"
     "        self._djev_bidir: dict[int, list[tuple[int, int]]] = {}\n"
     "        if self._djev_on:\n"
     "            for _m in model.modules():\n"
     "                if hasattr(_m, \"mm_prefix_clamp_sliding_window\"):\n"
     "                    _m.mm_prefix_clamp_sliding_window = False\n"
     "        self._inputs_embeds_buf = torch.zeros(\n"),
    ("            self.diffusion_states.prompt_len[req_index].fill_(prompt_len)\n",
     "            self.diffusion_states.prompt_len[req_index].fill_(prompt_len)\n"
     "        self._djev_bidir.pop(req_index, None)  " + M + "\n"
     "        if self._djev_on:\n"
     "            _sp = getattr(new_req_data, \"sampling_params\", None)\n"
     "            _rg = ((getattr(_sp, \"extra_args\", None) or {}) if _sp is not None else {}).get(\"djev_bidir\")\n"
     "            _rs = [_rg[i:i + 2] for i in range(0, len(_rg) - 1, 2)] if (_rg and not isinstance(_rg[0], (list, tuple))) else (_rg or [])\n"
     "            _rs = [(int(r[0]), int(r[1]) - 1) for r in _rs if 0 <= int(r[0]) < int(r[1]) - 1]\n"
     "            if _rs:\n"
     "                self._djev_bidir[req_index] = _rs\n"),
    ("            self.diffusion_states.remove_request(idx)\n",
     "            self.diffusion_states.remove_request(idx)\n"
     "            self._djev_bidir.pop(idx, None)  " + M + "\n"),
    ("        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]\n",
     "        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]\n"
     "        _djev_doc = None  " + M + "\n"
     "        if self._djev_bidir and not for_capture:\n"
     "            _slots = input_batch.idx_mapping_np[:actual_num_reqs].tolist()\n"
     "            _pre = input_batch.is_prefilling_np[:actual_num_reqs]\n"
     "            _djev_doc = {i: list(self._djev_bidir[s]) for i, s in enumerate(_slots)\n"
     "                         if s in self._djev_bidir and bool(_pre[i])} or None\n"),
    ("            causal=causal,\n",
     "            causal=causal,\n"
     "            mm_req_doc_ranges=_djev_doc,  " + M + "\n"),
])

# ---- scheduler: keep the block inside one prefill chunk
p = f"{V}/v1/core/sched/scheduler.py"
s = pristine(p)
if True:
    s += '''

''' + M + ''': a prefill chunk never ends inside the bidirectional range [a, b) of a request when the range fits
# the token budget; it stops at a instead and the block goes in the next chunk.
import os as _djev_os
if _djev_os.environ.get("DJEV_BIDIR_SERVE"):
    _djev_reserve_orig = Scheduler._reserve_prefill_lookahead

    def _djev_reserve(self, request, num_computed_tokens, num_new_tokens):
        n = _djev_reserve_orig(self, request, num_computed_tokens, num_new_tokens)
        sp = request.sampling_params
        rg = (sp.extra_args or {}).get("djev_bidir") if sp is not None else None
        rs = [rg[i:i + 2] for i in range(0, len(rg) - 1, 2)] if (rg and not isinstance(rg[0], (list, tuple))) else (rg or [])
        for r in sorted(rs, key=lambda r: int(r[0])):
            if n <= 0:
                break
            a, b = int(r[0]), int(r[1])
            end = num_computed_tokens + n
            if num_computed_tokens <= a < end < b and (b - a) <= self.max_num_scheduled_tokens:
                # keep the block whole in the next chunk, but never shrink this chunk below 60% of what was planned:
                # a 32K-token prompt with a range every ~1K tokens would otherwise prefill in twice the steps
                # (DI BRIGHT/ToolRet rows: 3.5 -> 7.9 s). A rarely split short range costs one question a partial view.
                if a - num_computed_tokens >= 0.6 * n:
                    n = a - num_computed_tokens
        return n

    Scheduler._reserve_prefill_lookahead = _djev_reserve
'''
    open(p, "w").write(s); print("patched", p)

# ---- KV cache manager: no prefix hit past a
patch(f"{V}/v1/core/kv_cache_manager.py", [
    ("""        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
""", """        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        _sp = request.sampling_params  """ + M + """
        _rg = (_sp.extra_args or {}).get("djev_bidir") if _sp is not None else None
        _rs = [_rg[i:i + 2] for i in range(0, len(_rg) - 1, 2)] if (_rg and not isinstance(_rg[0], (list, tuple))) else (_rg or [])
        _cut = min([int(r[0]) for r in _rs if int(r[0]) < num_new_computed_tokens <= int(r[1])], default=None)
        if _cut is not None:
            computed_blocks, num_new_computed_tokens, num_uncached = (
                self.coordinator.find_longest_cache_hit(
                    request.block_hashes, _cut
                )
            )
"""),
])

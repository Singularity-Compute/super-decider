# Examples

| File | What it shows |
|---|---|
| [`systemone.py`](systemone.py) | Route a support ticket: a choice, a score and a yes/no question in one call, standard library only |
| [`openai_sdk.py`](openai_sdk.py) | The same API through the OpenAI SDK (chat completions) |
| [`escalate.py`](escalate.py) | Act on confident answers and send the rest to a person, with the threshold in your code |
| [`batch.py`](batch.py) | Label a batch of emails in parallel |

All examples call the hosted endpoint by default. Set `API_KEY` to your key, or point `SUPER_DECIDER_URL` at your own
server (for example `http://localhost:8011`) and leave `API_KEY` unset if it runs without auth.

```bash
export API_KEY=...
python examples/systemone.py
```

The request and answer formats, question types, thinking modes and errors are described in [`docs/API.md`](../docs/API.md).

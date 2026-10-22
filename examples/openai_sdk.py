"""Call Super Decider through the OpenAI SDK (chat completions).

The system message holds the question schema as JSON, the user message holds the state as JSON.
The assistant message content is the answers object, also as JSON.

    pip install openai
    export API_KEY=...
    python examples/openai_sdk.py
"""
import json
import os

from openai import OpenAI

# The OpenAI SDK always sends a key; for a local server without auth any non-empty API_KEY works.
client = OpenAI(base_url=os.environ.get("SUPER_DECIDER_URL", "https://llm.a.singularitycompute.com") + "/v1",
                api_key=os.environ["API_KEY"])

schema = {"questions": [
    {"id": "intent", "type": "choice", "instructions": "What does the user want?",
     "options": [{"name": "refund", "description": "Money back for an order"},
                 {"name": "track", "description": "Where an order is"},
                 {"name": "other", "description": "Anything else"}]},
    {"id": "angry", "type": "noul", "instructions": "Is the user angry?"},
]}
state = "Tracking hasn't moved in five days and nobody answers my emails."

r = client.chat.completions.create(model="super-decider", messages=[
    {"role": "system", "content": json.dumps(schema)},
    {"role": "user", "content": json.dumps(state)},
])
answers = json.loads(r.choices[0].message.content)["answers"]
print("intent:", answers["intent"]["label"], round(answers["intent"]["confidence"], 2))
print("angry:", answers["angry"]["label"], round(answers["angry"]["noul"], 2))

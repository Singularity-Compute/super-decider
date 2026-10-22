"""Act on confident decisions, send the rest to a person.

The threshold lives in your code. Raise it and more cases go to review; lower it and more are automated.

    export API_KEY=...
    python examples/escalate.py
"""
from systemone import decide

THRESHOLD = 0.85

refund_requests = [
    "The parcel arrived crushed and the mug inside is broken. Photo attached.",
    "I changed my mind about the colour, can I get my money back? It's been 3 days.",
    "Bought this 14 months ago, it stopped working yesterday. Refund please.",
]

question = {"decision": {
    "type": "choice",
    "instructions": "Apply the policy: full refund for damaged items or returns within 30 days; "
                    "repair or replacement for defects within the 12-month warranty; otherwise decline.",
    "criteria": {"refund": "Full refund", "replace": "Repair or replacement", "decline": "Politely decline"},
}}

for text in refund_requests:
    answer = decide({"request": text}, question)["decision"]
    if answer["confidence"] >= THRESHOLD:
        print(f"auto   {answer['choice']:8s} {answer['confidence']:.2f}  {text}")
    else:
        print(f"review {answer['choice']:8s} {answer['confidence']:.2f}  {text}")

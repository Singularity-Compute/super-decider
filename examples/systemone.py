"""Route a support ticket with the System One API (standard library only).

    export API_KEY=...            # key for the hosted endpoint; unset for a local server without auth
    python examples/systemone.py
"""
import json
import os
import urllib.request

BASE_URL = os.environ.get("SUPER_DECIDER_URL", "https://llm.a.singularitycompute.com")


def decide(state, questions):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
    body = json.dumps({"model": "super-decider", "state": state, "questions": questions}).encode()
    req = urllib.request.Request(BASE_URL + "/v1/systemone", body, headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["answers"]


if __name__ == "__main__":
    answers = decide(
        state={"ticket": "I was charged twice for my March invoice.", "plan": "Pro", "customer_since": "2024-02"},
        questions={
            "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                     "criteria": {"billing": "Payments, invoices, refunds", "technical": "Bugs and outages",
                                  "account": "Login, profile, plan changes", "sales": "New plans and upgrades"}},
            "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                        "criteria": ["low", "normal", "high"]},
            "churn_risk": {"type": "noul", "instructions": "Does the customer sound likely to cancel?"},
        },
    )
    team = answers["team"]
    print(f"team: {team['choice']} ({team['confidence']:.2f})")
    print(f"urgency: {answers['urgency']['score']:.2f} on a 0-2 scale")
    print(f"churn risk: {answers['churn_risk']['noul']:.2f}")

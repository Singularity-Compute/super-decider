"""Label a batch of messages in parallel.

Each request answers all its questions in one pass, so put every question you need about an item into one call
and run several items at once.

    export API_KEY=...
    python examples/batch.py
"""
from concurrent.futures import ThreadPoolExecutor

from systemone import decide

messages = [
    "Your account has been locked. Verify your password here: http://secure-login.example-mail.ru",
    "Hi team, attached are the slides for Thursday's review.",
    "Congratulations! You won a $500 gift card, claim it in the next 24 hours.",
    "Reminder: the office is closed on Monday for the public holiday.",
]

questions = {
    "phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt?"},
    "category": {"type": "choice", "instructions": "What kind of email is this?",
                 "criteria": {"work": "Work communication", "promo": "Marketing or prizes", "security": "Security or account notice"}},
}

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda m: decide({"email": m}, questions), messages))

for message, answers in zip(messages, results):
    print(f"phishing {answers['phishing']['noul']:.2f}  {answers['category']['choice']:8s}  {message[:60]}")

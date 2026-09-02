import json
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

path = sys.argv[1]

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

labels = [
    "START",
    "CORRECT",
    "COMPREHENSION",
    "INCORRECT",
    "QUESTION",
    "CONFUSION",
    "IRRELEVANT",
]

gold = [r["gold_state"] for r in rows]
pred = [r["predicted_state"] for r in rows]

ConfusionMatrixDisplay.from_predictions(
    gold,
    pred,
    labels=labels,
    normalize=None,  # counts
    xticks_rotation=45,
    cmap="Blues",
)

plt.title("Learner State Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Gold")
plt.tight_layout()
plt.show()

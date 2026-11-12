"""CI quality gate: fail the build if the model is not good enough."""
import json
import sys

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90

with open("metrics/scores.json") as f:
    scores = json.load(f)

acc = scores["accuracy"]
print(f"accuracy={acc:.4f} (required >= {THRESHOLD})")
if acc < THRESHOLD:
    sys.exit("Quality gate failed: accuracy below threshold")

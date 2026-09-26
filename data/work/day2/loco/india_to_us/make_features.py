import csv
from pathlib import Path

root = Path("data/work/day2/loco/india_to_us")
root.mkdir(parents=True, exist_ok=True)

source1 = {}
with open("student_resource/dataset/train/train_source1.tsv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        source1[r["entity_id"]] = r["country"]

with open("data/work/day1/features.tsv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

for r in rows:
    country = source1[r["source1_entity_id"]]
    r["fold"] = "0" if country == "India" else "1"

out = root / "features.tsv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
    w.writeheader()
    w.writerows(rows)

print("CREATED:", out)
print("India rows:", sum(source1[r["source1_entity_id"]] == "India" for r in rows))
print("US rows:", sum(source1[r["source1_entity_id"]] == "US" for r in rows))

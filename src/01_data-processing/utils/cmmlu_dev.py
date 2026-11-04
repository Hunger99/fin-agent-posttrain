"""Write CMMLU's dev split, the source of CMMLU's 5 examples per subject (docs/history/04_evaluation.md section 2).

convert.py wrote only the test split; this reads the same pinned zip (haonan-li/cmmlu at commit efcc940,
SHA256 in MANIFEST.md) and writes the dev rows in the test rows' format, so mc_eval.py matches them by
"exam". The zip is downloaded into raw/cmmlu when it is missing and checked against the recorded hash.

Usage: python cmmlu_dev.py <data root>
"""
import csv
import hashlib
import io
import json
import os
import sys
import zipfile

from huggingface_hub import hf_hub_download

ROOT = sys.argv[1]
REVISION = "efcc940752ea4a1ea94d2727f11f83858d64fc8e"
SHA256 = "22ecf70b28bef447ee7d8aa5fe144f56996762f901a8537b03b7693773c672a6"
raw = os.path.join(ROOT, "raw", "cmmlu")
path = os.path.join(raw, "cmmlu_v1_0_1.zip")
if not os.path.exists(path):
    hf_hub_download("haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset", revision=REVISION, local_dir=raw)
with open(path, "rb") as f:
    if hashlib.sha256(f.read()).hexdigest() != SHA256:
        raise SystemExit(f"{path} does not match the SHA256 in MANIFEST.md")

rows = []
with zipfile.ZipFile(path) as z:
    for name in sorted(n for n in z.namelist() if "/dev/" in f"/{n}" and n.endswith(".csv")):
        subject = os.path.basename(name)[:-4]
        for r in csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8")):
            rows.append({"id": f"cmmlu-dev-{subject}-{r['']}", "source": "cmmlu", "split": "dev", "exam": subject,
                         "task": "单项选择题", "question": r["Question"], "choices": {k: r[k] for k in "ABCD"},
                         "answer": r["Answer"]})
out = os.path.join(ROOT, "processed", "eval", "cmmlu_dev.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"rows": len(rows), "subjects": len({r["exam"] for r in rows})}))

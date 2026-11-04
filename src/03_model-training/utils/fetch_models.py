"""Download the starting-model candidates and their reference models at pinned commits.

docs/history/02_model-training.md section 1: the starting model was first chosen from Qwen2.5-1.5B and
Qwen3-1.7B-Base, with Qwen2.5-7B-Instruct and Qwen3-8B as the E1 references; the author then chose
Qwen3-4B-Base, with Qwen3-8B as its reference. Files already present are skipped. The commits are the latest ones the Hub reported
on 2026-10-25 (research/log/2026-10-25-2030.md); the license tag the Hub reports is recorded with each.
Weights go to host 1's disk over NFS, since host 2's system disk is nearly full; huggingface_hub
resumes partial files and checks each LFS file against its published hash.

Usage: python fetch_models.py <models dir>
"""
import json
import os
import sys
import urllib.request

from huggingface_hub import snapshot_download

OUT = sys.argv[1]
MODELS = [("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
          ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"),
          ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
          ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
          # The author switched the starting model to Qwen3-4B-Base on 2026-10-26 (research/log/2026-10-26-0057.md)
          ("Qwen/Qwen3-4B-Base", "906bfd4b4dc7f14ee4320094d8b41684abff8539"),
          # Same-generation instruct model, for research/log/2026-10-26-0409.md's comparison
          ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
          # The author's larger E1 reference for Qwen3-4B-Base (research/log/2026-10-26-0453.md)
          ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18")]
rows = []
for repo, commit in MODELS:
    meta = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{commit}", timeout=60))
    path = snapshot_download(repo, revision=commit, local_dir=os.path.join(OUT, repo.split("/")[1]))
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs if ".cache" not in d)
    rows.append({"repo": repo, "commit": commit, "license": (meta.get("cardData") or {}).get("license"),
                 "path": path, "bytes": size})
    print(json.dumps(rows[-1]), flush=True)
with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=1)

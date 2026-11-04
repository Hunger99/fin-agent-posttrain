"""Download pinned Hugging Face files into one directory per repo and verify SHA-256.

Adapted from rsi-data/scripts/data_check/fetch_hf.py, which keeps the pinned-commit download and
the check against the publisher's LFS hashes. Changed here: this project's repos, an include list
per repo instead of a skip list (CFLUE's application.json and DianJin-R1-Data's English fin_qa.json
are not used by docs/history/01_data-processing.md), model repos for the tokenizer, the license tag the Hub
reports, and MANIFEST.md written at the data root rather than inside raw/.

Usage: python fetch_hf.py $FP_DATA_ZH [local dir name]...

With local dir names, only those repos are fetched and checked, and MANIFEST.md keeps its rows for the
other repos: a host that holds only part of raw/, such as host 2 (research/log/2026-10-25-0610.md), would
otherwise download everything again.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import urllib.request

ROOT = sys.argv[1]
# (local dir, repo type, repo, commit, files to fetch). The last field is None for every file but
# .gitattributes, a list of paths, or a predicate on the path for repos with hundreds of files.
JOBS = [
    ("financeiq", "datasets", "Duxiaoman-DI/FinanceIQ", "fca1d9ceb6925041264d211b9ddf7264e8145248", None),
    ("cflue", "datasets", "DianJin/CFLUE", "adcf2643b83dfaed79d43ed916716ac57e88544a",
     ["train.json", "val.json", "test.json"]),
    ("dianjin-r1-data", "datasets", "DianJin/DianJin-R1-Data", "da348edd64c5ad2705962ab7b2357ee850f6488a",
     ["README.md", "train/cflue_mcq.json", "train/cflue_oe.json"]),
    # Decontamination tokenizes with the starting model's tokenizer; the starting model is not chosen
    # yet, so this uses the first candidate and the check is rerun if the choice changes the tokenizer.
    ("qwen2.5-1.5b-tokenizer", "models", "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323",
     ["LICENSE", "tokenizer.json", "tokenizer_config.json"]),
    # Second round, approved in research/log/2026-10-25-0039.md.
    ("fincorpus", "datasets", "Duxiaoman-DI/FinCorpus", "dae52562603787e65ee00cc95561bab4fa818d9b", None),
    ("industrycorpus2-finance", "datasets", "BAAI/IndustryCorpus2_finance_economics",
     "0b8a03a36c7db0264f66cecc32f1e3a3caf6aada", lambda f: f == "README.md" or f.startswith("chinese/high/")),
    # The first 120 of the 1,000 files in the 4-5 tier (9.12 GB in all). 100 were taken first for the
    # roughly 0.4B general tokens continued pretraining needs; they measured 0.33B tokens
    # (research/log/2026-10-25-0800.md), so the author added 20 on 2026-10-25. After dedup and
    # decontamination the 120 gave 0.360B, 15M short of 0.375B, so 10 more (research/log/2026-10-25-1536.md).
    ("fineweb-edu-chinese-v2.1", "datasets", "opencsg/Fineweb-Edu-Chinese-V2.1",
     "a5b574efa48beb3a8f6887ef0b093becf004328b",
     ["README.md", "opencsg模型社区许可协议.pdf"] + [f"4_5/{i:06d}.parquet" for i in range(130)]),
    ("agentar-deepfinance-100k", "datasets", "antgroup/Agentar-DeepFinance-100K",
     "7f4cef5b057436c526ea6bd3da9a2320ee010fd2", None),
    ("industryinstruction-finance", "datasets", "BAAI/IndustryInstruction_Finance-Economics",
     "209c619155b65556be03e22a26e849cdf936df69", lambda f: f.endswith((".jsonl", ".md"))),
    ("alpaca-zh", "datasets", "shibing624/alpaca-zh", "f39db019a94f8dbea48ab30d2bdc090703284559", None),
    # old/ holds a superseded copy of the same data.
    ("smoltalk-chinese", "datasets", "opencsg/smoltalk-chinese", "5edf5d0d7b06794755cf78b274a150a4efe1c9d4",
     lambda f: "/" not in f and f.endswith((".parquet", ".md", ".pdf"))),
    ("tulu-3-sft-mixture", "datasets", "allenai/tulu-3-sft-mixture", "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be", None),
    ("dianjin-fin-prm-data", "datasets", "DianJin/DianJin-Fin-PRM-Data", "3aab11686fa74a66f1987b14aa58b78be9e94c31", None),
    ("gsm8k", "datasets", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
     ["README.md", "main/test-00000-of-00001.parquet"]),
    ("ifeval", "datasets", "google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", None),
    ("cmmlu", "datasets", "haonan-li/cmmlu", "efcc940752ea4a1ea94d2727f11f83858d64fc8e", ["README.md", "cmmlu_v1_0_1.zip"]),
    # Only what dense CLS embeddings need; the ONNX export and the sparse and ColBERT heads are not used.
    ("bge-m3", "models", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181",
     ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
      "sentencepiece.bpe.model"]),
    # English round (research/log/2026-10-26-2006.md), fetched into the English data root.
    ("tat-qa", "datasets", "next-tat/TAT-QA", "c96247f5077eac447f63527fd3dcfdc58bb56d6a", None),
    ("xfinbench", "datasets", "Zhihan/XFinBench", "9eb2a9d985e7a30de55b21de84eba176961453e0",
     ["README.md", "knowledge_bank.csv", "test_set.csv", "validation_set.csv"]),
    ("mmlu", "datasets", "cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe",
     ["README.md", "all/dev-00000-of-00001.parquet", "all/validation-00000-of-00001.parquet",
      "all/test-00000-of-00001.parquet"]),
    ("bizbench", "datasets", "kensho/bizbench", "0a793f2f886156902c72b4a22cab82bb9dceaecf",
     lambda f: f.endswith((".parquet", ".md"))),
    ("sujet-finance-instruct-177k", "datasets", "sujet-ai/Sujet-Finance-Instruct-177k",
     "5e72431b6db6c339613ae809f4ae3c274ea1d0c7", None),
    ("finance-instruct-500k", "datasets", "Josephgflowers/Finance-Instruct-500k",
     "583a98fb0ec14d904e9423b671d9d0fea88891b6", None),
    ("finance-alpaca", "datasets", "gbharti/finance-alpaca", "c88d3d5e7e2c7cab9f11a56f27bd5ba3ed68f075", None),
    ("dianjin-r1-finqa", "datasets", "DianJin/DianJin-R1-Data", "da348edd64c5ad2705962ab7b2357ee850f6488a",
     ["README.md", "train/fin_qa.json"]),
    ("nemotron-sft-agentic-finance", "datasets", "nvidia/Nemotron-SFT-Agentic-Finance-v1",
     "13f0aa9f04af54af991b6c249585dc59c5047938", None),
    # English document QA source and DocFinQA's evaluation splits (research/log/2026-10-26-2222.md).
    ("industrycorpus2-finance-en", "datasets", "BAAI/IndustryCorpus2_finance_economics",
     "0b8a03a36c7db0264f66cecc32f1e3a3caf6aada",
     ["README.md", "english/high/rank_00978.parquet", "english/high/rank_00979.parquet"]),
    ("docfinqa", "datasets", "kensho/DocFinQA", "64ebaff62f692495bcc182f45cf9a9606251b19b", ["dev.json", "test.json"]),
]
# Repos whose data is only on GitHub: (local dir, repo, commit, files, license from the GitHub API).
GH_JOBS = [
    ("finqa", "czyssrs/FinQA", "0f16e2867befa6840783e58be38c9efb9229d742",
     ["LICENSE", "README.md", "dataset/train.json", "dataset/dev.json", "dataset/test.json"], "MIT (GitHub)"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


ONLY = set(sys.argv[2:])
rows = []
for name, kind, repo, commit, include in JOBS:
    if ONLY and name not in ONLY:
        continue
    meta = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/{kind}/{repo}/revision/{commit}?blobs=true", timeout=60))
    # CFLUE has no dataset card, so the Hub reports no license; the record says so instead of guessing.
    license_tag = (meta.get("cardData") or {}).get("license") or "no tag on the Hub"
    siblings = {s["rfilename"]: s for s in meta["siblings"]}
    if include is None:
        wanted = [f for f in siblings if f != ".gitattributes"]
    elif callable(include):
        wanted = [f for f in siblings if include(f)]
    else:
        wanted = include
    out_dir = os.path.join(ROOT, "raw", name)
    for f in wanted:
        s = siblings[f]  # KeyError here means the include list names a file absent at this commit
        out = os.path.join(out_dir, f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        prefix = "datasets/" if kind == "datasets" else ""
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{commit}/{f}"
        want = (s.get("lfs") or {}).get("sha256")
        if not (os.path.exists(out) and want and sha256(out) == want):
            # Over HTTP/2, curl exited 92 ("HTTP/2 stream was not closed cleanly") 3.1 GB into FinCorpus's
            # 5.5 GB announcement file; over HTTP/1.1 it then exited 56 (connection reset) partway through
            # the 4.3 GB articles file, and curl's own --retry restarts from zero. So LFS files resume
            # with -C - and are retried here, each attempt continuing where the last one stopped. A small
            # file without a published hash is fetched whole, because resuming a complete file gets HTTP 416.
            if not want and os.path.exists(out):
                os.remove(out)
            for attempt in range(20):
                done = subprocess.run(["curl", "-sSL", "--fail", "--http1.1", *(["-C", "-"] if want else []),
                                       "-o", out, url]).returncode == 0
                if done:
                    break
                print(f"retry {attempt + 1} {repo} {f}", file=sys.stderr, flush=True)
            else:
                raise RuntimeError(f"download failed 20 times: {url}")
        got = sha256(out)
        # Small files are stored without LFS, so the Hub publishes no hash to compare against.
        check = "no published hash" if want is None else ("ok" if got == want else f"MISMATCH want {want}")
        rows.append((repo, commit[:7], license_tag, f, s.get("size"), got, check))
        if check.startswith("MISMATCH"):
            print("MISMATCH", repo, f, file=sys.stderr)

for name, repo, commit, files, license_tag in GH_JOBS:
    if ONLY and name not in ONLY:
        continue
    for f in files:
        out = os.path.join(ROOT, "raw", name, f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{f}"
        if subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-o", out, url]).returncode != 0:
            raise RuntimeError(f"download failed: {url}")
        # GitHub publishes no file hash, so the record keeps the hash of what was fetched at the commit.
        rows.append((repo, commit[:7], license_tag, f, os.path.getsize(out), sha256(out), "no published hash"))

kept = []
if ONLY and os.path.exists(os.path.join(ROOT, "MANIFEST.md")):
    fetched = {repo for name, _, repo, _, _ in JOBS if name in ONLY} | {g[1] for g in GH_JOBS if g[0] in ONLY}
    with open(os.path.join(ROOT, "MANIFEST.md")) as m:
        for line in m:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| ") and len(cells) == 7 and cells[0] not in fetched and cells[0] != "repo" \
                    and not cells[0].startswith("---"):
                kept.append(tuple(cells))
rows = kept + rows
with open(os.path.join(ROOT, "MANIFEST.md"), "w") as m:
    m.write("# fin-posttrain data\n\n"
            f"Downloaded {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC from Hugging Face at pinned commits by "
            "src/01_data-processing/utils/fetch_hf.py; SHA-256 checked against the publisher's LFS records. "
            "License is the tag the Hub reports at that commit.\n\n"
            "| repo | commit | license tag | file | bytes | sha256 | check |\n| --- | --- | --- | --- | --- | --- | --- |\n")
    for r in rows:
        m.write("| " + " | ".join(str(x) for x in r) + " |\n")
print("files", len(rows), "mismatches", sum(1 for r in rows if r[6].startswith("MISMATCH")))

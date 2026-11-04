"""Apply section 2's normalization and rule filters to every corpus document, and write the documents kept.

docs/history/01_data-processing.md section 2. The rules and their definitions are corpus_pilot.py's, whose keep
rates research/log/2026-10-25-0800.md measured on 10,000-document samples; here every document is
filtered. What is written follows the author's 2026-10-25 choices (section 9 items 8 and 9):

- Stored text gets OpenCC traditional to simplified and loses HTML tags and control characters. NFKC is
  applied only to the copy the rules measure, because it would turn Chinese full-width punctuation into ASCII.
- FinCorpus: a document that passes the rules enters the scoring pool when a hash of its id falls under
  FIN_RATE, which puts about 3B of the 0800 estimate of 7.62B post-rule tokens into the pool. The scoring
  sample, the classifier and the >= 3 cut then work on this pool only.
- IndustryCorpus2, which section 2 does not score, and Fineweb-Edu-Chinese: every document that passes is
  written.

FinCorpus's exam file is excluded whole (section 2) and is not read. Output: interim/corpus/<stem>.jsonl.gz,
one {"id", "source", "text", "n_tokens"} per line, and reports/corpus_filter_<stems>.md with per-step counts.

Usage: python corpus_filter.py $FP_DATA_ZH <source stem>...
"""
import glob
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import string
import subprocess
import sys
import unicodedata

import emoji
import opencc
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT, STEMS = sys.argv[1], sys.argv[2:]
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "interim", "corpus")
MIN_HAN, MIN_LEN, MAX_LEN, MAX_REP_LINES, MAX_SPECIAL = 0.30, 200, 100_000, 0.30, 0.25  # section 2
FIN_RATE = 3.0 / 7.62  # about 3B of FinCorpus's estimated 7.62B post-rule tokens (2026-10-25-0800)
MIN_FREE = 40 * 2**30  # the system disk is shared; stop before it falls below this
# Copied from corpus_pilot.py, which copied Data-Juicer 1.6.1's data_juicer/ops/common/special_characters.py
OTHER_SPECIAL = ('\x92 \x93\x97 \x9d\x99 \x96 \x95\x91\x9c　   \x98\x81 \x82￼'
                 '\x83\x84’“”–ー一▬…✦�\xad£​•€«»°·═×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰\x85'
                 '\x85‑≤≥‖◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑'
                 '٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖」﴾》')
SPECIAL = set(string.punctuation + string.digits + string.whitespace + OTHER_SPECIAL) | set(emoji.EMOJI_DATA)
HAN = re.compile(r"[一-鿿]")
TAG = re.compile(r"<[^<>\n]{1,200}>")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏﻿]")
# stem -> (source name, format, files under raw/, scored pool or not)
SOURCES = {"fincorpus_announcement": ("FinCorpus 公告", "gz", "fincorpus/data/announcement_data.jsonl.gz", True),
           "fincorpus_articles": ("FinCorpus 文章", "gz", "fincorpus/data/fin_articles_data.jsonl.gz", True),
           "fincorpus_news": ("FinCorpus 新闻", "gz", "fincorpus/data/fin_news_data_final.jsonl.gz", True),
           "industrycorpus2": ("IndustryCorpus2 金融中文高质量档", "parquet",
                               "industrycorpus2-finance/chinese/high/*.parquet", False),
           # The first 120 files; files 121-130 were added later (research/log/2026-10-25-1536.md) under their own
           # stem, so the documents already filtered, signed and decontaminated keep their ids and results
           "fineweb_edu_zh": ("Fineweb-Edu-Chinese-V2.1 4-5 档", "parquet",
                              ["fineweb-edu-chinese-v2.1/4_5/0000[0-9][0-9].parquet", "fineweb-edu-chinese-v2.1/4_5/0001[01][0-9].parquet"], False),
           "fineweb_edu_zh_121_130": ("Fineweb-Edu-Chinese-V2.1 4-5 档", "parquet", "fineweb-edu-chinese-v2.1/4_5/00012[0-9].parquet", False)}
STEPS = ["读入", "汉字 ≥30%", "长度 200–100,000", "重复行 ≤30%", "特殊字符 ≤25%"]
_cc = _tok = None


def init():
    global _cc, _tok
    _cc = opencc.OpenCC("t2s")
    _tok = Tokenizer.from_file(os.path.join(RAW, "qwen2.5-1.5b-tokenizer", "tokenizer.json"))


def chunks(stem):
    """Lists of (id, text) in file order; FinCorpus lines are decoded by pigz, which is faster than gzip."""
    _, kind, pattern, _ = SOURCES[stem]
    batch = []
    for path in sorted(p for pat in ([pattern] if isinstance(pattern, str) else pattern) for p in glob.glob(os.path.join(RAW, pat))):
        if kind == "gz":
            with subprocess.Popen(["pigz", "-dc", path], stdout=subprocess.PIPE) as p:
                for i, line in enumerate(p.stdout):
                    batch.append((f"{stem}-{i}", line))
                    if len(batch) == 2000:
                        yield batch
                        batch = []
        else:
            pf = pq.ParquetFile(path)
            base = os.path.basename(path)[:-8]
            for g in range(pf.metadata.num_row_groups):
                for i, t in enumerate(pf.read_row_group(g, columns=["text"]).column(0).to_pylist()):
                    batch.append((f"{stem}-{base}-{g}-{i}", t))
                    if len(batch) == 2000:
                        yield batch
                        batch = []
    if batch:
        yield batch


def keep_in_pool(doc_id):
    return int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16) / 2**32 < FIN_RATE


def work(args):
    stem, batch = args
    name, kind, _, scored = SOURCES[stem]
    counts = [[0, 0] for _ in range(len(STEPS) + 1)]  # [docs, chars] after each step, then the pool
    lines = []
    for doc_id, raw in batch:
        text = json.loads(raw)["text"] if kind == "gz" else raw
        text = text or ""
        stored = CONTROL.sub("", TAG.sub(" ", _cc.convert(text)))
        measured = unicodedata.normalize("NFKC", stored)
        n = len(measured)
        chars = [c for c in measured if not c.isspace()]
        lns = [x.strip() for x in measured.splitlines() if x.strip()]
        checks = [bool(chars) and len(HAN.findall(measured)) / len(chars) >= MIN_HAN,
                  MIN_LEN <= n <= MAX_LEN,
                  (1 - len(set(lns)) / len(lns) if lns else 0.0) <= MAX_REP_LINES,
                  (sum(c in SPECIAL for c in measured) / n if n else 0.0) <= MAX_SPECIAL]
        counts[0][0] += 1
        counts[0][1] += len(text)
        ok = True
        for k, c in enumerate(checks, 1):
            ok = ok and c
            if ok:
                counts[k][0] += 1
                counts[k][1] += len(stored)
        if not ok or (scored and not keep_in_pool(doc_id)):
            continue
        ntok = len(_tok.encode(stored, add_special_tokens=False).ids)
        counts[-1][0] += 1
        counts[-1][1] += ntok
        lines.append(json.dumps({"id": doc_id, "source": name, "text": stored, "n_tokens": ntok}, ensure_ascii=False))
    payload = gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), 6) if lines else b""
    return counts, payload


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    report = ["# 语料全量规则过滤", "",
              "由 src/01_data-processing/utils/corpus_filter.py 生成。每格为到这一步为止还留下的篇数和字数（读入一列为原文字数，其余为写出文字的字数）。"
              "规则同 corpus_pilot.py；最后一列是写出的文档：FinCorpus 只写进入打分池的部分，其余来源写出全部通过规则的文档；token 用 Qwen2.5-1.5B 的分词器计。", "",
              "| 来源 | " + " | ".join(STEPS) + " | 写出（篇 / token） |", "| --- | " + " | ".join("---" for _ in STEPS) + " | --- |"]
    with multiprocessing.Pool(24, initializer=init) as pool:  # 24 of 96 cores; others use the host too
        for stem in STEMS:
            total = [[0, 0] for _ in range(len(STEPS) + 1)]
            with open(os.path.join(OUT, stem + ".jsonl.gz"), "wb") as f:
                for counts, payload in pool.imap(work, ((stem, b) for b in chunks(stem)), chunksize=1):
                    for t, c in zip(total, counts):
                        t[0] += c[0]
                        t[1] += c[1]
                    f.write(payload)  # concatenated gzip members are one valid gzip stream
                    if shutil.disk_usage(OUT).free < MIN_FREE:
                        sys.exit(f"stopped: under {MIN_FREE >> 30} GiB free on {OUT}")
            report.append(f"| {SOURCES[stem][0]} | " + " | ".join(f"{d:,} / {c / 1e8:.1f} 亿字" for d, c in total[:-1])
                          + f" | {total[-1][0]:,} / {total[-1][1] / 1e9:.2f}B |")
            print(report[-1], flush=True)
    with open(os.path.join(ROOT, "reports", f"corpus_filter_{'_'.join(STEMS)}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

"""Measure what the corpus filters of docs/history/01_data-processing.md section 2 keep, on 10,000 documents per source.

Section 2 asks for each filter's keep rate to be measured on a 10,000-document sample per source before
the full run. Every document of each source is read once: a reservoir keeps a uniform sample, and the
pass counts the source's documents and characters, so the sample's rates scale to the whole source.

The filters run in section 2's order, each on what the previous ones keep, after normalization
(NFKC, OpenCC traditional to simplified, HTML tags and control characters removed):

1. Han share below 30% of the non-space characters: dropped (section 2's rule for Chinese corpora).
2. Length outside 200-100,000 characters: dropped. Data-Juicer's ``text_length_filter`` counts characters.
3. Repeated lines above 30%: dropped. Share of the non-empty lines that repeat an earlier line; Data-Juicer
   has no filter by that name, so this is the plain reading of section 2.
4. Special characters above 25%: dropped. Data-Juicer's ``special_characters_filter``, whose character set
   is copied below from Data-Juicer 1.6.1 (``data_juicer/ops/common/special_characters.py``, Apache-2.0).
   That set holds every digit and whitespace character, Chinese punctuation, and the characters 一 and 上.

Not measured: template-line removal, which section 2 names without a rule; FinCorpus's quality scoring,
which needs the teacher model; and dedup, since a 10,000-document sample of millions shows almost no
duplicates. FinCorpus's exam file is excluded whole by section 2 and is not read.

Token counts use the Qwen2.5-1.5B tokenizer. Fineweb-Edu-Chinese's 100 files are small, so all their
documents are tokenized, which gives their token count directly.

Usage: python corpus_pilot.py $FP_DATA_ZH
"""
import glob
import gzip
import json
import multiprocessing
import os
import random
import re
import string
import sys
import unicodedata

import emoji
import numpy as np
import opencc
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "raw")
K = int(os.environ.get("PILOT_K", 10_000))  # section 2; PILOT_K only for smoke tests
SEED = 20261025
MIN_HAN, MIN_LEN, MAX_LEN, MAX_REP_LINES, MAX_SPECIAL = 0.30, 200, 100_000, 0.30, 0.25  # section 2
OTHER_SPECIAL = ('\x92 \x93\x97 \x9d\x99 \x96 \x95\x91\x9c　   \x98\x81 \x82￼'
                 '\x83\x84’“”–ー一▬…✦�\xad£​•€«»°·═×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰\x85'
                 '\x85‑≤≥‖◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑'
                 '٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖」﴾》')
SPECIAL = set(string.punctuation + string.digits + string.whitespace + OTHER_SPECIAL) | set(emoji.EMOJI_DATA)
HAN = re.compile(r"[一-鿿]")
TAG = re.compile(r"<[^<>\n]{1,200}>")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏﻿]")
# (name, sample file stem, format, files under raw/)
SOURCES = [("FinCorpus 公告", "fincorpus_announcement", "gz", "fincorpus/data/announcement_data.jsonl.gz"),
           ("FinCorpus 文章", "fincorpus_articles", "gz", "fincorpus/data/fin_articles_data.jsonl.gz"),
           ("FinCorpus 新闻", "fincorpus_news", "gz", "fincorpus/data/fin_news_data_final.jsonl.gz"),
           ("IndustryCorpus2 金融中文高质量档", "industrycorpus2", "parquet", "industrycorpus2-finance/chinese/high/*.parquet"),
           ("Fineweb-Edu-Chinese-V2.1 4-5 档前 100 个文件", "fineweb_edu_zh", "parquet", "fineweb-edu-chinese-v2.1/4_5/*.parquet")]


def texts_of(kind, pattern):
    """Every document's text, in file order."""
    for path in sorted(glob.glob(os.path.join(RAW, pattern))):
        if kind == "gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)["text"]
        else:
            pf = pq.ParquetFile(path)
            for g in range(pf.metadata.num_row_groups):
                yield from pf.read_row_group(g, columns=["text"]).column(0).to_pylist()


def scan(args):
    """Reservoir sample of K documents, with the source's document and character counts."""
    name, _, kind, pattern = args
    rng = random.Random(SEED)
    sample, n, chars = [], 0, 0
    for t in texts_of(kind, pattern):
        t = t or ""
        chars += len(t)
        if n < K:
            sample.append(t)
        else:
            j = rng.randrange(n + 1)
            if j < K:
                sample[j] = t
        n += 1
    return name, n, chars, sample


def normalize(text, cc):
    text = cc.convert(unicodedata.normalize("NFKC", text))
    return CONTROL.sub("", TAG.sub(" ", text))


def han_share(text):
    chars = [c for c in text if not c.isspace()]
    return len(HAN.findall(text)) / len(chars) if chars else 0.0


def repeated_lines(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return 1 - len(set(lines)) / len(lines) if lines else 0.0


def special_share(text):
    return sum(c in SPECIAL for c in text) / len(text) if text else 0.0


if __name__ == "__main__":
    # One process per source: the reads are sequential and mostly bound by JSON and gzip decoding
    with multiprocessing.Pool(len(SOURCES)) as pool:
        scanned = pool.map(scan, SOURCES)
    tok = Tokenizer.from_file(os.path.join(RAW, "qwen2.5-1.5b-tokenizer", "tokenizer.json"))
    cc = opencc.OpenCC("t2s")
    os.makedirs(os.path.join(ROOT, "interim", "pilot"), exist_ok=True)
    rows, dist = [], []
    for (name, stem, _, _), (_, n, chars, sample) in zip(SOURCES, scanned):
        norm = [normalize(t, cc) for t in sample]
        han = np.array([han_share(t) for t in norm])
        length = np.array([len(t) for t in norm])
        rep = np.array([repeated_lines(t) for t in norm])
        spec = np.array([special_share(t) for t in norm])
        steps = [han >= MIN_HAN, (length >= MIN_LEN) & (length <= MAX_LEN), rep <= MAX_REP_LINES, spec <= MAX_SPECIAL]
        keep, after = np.ones(len(norm), dtype=bool), []
        for s in steps:
            keep &= s
            after.append(int(keep.sum()))
        ntok = np.array([len(e.ids) for e in tok.encode_batch(norm, add_special_tokens=False)])
        chars_per_tok = length.sum() / max(ntok.sum(), 1)
        kept_char_share = length[keep].sum() / max(length.sum(), 1)
        est_tokens = chars / chars_per_tok
        rows.append((name, n, chars, len(sample), after, chars_per_tok, est_tokens, est_tokens * kept_char_share))
        dist.append((name, np.percentile(spec, [25, 50, 75, 90]), float((spec > MAX_SPECIAL).mean()),
                     float(((length < MIN_LEN) | (length > MAX_LEN)).mean())))
        # The sample is kept for the scoring pilot, which also needs 10,000-document samples (section 2)
        with open(os.path.join(ROOT, "interim", "pilot", f"{stem}.sample.jsonl"), "w", encoding="utf-8") as f:
            for t, k in zip(sample, keep.tolist()):
                f.write(json.dumps({"source": name, "text": t, "kept": k}, ensure_ascii=False) + "\n")
        print(name, n, chars, after, flush=True)
    _, _, fw_kind, fw_pattern = SOURCES[-1]
    fineweb_tokens, batch = 0, []
    for t in texts_of(fw_kind, fw_pattern):
        batch.append(t or "")
        if len(batch) == 5000:
            fineweb_tokens += sum(len(e.ids) for e in tok.encode_batch(batch, add_special_tokens=False))
            batch = []
    fineweb_tokens += sum(len(e.ids) for e in tok.encode_batch(batch, add_special_tokens=False))

    lines = ["# 语料抽样试跑：规范化和过滤规则的保留率", "",
             f"由 src/01_data-processing/utils/corpus_pilot.py 生成。每个来源读一遍全部文档，用水库抽样均匀抽 {K:,} 篇（种子 {SEED}），"
             "按数据处理设计第 2 节的顺序依次过滤；每列是到这一步为止还留下的篇数。特殊字符按 Data-Juicer 1.6.1 的字符集计，"
             "其中包括全部数字、空白、中文标点和\"一\"\"上\"两个汉字。删模板行、FinCorpus 打分和去重没有试跑。"
             "token 用 Qwen2.5-1.5B 的分词器计。", "",
             "| 来源 | 全部篇数 | 全部字数 | 抽样 | 汉字 ≥30% | 长度 200–100,000 | 重复行 ≤30% | 特殊字符 ≤25% | 每 token 字数 | 全部 token（推算） | 过滤后 token（推算） |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, n, chars, k, after, cpt, est, kept in rows:
        lines.append(f"| {name} | {n:,} | {chars:,} | {k:,} | " + " | ".join(f"{a:,}（{a / k:.0%}）" for a in after)
                     + f" | {cpt:.2f} | {est / 1e9:.2f}B | {kept / 1e9:.2f}B |")
    lines += ["", "\"过滤后 token\"按抽样中留下的文档所占的字数比例推算，不含打分和去重。", "",
              "## 单看两条规则", "", "| 来源 | 特殊字符比例的 25/50/75/90 分位 | 只按特殊字符 >25% 会删掉 | 只按长度会删掉 |",
              "| --- | --- | --- | --- |"]
    for name, q, sp, ln in dist:
        lines.append(f"| {name} | " + " / ".join(f"{x:.2f}" for x in q) + f" | {sp:.0%} | {ln:.0%} |")
    lines += ["", f"Fineweb-Edu-Chinese-V2.1 4-5 档前 100 个文件全部分词，共 {fineweb_tokens:,} 个 token。"]
    with open(os.path.join(ROOT, "reports", "corpus_pilot.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

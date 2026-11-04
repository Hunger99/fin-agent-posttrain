"""Flag Chinese training samples that translate a GSM8K test item, which the 0.90 embedding threshold misses.

research/log/2026-10-25-0730.md read the smoltalk-chinese prompts nearest to GSM8K test items: every
read pair at cosine 0.80-0.90 was a Chinese translation of the item, 6 of 10 at 0.75-0.80, 1 of 10 at
0.70-0.75 and none of 10 at 0.65-0.70, while embed_decontam.py's 0.90 caught 87 of an estimated
640-780. A pair across languages scores lower than one within a language, so a Chinese sample is
flagged against a GSM8K test item at cosine >= 0.75, or at >= 0.70 when every number of the item (at
least two) occurs in the prompt; among the read pairs, every one passing that number check was a
translation.

An English sample is flagged against a Chinese evaluation item at cosine >= 0.80. research/log/2026-10-25-0742.md
read 27 Tülu 3 pairs against CMMLU mathematics: the same problem in 1 of 3 pairs at >= 0.85, 3 of 8 at
0.80-0.85 and 1 of 8 at 0.75-0.80; the author adopted 0.80 on 2026-10-25 (docs/history/decisions.md). The number
check does not carry over there, since common algebra numbers occur by chance. "English" means ASCII letters
make up at least half of the non-space characters once markup and URLs are dropped: the Han-share test alone
also called number-heavy Chinese exam items non-Chinese, and those are same-language pairs that
embed_decontam.py's 0.90 covers. The Chinese side is the evaluation set, not each item's own script, because
CMMLU mathematics items written mostly in LaTeX are still Chinese items.

A Chinese sample against IFEval has no read evidence of translated prompts; it is only counted at >= 0.75,
with examples, for the report. Vectors and texts are those of embed_decontam.py (saved vectors, first
non-empty user turn).

Usage: python xling_decontam.py $FP_DATA_ZH
"""
import collections
import html
import json
import os
import re
import sys
import unicodedata

import numpy as np

ROOT = sys.argv[1]
HIGH, LOW = 0.75, 0.70  # research/log/2026-10-25-0730.md, see the module docstring
EN_ZH = 0.80  # English sample against a Chinese evaluation item, research/log/2026-10-25-0742.md
ASCII_LETTER = re.compile(r"[A-Za-z]")
TAG_OR_URL = re.compile(r"<[^<>]*>|https?://\S+")
ZH_SETS = {"FinanceIQ test", "FinanceIQ dev", "CFLUE val", "CFLUE test", "CMMLU test"}
HAN = re.compile(r"[一-鿿]")
NUM = re.compile(r"\d+(?:\.\d+)?")
WRAPPER = re.compile(r"^(假设你是一位金融行业专家，请回答下列问题。|注意：.*|请一步步思考。)$")  # as embed_decontam.py
# (name, vector stem, file, how the text is built), the training sources of embed_decontam.py
SOURCES = [("DianJin-R1-Data cflue_mcq", "dianjin_r1_mcq", "processed/sft/dianjin_r1_mcq.jsonl", "sft"),
           ("DianJin-R1-Data cflue_oe", "dianjin_r1_oe", "processed/sft/dianjin_r1_oe.jsonl", "sft"),
           ("Agentar-DeepFinance-100K", "agentar_deepfinance", "processed/sft/agentar_deepfinance.jsonl", "sft"),
           ("IndustryInstruction 金融", "industryinstruction_finance", "processed/sft/industryinstruction_finance.jsonl", "sft"),
           ("CFLUE train", "train", "interim/cflue/train.jsonl", "item"),
           ("DianJin-Fin-PRM-Data", "dianjin_fin_prm", "interim/prm/dianjin_fin_prm.jsonl", "item"),
           ("alpaca-zh", "alpaca_zh", "processed/sft/alpaca_zh.jsonl", "sft"),
           ("smoltalk-chinese", "smoltalk_chinese", "processed/sft/smoltalk_chinese.jsonl", "sft"),
           ("Tülu 3 SFT", "tulu3", "processed/sft/tulu3.jsonl", "sft")]


def item_text(it):
    return it["question"] + " " + " ".join(f"{k}. {v}" for k, v in sorted(it["choices"].items()))


def prompt_text(r, kind):
    if kind == "item":
        return item_text(r)
    user = next((m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].strip()), "")
    return "\n".join(line for line in user.splitlines() if not WRAPPER.match(line.strip())).strip()


def chinese(text):
    # convert.py's rule for a Chinese sample: at least 30% of the non-space characters are Han
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and len(HAN.findall(text)) / len(chars) >= 0.3


def english(text):
    # Tags and URLs are dropped first: CFLUE items with table images carry escaped markup such as
    # "&lt;imgsrc=&quot;https:..." that made 19 Chinese CFLUE train items count as English (2026-10-25-0844).
    text = TAG_OR_URL.sub(" ", html.unescape(html.unescape(text)))
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and len(ASCII_LETTER.findall(text)) / len(chars) >= 0.5


def numbers(text):
    return set(NUM.findall(unicodedata.normalize("NFKC", text).replace(",", "")))


with open(os.path.join(ROOT, "interim", "embed", "ref.json"), encoding="utf-8") as f:
    meta = json.load(f)
ref = np.load(os.path.join(ROOT, "interim", "embed", "ref.npy")).astype(np.float32)
ref_text = {}
for name in ("gsm8k_test", "ifeval", "financeiq_test", "financeiq_dev", "cflue_val", "cflue_test", "cmmlu_test"):
    with open(os.path.join(ROOT, "processed", "eval", name + ".jsonl"), encoding="utf-8") as f:
        ref_text.update((r["id"], r["question"]) for r in map(json.loads, f))
cols = {k: np.array([j for j, s in enumerate(meta["sets"]) if pred(s)]) for k, pred in
        (("gsm", lambda s: s == "GSM8K test"), ("ifeval", lambda s: s == "IFEval"), ("zh", lambda s: s in ZH_SETS))}


def best_match(vec, key):
    """Best cosine and its reference index over the reference columns ``cols[key]``."""
    c = cols[key]
    best, arg = [], []
    for i in range(0, len(vec), 16384):
        s = vec[i:i + 16384] @ ref[c].T
        best.append(s.max(1))
        arg.append(c[s.argmax(1)])
    return np.concatenate(best), np.concatenate(arg)


lines = ["# 去污染报告：跨语言的同题", "",
         "由 src/01_data-processing/utils/xling_decontam.py 生成。中文样本（汉字占非空白字符 ≥30%）对 GSM8K 测试题，"
         f"向量相似度 ≥{HIGH}，或 ≥{LOW} 且这道题的数字（至少 2 个）全部出现在样本提示中，判为重叠；"
         "依据见 research/log/2026-10-25-0730.md。英文样本（ASCII 字母占非空白字符一半以上）对中文评测集，"
         f"相似度 ≥{EN_ZH} 也判为重叠，依据见 research/log/2026-10-25-0742.md。最后一列只统计，不删。", "",
         f"| 来源 | 中文样本 | GSM8K ≥{HIGH} | GSM8K {LOW}–{HIGH} 且数字全部出现 | 英文样本 | 英文样本对中文评测集 ≥{EN_ZH} | "
         f"标记合计 | 其中 0.90 已标的 | 中文样本对 IFEval ≥{HIGH}（只统计） |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
examples = collections.defaultdict(list)
for name, stem, rel, kind in SOURCES:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        rows = [(r["id"], prompt_text(r, kind)) for r in map(json.loads, f)]
    vec = np.load(os.path.join(ROOT, "interim", "embed", stem + ".npy")).astype(np.float32)
    zh = np.array([chinese(t) for _, t in rows])
    en = np.array([english(t) for _, t in rows])
    flagged, n_high, n_num, ife, en_zh = {}, 0, 0, 0, 0
    if zh.any():
        rz = np.nonzero(zh)[0]
        b, a = best_match(vec[rz], "gsm")
        for r, s, j in zip(rz.tolist(), b.tolist(), a.tolist()):
            sid, text = rows[r]
            gid = meta["ids"][j]
            if s >= HIGH:
                n_high += 1
            elif s >= LOW and len(g := numbers(ref_text[gid])) >= 2 and g <= numbers(text):
                n_num += 1
            else:
                continue
            flagged[sid] = [["GSM8K test", gid, round(s, 3)]]
        b, a = best_match(vec[rz], "ifeval")
        for r, s, j in zip(rz.tolist(), b.tolist(), a.tolist()):
            if s >= HIGH:
                ife += 1
                examples[(name, "IFEval")].append((round(s, 3), rows[r], meta["ids"][j]))
    if en.any():
        re_ = np.nonzero(en)[0]
        b, a = best_match(vec[re_], "zh")
        for r, s, j in zip(re_.tolist(), b.tolist(), a.tolist()):
            if s >= EN_ZH:
                en_zh += 1
                sid = rows[r][0]
                flagged.setdefault(sid, []).append([meta["sets"][j], meta["ids"][j], round(s, 3)])
                examples[(name, "中文评测集")].append((round(s, 3), rows[r], meta["ids"][j]))
    with open(os.path.join(ROOT, "interim", "decontam", f"{os.path.basename(rel)[:-6]}.xling.flags.jsonl"), "w",
              encoding="utf-8") as f:
        for sid, hs in flagged.items():
            f.write(json.dumps({"id": sid, "matches": hs}, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "interim", "decontam", f"{os.path.basename(rel)[:-6]}.embed.flags.jsonl"),
              encoding="utf-8") as f:
        already = {r["id"] for r in map(json.loads, f)}
    lines.append(f"| {name} | {int(zh.sum()):,} | {n_high:,} | {n_num:,} | {int(en.sum()):,} | {en_zh:,} | {len(flagged):,} "
                 f"| {len(already & set(flagged)):,} | {ife:,} |")
    print(name, int(zh.sum()), n_high, n_num, int(en.sum()), en_zh, len(flagged), ife, flush=True)
lines += ["", "## 其他跨语言配对的例子", "", "每格取相似度最高的 5 对，文本截到前 120 字。", ""]
for (name, target), ex in examples.items():
    for s, (sid, text), rid in sorted(ex, key=lambda x: -x[0])[:5]:
        lines.append(f"- {name} 对{target}，{s}：`{rid}` {' '.join(ref_text[rid].split())[:120]} ‖ `{sid}` {' '.join(text.split())[:120]}")
with open(os.path.join(ROOT, "reports", "decontam_xling.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

"""Flag English training samples that overlap the English reference sets by the 8-gram rule of decontam.py.

research/log/2026-10-26-2020.md. The rule, tokenizer, hashing and candidate search are decontam.py's: a
sample overlaps a reference item when 8-grams that also occur in the sample cover at least half of the item's
tokens. The reference text of a FinQA or TAT-QA item is its question plus its report excerpt, so a training
sample that carries the same excerpt is flagged even when it asks another question about it; the other
reference items are the question plus its options. This pass only flags; embed_decontam_en.py adds the
vector match.

Source names after the two roots limit the run to those sources, so sources can run in parallel processes;
each run writes its summary to reports/decontam_en_ngram.<first source>.json.

--strip-format (research/log/2026-10-28-0040.md) first lowercases both sides, drops thousands separators inside
numbers and turns table bars, JSON and other punctuation into spaces, so that a reference item copied with another
table layout ("||Total|" for " | Total | ") is caught; its flags go to <source>.ngram_strip.flags.jsonl.

--rule v2 (research/log/2026-10-28-0040.md, findings 28-32 of interim/decontam/audit/full_chain_audit.json) changes the
rule for the reference items without an excerpt (MMLU, XFinBench, GSM8K, IFEval). The audit found that on these items
the 50% rule flags thousands of samples that copy no item: the digit strings of MMLU miscellaneous-7521 and 7597 match
any long number, IFEval instruction templates ("a title wrapped in double angular brackets") match persona prompts,
option skeletons ("A. 1 B. 2 C. 3 D. 4") match any list of 1 to 4, and in a short item one common 8-gram covers half.
Under v2 a match on such an item that the 50% rule flags is kept only when, counting only countable 8-grams (not those
whose text is one number, such as "00000000" or " $300,000", and on an IFEval item not those found in 3 or more IFEval
items):
  - the shared countable 8-grams cover at least half of the tokens that countable 8-grams cover, and are at least 2
    distinct 8-grams unless the item has only one;
  - an item with options and a question of at least 8 tokens has at least half of its question covered, or at least 20
    countable option tokens of which 80% are covered (a rewritten question with copied options);
or when the shared countable 8-grams cover 90% of the question, whatever the other conditions say. An item with no
countable 8-gram keeps the 50% rule, as do the FinQA and TAT-QA excerpt items. v2 also appends XFinBench's options,
one string, as is; v1 joins that string character by character. The thresholds are the audit's, which checked that no
copy of an item that v1 flags (as rendered here, with "(A)" options, or the question alone) is lost. v2 does drop
rewrites that reword the question and copy short options: of 10 such Tülu 3 rewrites the audit read, both 8-gram checks
lose 8, and 2 of these (evol_codealpaca 58874 for MMLU elementary_mathematics-2190, 103558 for machine_learning-6876)
no other check flags either. The score in each match stays the coverage of the 50% rule. Flags go to
<source>.ngram_v2.flags.jsonl (ngram_strip_v2 with --strip-format) and the summary, with the pairs v2 dropped by
reference set, to reports/decontam_en_ngram_v2.<first source>.json.

Usage: python decontam_en.py <English data root> <Chinese data root> [source ...] [--strip-format] [--rule v2]
"""
import collections
import json
import os
import re
import sys
import unicodedata

import numpy as np
from tokenizers import Tokenizer

EN, ZH = sys.argv[1], sys.argv[2]
STRIP = "--strip-format" in sys.argv[3:]
sys.argv = [a for a in sys.argv if a != "--strip-format"]
RULE = sys.argv.pop(sys.argv.index("--rule") + 1) if "--rule" in sys.argv else "v1"
sys.argv = [a for a in sys.argv if a != "--rule"]
if RULE not in ("v1", "v2"):
    sys.exit(f"--rule takes v1 or v2, not {RULE}")
KIND = ("ngram_strip" if STRIP else "ngram") + ("_v2" if RULE == "v2" else "")
N, THRESHOLD, COMMON = 8, 0.5, 100
TEMPLATE_ITEMS, STEM_COV, OPT_TOKENS, OPT_COV, STEM_KEEP = 3, 0.5, 20, 0.8, 0.9
NUMBER = re.compile(r"^\$?[0-9][0-9.,]*%?$|^[.,][0-9][0-9.,]*%?$")
REF_SETS = [("FinQA test", EN, "processed/eval/finqa_test.jsonl"), ("FinQA dev", EN, "processed/eval/finqa_dev.jsonl"),
            ("TAT-QA test", EN, "processed/eval/tatqa_test.jsonl"), ("TAT-QA dev", EN, "processed/eval/tatqa_dev.jsonl"),
            ("XFinBench test", EN, "processed/eval/xfinbench_test.jsonl"),
            ("XFinBench val", EN, "processed/eval/xfinbench_val.jsonl"),
            ("MMLU test", EN, "processed/eval/mmlu_test.jsonl"), ("GSM8K test", ZH, "processed/eval/gsm8k_test.jsonl"),
            ("IFEval", ZH, "processed/eval/ifeval.jsonl"),
            ("FinQA train 20%", EN, "interim/seed/finqa_train_pref.jsonl"),
            ("TAT-QA train 20%", EN, "interim/seed/tatqa_train_pref.jsonl")]
SOURCES = [(name, EN, f"processed/sft/{name}.jsonl") for name in
           ("finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en",
            "industryinstruction_finance_en", "sujet_finance", "finance_alpaca", "finance_instruct_500k", "doc_qa_en")] + \
          [("tulu3", ZH, "processed/sft/tulu3.jsonl")]
SOURCES = [s for s in SOURCES if os.path.exists(os.path.join(s[1], s[2]))]  # doc_qa_en exists once synthesized
if sys.argv[3:]:
    SOURCES = [s for s in SOURCES if s[0] in sys.argv[3:]]
tok = Tokenizer.from_file(os.path.join(EN, "raw", "qwen2.5-1.5b-tokenizer", "tokenizer.json"))


def norm(text):
    text = unicodedata.normalize("NFKC", text)
    if STRIP:
        text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
        text = re.sub(r"[^\w.%$()-]+|(?<!\d)\.|\.(?!\d)", " ", text)  # keep decimal points, signs and units
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ref_text(it):
    if "context" in it:
        return it["question"] + " " + it["context"]
    choices = it.get("choices") or {}
    if isinstance(choices, str) and RULE == "v2":  # XFinBench; v1 keeps the character join its flags came from
        return it["question"] + " " + choices
    if isinstance(choices, dict):
        choices = [f"{k}. {v}" for k, v in sorted(choices.items())]
    return it["question"] + " " + " ".join(choices)


def gram_hashes(ids):
    a = np.asarray(ids, dtype=np.uint64)
    k = len(a) - N + 1
    if k <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(k, dtype=np.uint64)
    for j in range(N):
        h = h * np.uint64(0x100000001B3) + a[j:j + k]
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return h


def encode(texts, batch=20000):
    for i in range(0, len(texts), batch):
        yield from tok.encode_batch([norm(t) for t in texts[i:i + batch]], add_special_tokens=False)


def v2_keeps(ri, hit):
    """Whether v2 keeps a match that the 50% rule flags; hit marks the item's 8-grams that occur in the sample."""
    grams = refs[ri][3]
    countable, den, n_countable, stem_len, has_choices = strict[ri]
    if not den.any():
        return True
    use = hit & countable
    covered = np.convolve(use.astype(np.int8), ones) > 0
    stem_cov = covered[:stem_len].mean() if stem_len else 0.0
    if stem_cov >= STEM_KEEP:
        return True
    opt = den[stem_len:]
    return bool(covered[den].mean() >= THRESHOLD and (len(np.unique(grams[use])) >= 2 or n_countable <= 1)
                and (not has_choices or stem_len < N or stem_cov >= STEM_COV
                     or (opt.sum() >= OPT_TOKENS and covered[stem_len:][opt].mean() >= OPT_COV)))


refs, ref_stats, no_excerpt = [], [], []
for set_name, root, rel in REF_SETS:
    items = read_jsonl(root, rel)
    texts = [ref_text(it) for it in items]
    short = 0
    for it, e in zip(items, encode(texts)):
        g = gram_hashes(e.ids)
        short += not len(g)
        refs.append((set_name, it["id"], len(e.ids), g))
        if RULE == "v2" and "context" not in it and len(g):
            no_excerpt.append((len(refs) - 1, it, e.tokens))
    ref_stats.append({"set": set_name, "items": len(items), "too_short": short})
postings = collections.defaultdict(list)
for ri, (_, _, _, g) in enumerate(refs):
    for x in np.unique(g).tolist():
        postings[x].append(ri)
ref_sorted = np.sort(np.fromiter(postings.keys(), dtype=np.uint64, count=len(postings)))
ones = np.ones(N, dtype=np.int8)

strict = {}  # v2: reference index -> countable 8-grams, tokens they cover, distinct countable, question length, options
if no_excerpt:
    ifeval_items = collections.Counter(x for s, _, _, g in refs if s == "IFEval" for x in np.unique(g).tolist())
    stem_lens = [len(e.ids) for e in encode([it["question"] for _, it, _ in no_excerpt])]
    for (ri, it, toks), stem_len in zip(no_excerpt, stem_lens):
        set_name, _, n_tok, g = refs[ri]
        texts = ["".join(toks[i:i + N]).replace("Ġ", " ").strip() for i in range(len(g))]
        countable = np.array([not NUMBER.match(t) and not (set_name == "IFEval" and ifeval_items[x] >= TEMPLATE_ITEMS)
                              for t, x in zip(texts, g.tolist())])
        strict[ri] = (countable, np.convolve(countable.astype(np.int8), ones) > 0, len(np.unique(g[countable])),
                      min(stem_len, n_tok), bool(it.get("choices")))

os.makedirs(os.path.join(EN, "interim", "decontam"), exist_ok=True)
summary = []
for name, root, rel in SOURCES:
    rows = read_jsonl(root, rel)
    texts = [" ".join(m["content"] + " " + m.get("reasoning", "") for m in r["messages"]) for r in rows]
    flagged, per_set, dropped = [], collections.Counter(), collections.Counter()
    for r, e in zip(rows, encode(texts)):
        h = gram_hashes(e.ids)
        pos = np.searchsorted(ref_sorted, h)
        pos[pos == len(ref_sorted)] = 0
        shared = np.unique(h[ref_sorted[pos] == h]) if len(h) else h
        cands = set()
        for g in shared.tolist():
            p = postings[g]
            if len(p) <= COMMON:
                cands.update(p)
        hits = []
        for ri in cands:
            set_name, item_id, n_tok, grams = refs[ri]
            hit = np.isin(grams, shared)
            cov = np.count_nonzero(np.convolve(hit.astype(np.int8), ones)) / n_tok
            if cov >= THRESHOLD:
                if ri in strict and not v2_keeps(ri, hit):
                    dropped[set_name] += 1
                    continue
                hits.append((set_name, item_id, round(float(cov), 3)))
        if hits:
            flagged.append({"id": r["id"], "matches": sorted(hits, key=lambda x: -x[2])})
            per_set.update({s for s, _, _ in hits})
    with open(os.path.join(EN, "interim", "decontam", f"{name}.{KIND}.flags.jsonl"), "w", encoding="utf-8") as f:
        for x in flagged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary.append({"source": name, "samples": len(rows), "flagged": len(flagged), "by_set": dict(per_set)})
    if RULE == "v2":
        summary[-1]["v2_dropped_pairs"] = dict(dropped)
    print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
with open(os.path.join(EN, "reports", f"decontam_en_{KIND}.{SOURCES[0][0]}.json"), "w", encoding="utf-8") as f:
    json.dump({"reference_sets": ref_stats, "sources": summary}, f, ensure_ascii=False, indent=1)

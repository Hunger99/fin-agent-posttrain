"""Flag English training samples whose prompt is a rewrite of an English reference item, by bge-m3 cosine similarity.

research/log/2026-10-26-2020.md. The model, vectors (normalized CLS, at most 512 tokens), float32 comparison and
the 0.90 threshold are embed_decontam.py's. The reference text of a FinQA or TAT-QA item is its question followed
by its report excerpt, as in decontam_en.py; the other items' is the question followed by its options. A sample's
text is its first non-empty user turn without the instruction line that convert_en.py puts before FinQA and TAT-QA
samples (Tülu 3 keeps embed_decontam.py's text, so its saved vectors match). Vectors are cached under the English
root's interim/embed/ by a hash of the texts; Tülu 3's are the Chinese root's, linked there, since its prompts
are the same.

--encoder nemotron (research/log/2026-10-27-2316.md) uses Nemotron-3-Embed-1B instead, since bge-m3's weights are
deleted: mean-pooled, L2-normalized vectors of "passage: " + text, at most 512 tokens, with the threshold given by
--threshold; its vectors, flags and report carry ".nemotron" / "_nemotron" in their names, so bge-m3's stay as they
are. --refs-only embeds the reference sets alone and prints, for each item, its best similarity to an item of another
reference set (the calibration of 2316).

research/log/2026-10-28-0040.md (the full-chain audit):
  - XFinBench stores an item's options as one string, which the reference text used to split into single
    characters; it is now the question, a space and that string. The new texts change the hash, so the next run
    embeds the reference sets again; bge-m3's saved reference vectors are of the old texts and its weights are
    deleted, so a bge-m3 run now stops when it would load them.
  - --all-user-turns covers what a later turn of a multi-turn conversation asks (Tülu 3 has 32,203 such samples):
    for the samples with more than one non-empty user turn, each turn is embedded on its own, as the first turn
    is, and a sample's similarity to a reference item is its best over its turns. Samples with one turn are left
    to the run without the option. The turn vectors are cached as <source>.turns; flags and report carry "_turns".
  - --threshold-general and --threshold-finance (both default --threshold) are the thresholds for MMLU, GSM8K
    and IFEval items and for the other items. In the audit's blind judgment, 11 of 60 Tülu 3 samples at
    0.72-0.845 to a general item were rewrites of it, against 1 of 50 finance samples to a FinQA or TAT-QA item.
  - With --shard k/n the report name carries the shard, as the vectors and flags do, so shards no longer
    overwrite one report.

Output: interim/decontam/<source>[.shard<k>of<n>].embed[_nemotron][_turns].flags.jsonl and
        reports/decontam_en_embed[_nemotron][_turns][.<source> ...][.shard<k>of<n>].json.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python embed_decontam_en.py <English data root> <Chinese data root> [source ...]
       [--encoder bge-m3|nemotron] [--threshold 0.90] [--threshold-general t] [--threshold-finance t]
       [--all-user-turns] [--refs-only] [--batch 64] [--shard k/n]
"""
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EN, ZH = sys.argv[1], sys.argv[2]
ARGS = sys.argv[3:]


def option(name, default):
    if name in ARGS:
        i = ARGS.index(name)
        value = ARGS[i + 1]
        del ARGS[i:i + 2]
        return value
    return default


ENCODER = option("--encoder", "bge-m3")
THRESHOLD = float(option("--threshold", "0.90"))
THRESHOLD_GENERAL = float(option("--threshold-general", THRESHOLD))
THRESHOLD_FINANCE = float(option("--threshold-finance", THRESHOLD))
SPLIT = (THRESHOLD_GENERAL, THRESHOLD_FINANCE) != (THRESHOLD, THRESHOLD)
BATCH = int(option("--batch", "64"))
SHARD = option("--shard", None)  # "k/n": every n-th row from row k, for running one large source on several GPUs
REFS_ONLY = "--refs-only" in ARGS
TURNS = "--all-user-turns" in ARGS
ARGS = [a for a in ARGS if a not in ("--refs-only", "--all-user-turns")]
NEMOTRON = os.environ.get("FP_MODELS_D1", "<FP_MODELS_D1 unset>") + "/Nemotron-3-Embed-1B-BF16"
TAG = "" if ENCODER == "bge-m3" else "." + ENCODER  # cache names; bge-m3's keep their original names
REF_SETS = [("FinQA test", EN, "processed/eval/finqa_test.jsonl"), ("FinQA dev", EN, "processed/eval/finqa_dev.jsonl"),
            ("TAT-QA test", EN, "processed/eval/tatqa_test.jsonl"), ("TAT-QA dev", EN, "processed/eval/tatqa_dev.jsonl"),
            ("XFinBench test", EN, "processed/eval/xfinbench_test.jsonl"),
            ("XFinBench val", EN, "processed/eval/xfinbench_val.jsonl"),
            ("MMLU test", EN, "processed/eval/mmlu_test.jsonl"), ("GSM8K test", ZH, "processed/eval/gsm8k_test.jsonl"),
            ("IFEval", ZH, "processed/eval/ifeval.jsonl"),
            ("FinQA train 20%", EN, "interim/seed/finqa_train_pref.jsonl"),
            ("TAT-QA train 20%", EN, "interim/seed/tatqa_train_pref.jsonl")]
GENERAL_SETS = {"MMLU test", "GSM8K test", "IFEval"}  # --threshold-general; the other sets take --threshold-finance
SOURCES = [(name, EN, f"processed/sft/{name}.jsonl") for name in
           ("finqa_train80", "tatqa_train80", "dianjin_r1_finqa", "bizbench", "agentar_deepfinance_en",
            "industryinstruction_finance_en", "sujet_finance", "finance_alpaca", "finance_instruct_500k", "doc_qa_en")] + \
          [("tulu3", ZH, "processed/sft/tulu3.jsonl")]
SOURCES = [s for s in SOURCES if os.path.exists(os.path.join(s[1], s[2]))]  # doc_qa_en exists once synthesized
if ARGS:  # only the named sources, as decontam_en.py; the others' flags are left as they are
    SOURCES = [s for s in SOURCES if s[0] in ARGS]
WRAPPER = ("Answer the question from the excerpt of a company's annual report. Write the calculation, then the "
           "final answer as a number on the last line, after \"Answer:\".")
DOC_WRAPPER = ("Answer the question from the excerpt of a financial document. Write the calculation, then the "
               "final answer as a number on the last line, after \"Answer:\".")  # docqa_to_sft_en.py's instruction
ZH_WRAPPER = re.compile(r"^(假设你是一位金融行业专家，请回答下列问题。|注意：.*|请一步步思考。)$")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (torch.bfloat16 if ENCODER == "nemotron" else torch.float16) if DEVICE == "cuda" else torch.float32
torch.set_num_threads(32)
EMBED_DIR = os.path.join(EN, "interim", "embed")
tokenizer = model = None


def load_encoder():
    """Loaded only when a text has no saved vector, so bge-m3's saved vectors stay usable without its weights."""
    global tokenizer, model
    if model is None:
        path = NEMOTRON if ENCODER == "nemotron" else os.path.join(EN, "raw", "bge-m3")
        if not os.path.isdir(path):  # from_pretrained would take the missing path for a Hub name
            sys.exit(f"{path} is missing, and some texts have no saved vector")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path, dtype=DTYPE).to(DEVICE).eval()


def read_jsonl(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ref_text(it):
    if "context" in it:
        return it["question"] + "\n" + it["context"]
    choices = it.get("choices") or ""
    if isinstance(choices, dict):  # MMLU; XFinBench keeps "A. ...\nB. ..." as one string
        choices = " ".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    return it["question"] + " " + choices


def user_turns(r):
    return [m["content"] for m in r["messages"] if m["role"] == "user" and m["content"].strip()]


def clean(user, name):
    if name == "tulu3":  # embed_decontam.py's text, so its saved vectors are reused
        return "\n".join(line for line in user.splitlines() if not ZH_WRAPPER.match(line.strip())).strip()
    return user.replace(WRAPPER, "", 1).replace(DOC_WRAPPER, "", 1).strip()


def prompt_text(r, name):
    turns = user_turns(r)
    return clean(turns[0], name) if turns else ""


@torch.inference_mode()
def embed(texts, batch=BATCH):
    load_encoder()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = None
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        chunk = [("passage: " if ENCODER == "nemotron" else "") + texts[j] for j in idx]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        hidden = model(**enc).last_hidden_state
        if ENCODER == "nemotron":  # average over the real tokens
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1)
        else:  # bge-m3: the CLS vector
            vec = hidden[:, 0]
        if out is None:
            out = torch.empty(len(texts), vec.shape[-1], dtype=DTYPE, device=DEVICE)
        out[idx] = torch.nn.functional.normalize(vec.float(), dim=-1).to(DTYPE)
        if i // batch % 500 == 0:
            print(f"  embedded {i + len(idx):,}/{len(texts):,}", flush=True)
    return out


def embed_cached(name, texts):
    """Vectors of texts, reused only when they were saved from exactly these texts.

    Vectors and hash are each written to a temporary file and renamed, the hash last, so a run that stops midway
    leaves no hash beside partly written vectors, and shards that embed the reference sets at once replace whole
    files.
    """
    name += TAG
    digest = hashlib.sha256("\0".join(texts).encode("utf-8")).hexdigest()
    npy, info = os.path.join(EMBED_DIR, name + ".npy"), os.path.join(EMBED_DIR, name + ".json")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            if json.load(f)["sha256"] == digest:
                saved = np.load(npy)
                if len(saved) == len(texts):
                    print(f"  reusing {npy}", flush=True)
                    return torch.from_numpy(saved).to(DEVICE, DTYPE)
    emb = embed(texts)
    os.makedirs(EMBED_DIR, exist_ok=True)
    tmp = f"{npy}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, emb.to(torch.float16).cpu().numpy())
    os.replace(tmp, npy)
    tmp = f"{info}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"rows": len(texts), "sha256": digest}, f)
    os.replace(tmp, info)
    return emb


def reference_items():
    ref_ids, ref_sets, ref_texts = [], [], []
    for set_name, root, rel in REF_SETS:
        for it in read_jsonl(root, rel):
            ref_ids.append(it["id"])
            ref_sets.append(set_name)
            ref_texts.append(ref_text(it))
    return ref_ids, ref_sets, ref_texts


def thresholds(ref_sets):
    """One threshold per reference item, float32 as the similarities, so that a single threshold compares as before."""
    return torch.tensor([THRESHOLD_GENERAL if s in GENERAL_SETS else THRESHOLD_FINANCE for s in ref_sets],
                        dtype=torch.float32, device=DEVICE)


def flag(emb, owner, ids, ref32, thr, ref_sets, ref_ids):
    """Flags of the samples ``ids``; row j of ``emb`` is a text of sample ``owner[j]``, with ``owner`` ascending.

    A sample's similarity to a reference item is its best over its rows. Samples go in chunks of at most 4096 rows,
    so one row per sample gives the chunks, and the results, of the run before --all-user-turns.
    """
    flagged, per_set, bins = [], collections.Counter(), collections.Counter()
    starts = np.searchsorted(owner, np.arange(len(ids) + 1))  # the rows of sample s are starts[s]:starts[s + 1]
    i = 0
    while i < len(ids):
        j = max(i + 1, int(np.searchsorted(starts, starts[i] + 4096, side="right")) - 1)
        a, b = int(starts[i]), int(starts[j])
        sims = emb[a:b].float() @ ref32.T
        if b - a > j - i:
            seg = torch.from_numpy(owner[a:b] - i).to(DEVICE)[:, None].expand_as(sims)
            sims = torch.full((j - i, sims.shape[1]), -torch.inf, device=DEVICE).scatter_reduce_(0, seg, sims, "amax")
        top = sims.max(1).values
        bins.update(sum(t >= e for e in (0.8, 0.85, 0.9, 0.95)) for t in top.tolist())
        hit_rows, hit_cols = (sims >= thr).nonzero(as_tuple=True)
        hits = collections.defaultdict(list)
        for r_i, c_i in zip(hit_rows.tolist(), hit_cols.tolist()):
            hits[r_i].append((ref_sets[c_i], ref_ids[c_i], round(float(sims[r_i, c_i]), 3)))
        for r_i, hs in hits.items():
            flagged.append({"id": ids[i + r_i], "matches": sorted(hs, key=lambda x: -x[2])[:5]})
            per_set.update({s for s, _, _ in hs})
        i = j
    return flagged, per_set, bins


def main():
    ref_ids, ref_sets, ref_texts = reference_items()
    ref32 = embed_cached("ref_en", ref_texts).float()
    if REFS_ONLY:  # each reference item's best similarity to an item of another reference set
        sets = np.array(ref_sets)
        best = []
        for i in range(0, len(ref_ids), 2048):
            sims = ref32[i:i + 2048] @ ref32.T
            same = torch.from_numpy(sets[i:i + 2048, None] == sets[None, :]).to(DEVICE)
            best += sims.masked_fill(same, -1).max(1).values.tolist()
        best = np.array(best)
        print(json.dumps({"encoder": ENCODER, "items": len(best),
                          "share_at_or_above": {t: round(float((best >= t).mean()), 5) for t in (0.8, 0.85, 0.9, 0.95)},
                          "quantiles": {q: round(float(np.quantile(best, q)), 4)
                                        for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)}}))
        np.save(os.path.join(EMBED_DIR, f"ref_en_best_other{TAG}.npy"), best)
        return

    thr = thresholds(ref_sets)
    suffix = "_turns" if TURNS else ""
    summary = []
    os.makedirs(os.path.join(EN, "interim", "decontam"), exist_ok=True)
    for source, root, rel in SOURCES:
        rows, name = read_jsonl(root, rel), source
        if SHARD:
            k, n = map(int, SHARD.split("/"))
            rows, name = rows[k::n], f"{source}.shard{k}of{n}"
        if TURNS:
            rows = [r for r in rows if len(user_turns(r)) > 1]
            turns = [[clean(u, source) for u in user_turns(r)] for r in rows]
            texts = [t for ts in turns for t in ts]
            owner = np.repeat(np.arange(len(rows)), [len(ts) for ts in turns])
        else:
            texts = [prompt_text(r, source) for r in rows]
            owner = np.arange(len(rows))
        emb = embed_cached(name + (".turns" if TURNS else ""), texts) if texts else None
        flagged, per_set, bins = flag(emb, owner, [r["id"] for r in rows], ref32, thr, ref_sets, ref_ids)
        flags = f"{name}.embed{'' if ENCODER == 'bge-m3' else '_' + ENCODER}{suffix}.flags.jsonl"
        with open(os.path.join(EN, "interim", "decontam", flags), "w", encoding="utf-8") as f:
            for x in flagged:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        summary.append({"source": name, "samples": len(rows), "flagged": len(flagged), "by_set": dict(per_set),
                        "best_similarity_bins": {k: bins[i] for i, k in enumerate(["<0.80", "0.80-0.85", "0.85-0.90",
                                                                                    "0.90-0.95", ">=0.95"])}})
        if TURNS:
            summary[-1]["turns"] = len(texts)
        if SPLIT:
            summary[-1]["thresholds"] = {"general": THRESHOLD_GENERAL, "finance": THRESHOLD_FINANCE}
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    report = ("decontam_en_embed" + ("" if ENCODER == "bge-m3" else "_" + ENCODER) + suffix
              + "".join("." + a for a in ARGS) + (".shard{}of{}".format(*SHARD.split("/")) if SHARD else "") + ".json")
    with open(os.path.join(EN, "reports", report), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

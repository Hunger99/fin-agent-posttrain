"""Build the CFLUE selection and cross-check sets without FinanceIQ test items, and total what decontamination keeps.

Decided in research/log/2026-10-25-0039.md: CFLUE items used for selection or cross-checking drop every
item that the 8-gram or the embedding match ties to a FinanceIQ test item, since selecting on those
would read the test set. Training sources lose every sample either match ties to any evaluation set,
and SFT sources also every sample that repeats an item of CFLUE train's held-out 20% (split_cflue.py).

Usage: python select_sets.py $FP_DATA_ZH
"""
import collections
import json
import os
import sys

ROOT = sys.argv[1]
FLAGS = os.path.join(ROOT, "interim", "decontam")
EVAL_SETS = {"FinanceIQ test", "FinanceIQ dev", "CFLUE val", "CFLUE test", "CMMLU test", "GSM8K test", "IFEval"}
HELDOUT = {"CFLUE 留出 20%（8-gram）", "CFLUE 留出 20%（向量）"}  # split_cflue.py
INHERIT = {"CFLUE 训练题已删"}  # split_cflue.py: DianJin-R1-Data cflue_mcq samples whose CFLUE item was dropped
DEDUP = {"精确重复", "近似重复"}  # dedup_sft.py: SFT samples whose user prompts repeat a kept sample's
NOT_SFT = {"train", "dianjin_fin_prm"}  # seed questions and the preference supplement, not deduplicated as SFT
N_BUDGET = 100_000  # single-run SFT sample budget N, adjudicated 2026-10-25 (docs/history/decisions.md item 1)
CAP = 0.40  # largest share of a pool one source may take, docs/history/01_data-processing.md section 7
# (name, file stem, pool). CFLUE train enters the domain pool only through its 80% part (section 5 of
# the data design); its flags cover the whole split, so they are cut to the ids of that part.
SOURCES = [("DianJin-R1-Data cflue_mcq", "dianjin_r1_mcq", "领域"), ("DianJin-R1-Data cflue_oe", "dianjin_r1_oe", "领域"),
           ("Agentar-DeepFinance-100K", "agentar_deepfinance", "领域"),
           ("IndustryInstruction 金融", "industryinstruction_finance", "领域"), ("CFLUE train", "train", "领域"),
           ("DianJin-Fin-PRM-Data", "dianjin_fin_prm", "偏好补充"), ("alpaca-zh", "alpaca_zh", "通用"),
           ("smoltalk-chinese", "smoltalk_chinese", "通用"), ("Tülu 3 SFT", "tulu3", "通用")]
SOURCE_FILES = {"train": "interim/cflue/train80.jsonl", "dianjin_fin_prm": "interim/prm/dianjin_fin_prm.jsonl"}


def flagged(stem, kind, sets):
    """Ids flagged against ``sets``, or None when that match has not run on this source."""
    path = os.path.join(FLAGS, f"{stem}.{kind}.flags.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {r["id"] for r in rows if any(m[0] in sets for m in r["matches"])}


def count_lines(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return sum(1 for _ in f)


lines = ["# 去污染汇总", "", "由 src/01_data-processing/utils/select_sets.py 生成，合并 decontam.md（8-gram）和 decontam_embed.md（向量）两步的标记。", "",
         "## 选择集和交叉检验集", "", "去掉与 FinanceIQ 测试集重叠的题（两种匹配任一命中）。", "",
         "| 集合 | 原题数 | 去掉 | 剩下 | 剩下的单选题 | 输出文件 |", "| --- | --- | --- | --- | --- | --- |"]
for split, out in (("val", "cflue_val_select"), ("test", "cflue_test_check")):
    with open(os.path.join(ROOT, "processed", "eval", f"cflue_{split}.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    drop = flagged(f"cflue_{split}", "ngram", {"FinanceIQ test"}) | flagged(f"cflue_{split}", "embed", {"FinanceIQ test"})
    keep = [it for it in items if it["id"] not in drop]
    with open(os.path.join(ROOT, "processed", "eval", f"{out}.jsonl"), "w", encoding="utf-8") as f:
        for it in keep:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    single = sum(it["task"] == "单项选择题" for it in keep)
    lines.append(f"| CFLUE {split} | {len(items):,} | {len(items) - len(keep):,} | {len(keep):,} | {single:,} | processed/eval/{out}.jsonl |")


def file_ids(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


lines += ["", "## 训练来源去污染后剩下的条数", "",
          "\"其中只被向量标记\"是 8-gram 没有标记、只被向量相似度标记的样本，即改写过的题。"
          "\"留出 20%\"是与 CFLUE 训练集留出部分重叠的样本（split_cflue.py，见 heldout_decontam.md）；"
          "CFLUE 训练集只算 80% 部分，这一列是其中与留出部分同题的题。DianJin-Fin-PRM-Data 不进 SFT，不按留出部分删。"
          "\"随 CFLUE 题删\"是 DianJin-R1-Data 选择题部分中，最相近的 CFLUE 训练题已被删掉的样本。"
          "\"跨语言\"是中文样本中 GSM8K 测试题的翻译，按 xling_decontam.py 的规则标出（见 decontam_xling.md），含向量已标的。"
          "\"提示重复\"是去污染之后、用户提示与另一条留下的样本相同或近似的样本（dedup_sft.py，见 dedup_sft.md）。", "",
          "| 来源 | 池 | 条数 | 8-gram 标记 | 向量标记 | 其中只被向量标记 | 跨语言 | 留出 20% | 随 CFLUE 题删 | 提示重复 | 合计删掉 | 剩下 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
pool, kept_ids, removed_ids, kept_counts = collections.Counter(), {}, {}, {}
for name, stem, kind in SOURCES:
    rel = SOURCE_FILES.get(stem, f"processed/sft/{stem}.jsonl")
    total = count_lines(rel)
    ng, em = flagged(stem, "ngram", EVAL_SETS), flagged(stem, "embed", EVAL_SETS)
    xl = flagged(stem, "xling", EVAL_SETS)
    ho = set() if kind == "偏好补充" else flagged(stem, "heldout", HELDOUT)
    inh = flagged(stem, "inherit", INHERIT) if stem == "dianjin_r1_mcq" else set()
    if stem == "train":
        part = file_ids(rel)
        ng, em, ho = ng & part, (None if em is None else em & part), (None if ho is None else ho & part)
        xl = None if xl is None else xl & part
    removed = ng | (em or set()) | (xl or set()) | (ho or set()) | (inh or set())
    # Decontamination alone is written out for dedup_sft.py, which deduplicates only what it keeps
    removed_ids[stem] = sorted(removed)
    dd = None if stem in NOT_SFT else flagged(stem, "dedup", DEDUP)
    dd_cell = "—" if stem in NOT_SFT else ("未跑" if dd is None else f"{len(dd - removed):,}")
    removed = removed | (dd or set())
    kept = total - len(removed)
    pool[kind] += kept
    if kind == "领域":
        kept_counts[stem] = kept
    if stem in ("train", "dianjin_r1_mcq"):
        kept_ids[stem] = file_ids(rel) - removed
    em_cells = "未跑 | 未跑" if em is None else f"{len(em):,} | {len(em - ng):,}"
    ho_cell = "—" if kind == "偏好补充" else ("未跑" if ho is None else f"{len(ho):,}")
    inh_cell = "—" if stem != "dianjin_r1_mcq" else ("未跑" if inh is None else f"{len(inh):,}")
    xl_cell = "未跑" if xl is None else f"{len(xl):,}"
    lines.append(f"| {name} | {kind} | {total:,} | {len(ng):,} | {em_cells} | {xl_cell} | {ho_cell} | {inh_cell} "
                 f"| {dd_cell} | {len(removed):,} | {kept:,} |")
with open(os.path.join(FLAGS, "removed_ids.json"), "w", encoding="utf-8") as f:
    json.dump(removed_ids, f, ensure_ascii=False)
deduped = all(os.path.exists(os.path.join(FLAGS, f"{stem}.dedup.flags.jsonl")) for _, stem, _ in SOURCES if stem not in NOT_SFT)
# A kept cflue_mcq sample whose nearest CFLUE train item (split_cflue.py) is kept in the 80% part is the
# same question a second time, with R1 reasoning.
with open(os.path.join(ROOT, "interim", "cflue", "dianjin_mcq_nearest.jsonl"), encoding="utf-8") as f:
    twice = sum(n["id"] in kept_ids["dianjin_r1_mcq"] and n["same_item"] and n["cflue_id"] in kept_ids["train"]
                for n in map(json.loads, f))
once = pool["领域"] - twice


def capacity(counts):
    """Largest run drawable without repeats when no source may exceed CAP of it (water-filling)."""
    capped, m = set(), sum(counts.values())
    while True:
        over = {s for s, n in counts.items() if s not in capped and n > CAP * m}
        if not over:
            return m
        capped |= over
        m = sum(n for s, n in counts.items() if s not in capped) / (1 - CAP * len(capped))


# Each CFLUE question once: the cflue_mcq twins are left out, as the "once" count above leaves them out
domain_once = {stem: (kept - twice if stem == "dianjin_r1_mcq" else kept) for stem, kept in kept_counts.items()}
cap_m = capacity(domain_once)
lines += ["", f"领域池不含合成数据，CFLUE 训练集只算 80% 部分，约 {pool['领域']:,} 条；"
          f"同一道 CFLUE 题只算一次时（DianJin-R1-Data 选择题部分与 CFLUE 训练集是同一批题）约 {once:,} 条。"
          f"单个来源最多占 {CAP:.0%}，不重复使用样本，同题只算一次时，一次运行最多能取约 {cap_m:,.0f} 条领域样本，"
          f"是单次 SFT 样本预算 N = {N_BUDGET:,} 条的 {cap_m / N_BUDGET:.2f} 倍。"
          + ("SFT 来源已按用户提示去重；DianJin-R1-Data 选择题部分与 CFLUE 训练集之间的同题按只算一次处理。"
             if deduped else "这些数都在去重之前。")
          + f"通用池约 {pool['通用']:,} 条。"]
with open(os.path.join(ROOT, "reports", "decontam_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

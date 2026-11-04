"""Turn kept English document-QA questions into SFT samples: document excerpt and question -> calculation and answer.

research/log/2026-10-26-2222.md. The author chose (2026-10-26) the open-book form, the same task as the English
domain score, so the user turn is the instruction, the chunk and the question, in num_eval.py's layout with
"Document" for "Report", and the assistant turn is the teacher's calculation and its numeric answer. A question
is taken when doc_qa_en.py kept it; exact duplicates of the normalized question keep their first occurrence in
chunk-id order. Decontamination and dedup against the other sources then run as for every English source
(decontam_en.py, embed_decontam_en.py), so "source" is "doc_qa_en (synthetic)" and R3-synthetic can remove it whole.

Usage: python docqa_to_sft_en.py <output.jsonl> <doc_qa_en output.jsonl>...
"""
import json
import re
import sys

INSTRUCTION = ("Answer the question from the excerpt of a financial document. Write the calculation, then the final "
               "answer as a number on the last line, after \"Answer:\".")
out_path, raws = sys.argv[1], sys.argv[2:]
rows = [json.loads(line) for p in raws for line in open(p, encoding="utf-8")]
rows.sort(key=lambda r: r["chunk_id"])
seen, out, n_kept = set(), [], 0
for r in rows:
    for j, q in enumerate(r["questions"]):
        if not q.get("kept"):
            continue
        n_kept += 1
        key = re.sub(r"\s+", " ", q["question"]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        user = f"{INSTRUCTION}\n\nDocument:\n{r['chunk'].strip()}\n\nQuestion: {q['question'].strip()}"
        answer = f"Calculation: {q['calculation'].strip()}\nAnswer: {q['answer_value']:g}"
        out.append({"id": f"{r['chunk_id']}#q{j}", "source": "doc_qa_en (synthetic)",
                    "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
                    "meta": {"license": "generated from IndustryCorpus2 (Apache-2.0) by gpt-oss-120b", "synthetic": True,
                             "chunk_id": r["chunk_id"], "unit": q.get("unit"), **r["meta"]}})
with open(out_path, "w", encoding="utf-8") as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(json.dumps({"chunks": len(rows), "kept_questions": n_kept, "after_exact_dedup": len(out)}))

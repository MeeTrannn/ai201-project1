"""End-to-end smoke test over the planning.md Evaluation Plan questions."""
from embed_index import retrieve
from generator import generate_answer

CASES = [
    ("What is the name of session 3 in CS110?", "Analyzing elementary sorting algorithms", "Session 3"),
    ("What is the learning outcome of CS114?", "list of #-tags", None),
    ("What are the course prerequisites of CS166?", "CS114 or CS130; CS110", None),
    ("What is the reading and learning materials of session 1 in CS130?", "Kleinberg / Simpson's Paradox / King video", "Session 1"),
    ("How many assignments in CS156 and what are they?", "3: Pipeline First/Second/Final Draft", None),
]

for i, (q, expected, sid) in enumerate(CASES, 1):
    chunks = retrieve(q, top_k=8, session_id=sid)
    files = {c.get("filename") for c in chunks[:4]}
    ans = generate_answer(q, chunks, max_context_chunks=4)
    print(f"\n{'='*90}\nQ{i}: {q}\nExpected: {expected}\nTop-4 source files: {files}\n--- ANSWER ---\n{ans}")

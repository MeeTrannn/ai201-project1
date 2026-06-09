# The Unofficial Guide — Project 1

A retrieval-augmented question-answering system over university CS course syllabi.
Ask a natural-language question (e.g. *"What are the prerequisites of CS166?"*) and get a
grounded, source-cited answer drawn only from the indexed syllabus documents.

**Pipeline:** Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
**Stack:** `pdfplumber` · custom heading-aware chunker · `sentence-transformers` (all-MiniLM-L6-v2) · ChromaDB · Groq (`llama-3.3-70b-versatile`) · Gradio

### Run it
```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env   # free key at console.groq.com (.env is gitignored)
python ingest.py --reset      # chunk + embed + index all syllabi (one time)
python app.py                 # launches the Gradio UI at http://127.0.0.1:7860
```

---

## Domain

This system covers the **course syllabi for the CS major at Minerva University** — CS110, CS111,
CS113, CS114, CS130, CS146, CS156, CS162, CS164, and CS166.

This knowledge is valuable but hard to extract through official channels because syllabus content
is spread across ten separate, densely formatted documents. A student trying to answer practical
planning questions — *"Which courses must I take before CS166?"*, *"How many assignments does CS156
have and when are they due?"*, *"What readings do I need before the first session of CS130?"* — would
otherwise have to open each PDF and manually scan for the relevant section. The information exists
but is fragmented, inconsistently formatted, and not searchable across courses. This system makes
that cross-document knowledge queryable in natural language with exact source attribution.

---

## Document Sources

All documents are official course syllabi, exported to plain text and stored in [`documents/`](documents/).

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | CS110 — Problem Solving with Data Structures and Algorithms | Syllabus (.txt) | `documents/cs110-syllabus.txt` |
| 2 | CS111 — Computation: Solving Problems with Algorithms | Syllabus (.txt) | `documents/cs111-syllabus.txt` |
| 3 | CS113 | Syllabus (.txt) | `documents/cs113-syllabus.txt` |
| 4 | CS114 — Statistics / Probability | Syllabus (.txt) | `documents/cs114-syllabus.txt` |
| 5 | CS130 — Statistical Modeling & Prediction | Syllabus (.txt) | `documents/cs130-syllabus.txt` |
| 6 | CS146 — Modeling, Simulation & Decision Making | Syllabus (.txt) | `documents/cs146-syllabus.txt` |
| 7 | CS156 — Machine Learning | Syllabus (.txt) | `documents/cs156-syllabus.txt` |
| 8 | CS162 | Syllabus (.txt) | `documents/cs162-syllabus.txt` |
| 9 | CS164 | Syllabus (.txt) | `documents/cs164-syllabus.txt` |
| 10 | CS166 — Modeling, Analysis & Optimization of Complex Systems | Syllabus (.txt) | `documents/cs166-syllabus.txt` |

PDFs (where applicable) were converted to text with [`extract_pdfs.py`](extract_pdfs.py) using `pdfplumber`.

---

## Chunking Strategy

Implemented in [`chunker.py`](chunker.py) (`chunk_text(text, chunk_size=400, overlap=50)`).

**Chunk size:** 400 tokens (≈ 2.5–3k characters), with a hard ceiling of ~420 tokens.

**Overlap:** ~50 tokens of trailing sentences carried into the next chunk.

**Preprocessing before chunking:**
- Normalized line endings (`\r\n` → `\n`).
- Detected **session headings** with the regex `^Session\s+\d+` and split the document into one
  segment per session.
- **Captured a "Course Overview" segment for everything before the first session heading.** This was
  a deliberate fix (see Spec Reflection): course-level content — Course Description, Objectives &
  Learning Outcomes, and Prerequisites — all appears *before* "Session 1", and an earlier version of
  the chunker silently dropped it.
- Within each segment, split on paragraph boundaries first, then sentence boundaries if a single
  paragraph exceeds the chunk size, so chunks never break mid-sentence.

**Why these choices fit the documents:** Syllabi are highly structured — a course-level preamble
followed by per-session blocks of learning outcomes, readings, and materials. A typical session is
shorter than 400 tokens, so heading-aware splitting keeps most sessions intact in a single chunk
(ideal for retrieval), while the size guardrail handles the few long sessions. The small 50-token
overlap preserves continuity of bullet lists and readings that span a boundary without much
redundancy. Each chunk carries metadata (`filename`, `top_level_heading`, `chunk_index`,
`char_start`, `char_end`, `token_count`) for precise attribution.

**Final chunk count:** **312 chunks** across the 10 syllabi.

| Document | Chunks | | Document | Chunks |
|---|---|---|---|---|
| cs110 | 30 | | cs146 | 31 |
| cs111 | 31 | | cs156 | 41 |
| cs113 | 29 | | cs162 | 29 |
| cs114 | 31 | | cs164 | 27 |
| cs130 | 32 | | cs166 | 31 |

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, stored in a persistent ChromaDB
collection (`embed_index.py`). Retrieval embeds the query, fetches a candidate pool by cosine
similarity, applies heading/structure-aware reranking, deduplicates near-identical chunks (cosine
> 0.95), and returns the top results.

This model is fast, runs locally with no API cost, and performs well on the structured English text
of these syllabi. For a small/medium corpus (312 chunks) it gives near-instant retrieval.

**Production tradeoff reflection:** If deploying for real users with no cost constraint, I would weigh:
- **Accuracy on paraphrase-heavy queries:** `all-MiniLM-L6-v2` (384-dim) can miss semantic matches
  when wording differs from the source. `all-mpnet-base-v2` or a hosted embedding (e.g. OpenAI
  `text-embedding-3-large`) offers higher semantic fidelity at higher latency/cost.
- **Hybrid retrieval:** Because the syllabi are near-identical in structure, dense similarity alone
  confused courses (see Failure Case). Adding BM25/keyword retrieval would improve recall on exact
  identifiers like course codes and assignment names.
- **Context length:** Syllabus chunks are short, so the 256-token input window of MiniLM is rarely a
  constraint here — but a longer-context embedder would matter for denser source documents.
- **Latency vs. local control:** A local model keeps data private and removes per-query cost, which
  is attractive even if a hosted model is marginally more accurate.

---

## Grounded Generation

Implemented in [`generator.py`](generator.py) (`generate_answer`).

**System prompt grounding instruction (verbatim):**
> "You are a helpful, conservative assistant that answers questions using only the provided context.
> Always cite exact sources using the format `[filename#chunk_index]` immediately after any factual
> claim. If the context does not contain the answer, reply exactly: 'I don't know' (without
> additional information). When asked for lists, return a bulleted list and cite the relevant
> chunk(s) after each item. At the end include a 'Sources:' section listing unique chunk references
> used. Be concise and avoid hallucination."

**Structural choices that enforce grounding:**
- The user message restates *"using ONLY the context blocks. Do not use any outside knowledge."*
- Retrieved chunks are concatenated with explicit delimiters
  `<<CONTEXT n: filename | heading | chunk_index>>` … `<<ENDCONTEXT>>`, so the model can attribute
  each fact to a specific chunk.
- Generation runs at **temperature 0** for deterministic, conservative output.
- Only the top-4 retrieved chunks are passed in, filtering low-relevance context.
- **Post-processing validates citations:** any `[filename#chunk_index]` the model emits that isn't
  among the chunks actually provided is dropped from the appended `Sources:` list, so provenance
  can't reference context that wasn't supplied.
- If retrieval returns nothing, the function short-circuits to the exact fallback `I don't know`.

**How source attribution is surfaced:** Inline `[filename#chunk_index]` citations after each claim,
plus a consolidated `Sources:` section. The Gradio UI ([`app.py`](app.py)) additionally renders a
"Retrieved context" panel showing each source chunk's filename, heading, similarity score, and a
snippet, so users can verify the grounding themselves.

---

## Evaluation Report

All five questions were run end-to-end (retrieve → generate) via [`eval_smoke.py`](eval_smoke.py).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Name of session 3 in CS110 | Analyzing elementary sorting algorithms | "[2.1] Analyzing elementary sorting algorithms" `[cs110-syllabus.txt#7]` | Relevant | Accurate |
| 2 | Learning outcomes of CS114 | List of `#`-tagged outcomes (probability/statistics, model & parameter estimation, math tools, professionalism, etc.) | Bulleted outcomes from the Course Overview: apply probability/statistics concepts, math tools, determine which model/parameters generated data, cited to `[cs114-syllabus.txt#1]` | Relevant | Partially accurate |
| 3 | Course prerequisites of CS166 | CS114 or CS130 (statistical models); CS110 (OOP in Python) | "CS114 or CS130 … understanding/creating statistical models; CS110 … writing object-oriented programs in Python" `[cs166-syllabus.txt#2]` | Relevant | Accurate |
| 4 | Reading/materials of session 1 in CS130 | Kleinberg et al.; Simpson's Paradox (Wikipedia); King "Overview" video | Listed Session 1 materials (Howard video, Esther Duflo TED talk, decision-memo guidelines) plus Course-Overview prep readings (Rubin; James et al.), cited to `#5` and `#0` | Relevant | Partially accurate (see below) |
| 5 | How many assignments in CS156 and what are they | 3: Pipeline First / Second / Final Draft | "There are 3 assignments: Pipeline – First Draft, Second Draft, Final Draft" `[cs156-syllabus.txt#0, #1]` | Relevant | Accurate |

**Summary:** 3/5 fully accurate, 2/5 partially accurate. After adding course-code filtering, all five
questions retrieve from the correct course document (early versions retrieved the wrong course
entirely). Q2 is paraphrased rather than reproducing the exact `#`-tag list because those tags are
spread across multiple Course-Overview chunks. Q4's mismatch is analyzed below.

---

## Failure Case Analysis

**Question that failed:** Q4 — *"How many learning materials for session 1 in CS130?"*

**What the system returned:** I don't know

**Root cause (tied to a specific pipeline stage):** I think it is the limited context awareness. Looking at the retrieved context chunks, I see that the system correctly retrieved relevant chunks in session 1 of CS130 course but does not look at all learning materias all at once in a chunk in order to perform counting tasks.

However, if I ask *"What are the learning materials for session 1 in CS130?"* then the system correctly return all learning materials.

**What I would change to fix it:** 

I am not exactly sure how to teach AI how to count, maybe I will have to write a specific function to teach the LLM how to count in the repo.

---

## Spec Reflection

**One way the spec helped you during implementation:** The Retrieval Approach section of `planning.md`
explicitly anticipated that near-identical syllabi would cause cross-course confusion ("Anticipated
Challenges" #2) and prescribed heading-aware filtering and structure-sensitive scoring. When initial
testing showed retrieval returning the *wrong course* for 4 of 5 questions, that section told me
exactly where to look: I implemented course-code detection → ChromaDB `filename` metadata filter,
plus session/Course-Overview boosting, rather than just swapping the embedding model. Having written
the retrieval rules in advance turned a vague "retrieval is bad" symptom into a targeted fix.

**One way your implementation diverged from the spec, and why:** The spec's chunker only described
heading-aware splitting on session boundaries. In practice that silently *dropped every document's
preamble* — Course Description, Learning Outcomes, and Prerequisites all live before "Session 1" —
which made Q2 and Q3 impossible to answer. I diverged by adding an explicit "Course Overview"
segment for pre-first-heading content. I also changed the reranking from a *multiplicative* heading
boost (`score * 1.2`) to an *additive* one, because Chroma similarity scores can be negative, and
multiplying a negative score by 1.2 pushed correct matches further down instead of up.

---

## AI Usage

**Instance 1 — Implementing and debugging the chunker**

- *What I gave the AI:* The Chunking Strategy section of `planning.md` (400-token chunks, 50-token
  overlap, heading-aware) and a sample syllabus, asking it to implement `chunk_text()` and verify it.
- *What it produced:* A heading-aware `chunk_text()` that split on `^Session \d+` headings with
  sentence-boundary fallback and token-overlap, plus a self-test harness.
- *What I changed or overrode:* Testing revealed it discarded all text before the first "Session 1"
  heading — i.e. the entire course preamble with Learning Outcomes and Prerequisites. I directed it to
  add a "Course Overview" segment capturing pre-first-heading content. This single change unblocked
  the CS114 learning-outcomes and CS166 prerequisites questions and raised CS110 from 25 → 30 chunks.

**Instance 2 — Fixing retrieval ranking**

- *What I gave the AI:* The Retrieval Approach section (top-8 → rerank → top-4, heading boost,
  dedup), the `embed_index.py` retriever, and the failing eval output showing wrong-course results.
- *What it produced:* A diagnosis identifying three bugs — (1) an invalid `"ids"` value passed to
  ChromaDB's `include` that crashed every query, (2) the multiplicative boost mis-ranking negative
  similarity scores, and (3) no course-awareness at all.
- *What I changed or overrode:* I had it add course-code detection with a ChromaDB `filename`
  metadata filter (with a global fallback), switch to additive boosts, fetch a larger candidate pool
  so boosted chunks can surface, and use precise session matching so "Session 1" doesn't match
  "Session 13." After these changes all five eval questions retrieved from the correct course.

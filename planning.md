# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

---

## Documents

| # | Description | URL or location |
|---|-------------|-----------------|
| 1 |Syllabus of course CS110 |documents/cs110-syllabus.txt |
| 2 |Syllabus of course CS111 |documents/cs111-syllabus.txt |
| 3 |Syllabus of course CS1134 |documents/cs114-syllabus.txt |
| 5 |Syllabus of course CS130 |documents/cs130-syllabus.txt |
| 6 |Syllabus of course CS146 |documents/cs146-syllabus.txt |
| 7 |Syllabus of course CS156 |documents/cs156-syllabus.txt |
| 8 |Syllabus of course CS162 |documents/cs162-syllabus.txt |
| 9 |Syllabus of course CS164 |documents/cs164-syllabus.txt |
| 10 |Syllabus of course CS166|documents/cs166-syllabus.txt |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:** 400 tokens (≈ 2.5–3k characters). If a single logical section (e.g., a full session) exceeds 800 tokens, split into sequential 400-token chunks with preserved sub-heading boundaries.

**Overlap:** 50 tokens (≈ 300–400 characters) between adjacent chunks.

**Reasoning:**
- **Heading-aware first:** Syllabi are highly structured (course-level headings, then per-session headings). Prefer splitting on semantic boundaries (section or session headings) so each chunk contains a whole logical unit (e.g., a single session's outcomes and materials) when possible.
- **Size guardrails:** Some sessions or sections may be long; when a semantic unit exceeds the chunk size, split it into contiguous 400-token pieces so retrieval still returns coherent fragments.
- **Small overlap for context:** A 50-token overlap preserves nearby sentence/paragraph continuity and prevents important list items or transition phrases from being lost across chunks without adding excessive redundancy or cost.
- **Preserve bullets/paragraphs:** When splitting, prefer cut points at paragraph or bullet boundaries (never mid-sentence if possible) to keep learning outcomes and materials readable.
- **Chunk metadata:** Store source filename, top-level heading (e.g., session number/title), chunk index, and character/token offsets as metadata to enable precise attribution and reassembly during generation.
- **Rationale fit:** Typical session content in these syllabi is shorter than 400 tokens, so this size keeps most sessions intact while remaining small enough for efficient embedding and retrieval.

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
all-MiniLM-L6-v2 via sentence-transformers

**Top-k:**
Initial retrieve: top-8 by cosine similarity. Rerank and return top-4 for generation. For short, session-specific queries prefer top-3.

**Input (to retriever):**
- `query_text`: raw user question or instruction.
- `query_type`: one of {`session_query`, `outcome_query`, `materials_query`, `general`}, auto-detected from the query.
- optional `session_id` or `heading_hint`: explicit session number/title when available.

**Output (from retriever):**
- Ordered list of chunks with: `text`, `source_filename`, `top_level_heading` (e.g., session title), `chunk_index`, `score`.
- A short provenance summary merging headings (e.g., "Session 3: Analyzing elementary sorting algorithms — 2 chunks").

**Retrieval rules (heading-aware, syllabus style):**
- **Prefer exact-heading matches:** If `session_id` or heading-like terms are present, boost chunks whose `top_level_heading` matches.
- **Query-type routing:** If `query_type==session_query`, restrict candidates to chunks within the relevant session (or course-level sections if session not found) before global reranking.
- **Structure-sensitive scoring:** When the query requests learning outcomes or materials, boost chunks containing bullet lists or the terms `Learning Outcomes`, `Reading`, `Session`, `Materials` in their text/metadata.
- **Context window and concat rule:** After reranking, concatenate up to top-4 full chunks (preserving chunk order by original offsets) and pass them to the generator, keeping chunk boundaries and source metadata.
- **Fallback:** If heading-aware filter returns fewer than 2 chunks, expand to course-level chunks (same filename) then to global top-k.
- **Deduplication:** Remove nearly-identical chunks (cosine sim > 0.95) to avoid repetition in generated responses.

**Production tradeoff reflection:**
- `all-MiniLM-L6-v2` is fast and cost-effective for small/medium collections and works well for structured English syllabus text. For higher accuracy on paraphrase-heavy queries, consider upgrading to `all-mpnet-base-v2` or a proprietary embedding (better semantic fidelity, higher cost and latency).
- For large-scale deployment, increase retrieval parallelism, shard the vector store by course, and consider hybrid retrieval (BM25 + dense) to improve recall on keyword-heavy queries like exact assignment names.
---

## Evaluation Plan


| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What is the name of session 3 in CS110| Analyzing elementary sorting algorithms|
| 2 | What is the learning outcome of CS114| #distributions, #probability, #cs114-ModelSelection, #cs114-ParameterEstimation, #dataviz, #confidenceintervals, #descriptivestats, #significance, #variables, #cs111-Integration, #cs114-MathTools, #CS-ComputationalTools,#composition, #organization, #professionalism, #ClassParticipation|
| 3 | What is the Course prerequisites of CS166| CS114 ++or++ CS130. Reason: Understanding and creating statistical models. CS110. Reason: Writing object-oriented programs in Python.|
| 4 | What is the reading and learning materials of session 1 in CS130| Howard, J. (2012). *From predictive modelling to optimization: The next frontier *[Video file]. Retrieved March 10, 2016 from:
https://www.youtube.com/watch?v=vYrWTDxoeGg&feature=youtu.be
Ted Talk. (2010), February. Esther Duflo: Social experiments to fight poverty [Video file]. Retrieved March 10, 2016 from:
https://www.ted.com/talks/esther_duflo_social_experiments_to_fight_poverty?language=en
Abbanat, C.. Guidelines for writing decision memos (n.d.) [PDF Document: MIT11_027S11_decision_memo.pdf]. Retrieved
March 10, 2016 from:
https://ocw.mit.edu/courses/11-027-city-to-city-comparing-researching-and-writing-about-cities-new-orleans-spring-
2011/resources/mit11_027s11_decision_memo/
(From the "Prerequisites and Working Knowledge" section, above)
Rubin, D. B. (2003). Basic concepts of statistical inference for causal effects in experiments and observational
studies.Cambridge, MA: Harvard University, Department of Statistics. Read pages 1-10 (up through I-3.2). (permission of the
author)
https://drive.google.com/file/d/1nNDMdCfR6BvNepx2bOb7VplDauKdNdwK/view?usp=sharing
(From the "Prerequisites and Working Knowledge" section, above)
James, G., Witten, D., Hastie, T., & Tibshirani, R. (June 2023). An introduction to statistical learning: With applications in R.
New York: Springer. (Creative Commons). Pages 1-14 (Introduction).
https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf.download.html|
| 5 | How many assignments in CS156 and what are they| There are 3 assignments. They are: Pipeline - First Draft 4x Thu, Week 6 Mon of Week 1 Sun of Week 7
Pipeline - Second Draft 6x Mon, Week 12 Mon of Week 7 Thu of Week 12
Pipeline - Final Draft 10x Thu, Week 15 Mon of Week 12 Fri of Week 15|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->
1. Broken/partial lists: Chunking or retrieval may split bullet lists (learning outcomes, readings) across chunks, causing the LLM to return incomplete or reordered lists (e.g., Q2 and Q4’s long tag/list answers). This risks missing items or merging fragments from different sessions.

2. Noisy / nonstandard formatting and cross-course references: Weird strings (e.g., “CS114 ++or++ CS130”, compact schedule lines, or mixed citation formats) and similar headings across syllabi can cause parsing errors or hallucinated reasoning about prerequisites/dates, and may make it hard to attribute facts to the correct course/session.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
```mermaid
flowchart LR
     A[Document Ingestion]\n(File reader, OCR where needed)
     B[Chunking]\n(heading-aware, 400t chunks, 50t overlap)
     C[Embedding + Vector Store]\n(all-MiniLM-L6-v2 → sentence-transformers → Chroma/FAISS)
     D[Retrieval]\n(dense retrieve top-8 → rerank → top-4; heading boost)
     E[Generation]\n(LLM + prompt templates; attach provenance)

     A --> B --> C --> D --> E
```

- **Document Ingestion:** read files (text, PDF, HTML), normalize text, extract headings and metadata. Tools: Python `read`, `pdfminer`/`pypdf`, simple OCR fallback.
- **Chunking:** apply heading-aware chunk_text() producing 400-token chunks with 50-token overlap; emit metadata (filename, heading, offsets).
- **Embedding + Vector Store:** embed chunks with `all-MiniLM-L6-v2` (sentence-transformers) and index into a vector DB (Chroma or FAISS) with chunk metadata.
- **Retrieval:** dense retrieval (top-8) then heading- and structure-aware reranking; deduplicate and concatenate up to top-4 chunks for generation.
- **Generation:** LLM consumes concatenated context + a provenance-informed prompt template to produce answers with source attributions.
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

**Milestone 3 — Ingestion and chunking:**

Input to AI: the repository `documents/` folder file list and a single target filename (or batch), plus the Chunking Strategy section (400t, 50t overlap). Provide the raw text and extracted heading boundaries.

Expected output:
- A working `chunk_text(text, chunk_size=400, overlap=50)` implementation in Python that:
     - Splits on heading/paragraph boundaries when possible;
     - Produces chunks with fields: `id`, `text`, `filename`, `top_level_heading`, `chunk_index`, `char_start`, `char_end`, `token_count`.
     - Emits a small test harness that runs on `documents/cs110-syllabus.txt` and prints chunk counts and a sample chunk.

Verification:
- Run the test harness: confirm no chunk exceeds 420 tokens, overlap≈50 tokens, and most session headings remain wholly inside single chunks.
- Unit tests: assert round-trip reconstruction (concatenate chunks minus overlaps) contains original text.

Production-ready prompt (for developer AI to generate the code):
"Implement a Python function `chunk_text(text, chunk_size=400, overlap=50)` that is heading-aware and preserves bullet/paragraph boundaries. Input: raw syllabus text and a list of heading regexes (e.g., '^Session \d+'). Output: JSON array of chunk objects with metadata fields `id`, `text`, `filename`, `top_level_heading`, `chunk_index`, `char_start`, `char_end`, `token_count`. Include a CLI test that reads `documents/cs110-syllabus.txt`, runs `chunk_text`, and asserts: (1) every chunk.token_count <= 420, (2) adjacent chunks have >=40 tokens overlap and <=60 tokens overlap, (3) no chunk breaks a sentence. Provide clear docstrings and type hints, and a short README snippet showing how to run the test. Use `tiktoken` (or fallback to simple whitespace tokenization) for token counts."

**Milestone 4 — Embedding and retrieval:**

Input to AI: the chunk JSON output from Milestone 3 and the configuration: `embedding_model='all-MiniLM-L6-v2'`, vector DB choice (`Chroma` default), and retrieval rules (top-8 → rerank → top-4, heading boost).

Expected output:
- A Python module `embed_index.py` exposing:
     - `embed_chunks(chunks) -> list[dict]` (adds `embedding` to each chunk),
     - `index_chunks(chunks, persist_dir)` (stores embeddings + metadata in Chroma/FAISS),
     - `retrieve(query_text, query_type=None, session_id=None, top_k=8) -> list[chunk_objects]` implementing heading-boost reranking and deduplication.
- Integration test that indexes `documents/` chunks and returns sensible results for queries: "Session 3 CS110" and "learning outcomes CS114".

Verification:
- Embeddings shape sanity checks, persisted index files exist, retrieval returns expected `top_level_heading` matches when `session_id` provided, and deduplication removes near-duplicates.

Production-ready prompt (for developer AI to generate the code):
"Create `embed_index.py` that embeds chunk JSON using `sentence-transformers` model `all-MiniLM-L6-v2`, persists to Chroma (or FAISS if Chroma not available), and exposes `retrieve(query_text, query_type=None, session_id=None, top_k=8)` which: (1) embeds the query, (2) retrieves top_k by cosine similarity, (3) if `session_id` is present, boosts chunks whose `top_level_heading` matches exactly, (4) reranks by combined score and removes duplicates with cosine>0.95, (5) returns chunk objects with `score` and metadata. Include an integration test that asserts `retrieve('Session 3', session_id='Session 3')` returns at least 2 chunks from `cs110-syllabus.txt`. Provide installation notes for `sentence-transformers` and `chromadb`."

**Milestone 5 — Generation and interface:**

Input to AI: retrieved chunks (up to top-4 concatenated, with metadata), original user query, and the response spec (short answer, bulleted lists when asking for lists, explicit source attributions per bullet).

Expected output:
- A prompt template and a wrapper `generate_answer(query, retrieved_chunks, config)` that:
     - Constructs a system instruction emphasizing accuracy and source attribution;
     - Concatenates chunk contexts with headings and `---` separators;
     - Instructs the LLM to answer concisely, include a `Sources:` section listing `[filename#chunk_index]` references for each fact or bullet, and to respond `I don't know` when information is not present.
- A small API endpoint or CLI `answer.py` that accepts a query and prints the model response and provenance.

Verification:
- Smoke tests: for each Evaluation Plan question, call `retrieve` + `generate_answer` and assert the model cites at least one matching `source_filename` and the main fact matches the expected answer in the Evaluation Plan.
- Negative test: ask an out-of-corpus question and confirm the response is a safe `I don't know` or fallback that doesn't hallucinate.

Production-ready prompt template (system + user) for the LLM:
"System: You are a helpful, conservative assistant that answers questions using only the provided context. Always cite exact sources using the format `[filename#chunk_index]` after any factual claim. If the context does not contain the answer, reply 'I don't know' and optionally suggest a safe next step (e.g., 'check the course syllabus').\n\nUser: Answer the query below using only the context blocks. Context blocks are delimited by `<<CONTEXT n: filename | heading | chunk_index>>` and `<<ENDCONTEXT>>`. When asked for lists, return a bulleted list and cite the relevant chunk(s) after each item. Finally, include a `Sources:` section listing unique chunk references used.\n\nQuery: {user_query}\n\nContext:\n{concatenated_chunks}" 

Notes: include temperature=0 (deterministic), max_tokens tuned so prompt+context fit model limits, and post-processing to parse `[filename#chunk_index]` links into human-readable source attributions.

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**

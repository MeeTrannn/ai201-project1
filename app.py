"""Gradio querying interface for the syllabus RAG pipeline.

Run:
    python app.py
Then open the printed local URL. Make sure you have ingested the documents first:
    python ingest.py --reset
"""
from typing import List, Dict, Tuple

import gradio as gr

from config import N_RESULTS
from embed_index import retrieve, detect_course_filename, detect_session_id
from generator import generate_answer

EXAMPLES = [
    "What is the name of session 3 in CS110?",
    "What is the learning outcome of CS114?",
    "What are the course prerequisites of CS166?",
    "What is the reading and learning materials of session 1 in CS130?",
    "How many assignments in CS156 and what are they?",
]


def _format_sources(chunks: List[Dict]) -> str:
    """Render the retrieved chunks as a readable provenance panel."""
    if not chunks:
        return "_No chunks retrieved._"
    lines = ["### Retrieved context\n"]
    for i, c in enumerate(chunks, start=1):
        fname = c.get("filename") or "unknown"
        heading = c.get("top_level_heading") or "—"
        idx = c.get("chunk_index")
        score = c.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        snippet = (c.get("text") or "").strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "…"
        lines.append(
            f"**{i}. `{fname}#{idx}`** · _{heading}_ · score `{score_str}`\n\n> {snippet}\n"
        )
    return "\n".join(lines)


def answer_query(query: str, top_k: int, max_context: int) -> Tuple[str, str, str]:
    query = (query or "").strip()
    if not query:
        return ("Please enter a question.", "", "")

    chunks = retrieve(query, top_k=int(top_k))
    answer = generate_answer(query, chunks, max_context_chunks=int(max_context))

    # Show what routing the retriever inferred, for transparency.
    course = detect_course_filename(query) or "—"
    session = detect_session_id(query) or "—"
    routing = f"**Detected course:** `{course}` · **Detected session:** `{session}` · **Chunks retrieved:** {len(chunks)}"

    return answer, routing, _format_sources(chunks[: int(max_context)])


with gr.Blocks(title="Syllabus Q&A — The Unofficial Guide") as demo:
    gr.Markdown(
        "# 📚 Syllabus Q&A\n"
        "Ask questions about the CS course syllabi. Answers are grounded only in the "
        "indexed documents and cite their sources as `[filename#chunk_index]`."
    )

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the course prerequisites of CS166?",
                lines=2,
            )
            with gr.Row():
                top_k = gr.Slider(2, 16, value=N_RESULTS, step=1, label="Retrieve top-k")
                max_context = gr.Slider(1, 8, value=4, step=1, label="Context chunks for generation")
            with gr.Row():
                submit = gr.Button("Ask", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(examples=EXAMPLES, inputs=query_box)
        with gr.Column(scale=4):
            answer_out = gr.Markdown(label="Answer")
            routing_out = gr.Markdown()
            sources_out = gr.Markdown()

    submit.click(answer_query, inputs=[query_box, top_k, max_context], outputs=[answer_out, routing_out, sources_out])
    query_box.submit(answer_query, inputs=[query_box, top_k, max_context], outputs=[answer_out, routing_out, sources_out])
    clear.click(lambda: ("", "", "", ""), outputs=[query_box, answer_out, routing_out, sources_out])


if __name__ == "__main__":
    demo.launch()

import os
import re
from typing import List, Dict, Optional

FALLBACK = "I don't know"


def _format_context_blocks(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        filename = c.get('filename') or 'unknown'
        heading = c.get('top_level_heading') or ''
        chunk_index = c.get('chunk_index') if c.get('chunk_index') is not None else c.get('id') or i - 1
        header = f"<<CONTEXT {i}: {filename} | {heading} | {chunk_index}>>"
        parts.append(header)
        parts.append(c.get('text', ''))
        parts.append(f"<<ENDCONTEXT>>")
    return "\n\n".join(parts)


SYSTEM_PROMPT = (
    "You are a helpful, conservative assistant that answers questions using only the provided context. "
    "Always cite exact sources using the format [filename#chunk_index] immediately after any factual claim. "
    "If the context does not contain the answer, reply exactly: 'I don't know' (without additional information). "
    "When asked for lists, return a bulleted list and cite the relevant chunk(s) after each item. "
    "At the end include a 'Sources:' section listing unique chunk references used. Be concise and avoid hallucination."
)


def generate_answer(query: str, retrieved_chunks: List[Dict], model: Optional[str] = None, max_context_chunks: int = 4) -> str:
    """Generate an answer grounded in retrieved_chunks.

    Each chunk should include: `text`, `filename`, `top_level_heading`, `chunk_index` (or `id`).
    Returns the assistant text (string). If no chunks or no answer found, returns the exact FALLBACK string.
    """
    # If nothing retrieved, return fallback
    if not retrieved_chunks:
        return FALLBACK

    # Limit number of chunks concatenated
    chunks = retrieved_chunks[:max_context_chunks]
    context = _format_context_blocks(chunks)

    user_template = (
        "Answer the query below using ONLY the context blocks. Do not use any outside knowledge.\n\n"
        "Query: {query}\n\nContext:\n{context}\n\n"
        "If the context doesn't contain the answer, reply exactly: 'I don't know' (no extra text). "
        "Cite sources in-line like [filename#chunk_index] after factual claims and provide a final 'Sources:' list."
    )

    user_prompt = user_template.format(query=query, context=context)

    # Prefer Groq client (project uses Groq) if available, otherwise fall back to OpenAI
    try:
        from groq import Groq
        from config import GROQ_API_KEY, LLM_MODEL
        _client = Groq(api_key=GROQ_API_KEY)
        try:
            resp = _client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=LLM_MODEL,
                temperature=0,
                max_tokens=512,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            text = f"[LLM call failed: {e}]"
    except Exception:
        # Fallback to OpenAI if Groq not available
        openai_key = os.environ.get('OPENAI_API_KEY')
        openai_model = model or os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')
        if openai_key:
            try:
                import openai

                openai.api_key = openai_key
                resp = openai.ChatCompletion.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=512,
                )
                text = resp['choices'][0]['message']['content'].strip()
            except Exception as e:
                text = f"[LLM call failed: {e}]"
        else:
            text = "[No LLM configured: set GROQ_API_KEY or OPENAI_API_KEY to enable generation]"

    # Post-process: ensure exact fallback when model tries to hedge
    if text.strip().lower().startswith("i don't know") or text.strip().lower() == "i don't know":
        return FALLBACK

    # Validate cited sources exist in provided chunks; if model cited unknown sources, remove them from provenance
    cited = set(re.findall(r"\[([^\]]+)\]", text))
    valid_refs = set()
    available_refs = {f"{c.get('filename')}#{c.get('chunk_index')}": True for c in chunks}
    for ref in cited:
        if ref in available_refs:
            valid_refs.add(ref)
    # Append Sources: section if not present
    if 'Sources:' not in text and valid_refs:
        text = text + "\n\nSources: " + ", ".join(sorted(valid_refs))

    return text


if __name__ == '__main__':
    # Simple CLI: retrieve using embed_index if available, then generate
    import sys

    query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not query:
        print('Usage: python generator.py "your question"')
        exit(0)

    # Attempt to use embed_index.retrieve if present to get chunks
    try:
        from embed_index import retrieve

        chunks = retrieve(query, top_k=8)
    except Exception:
        print('Warning: embed_index.retrieve not available or failed; expecting chunks to be provided programmatically.')
        chunks = []

    ans = generate_answer(query, chunks)
    print(ans)

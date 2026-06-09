import re
import json
from typing import List, Dict, Optional, Pattern, Tuple


def _tokenize(text: str) -> List[str]:
    """Tokenize text for token-counting.

    Preference order:
      1. tiktoken (LLM-style tokens, cl100k_base)
      2. HuggingFace `transformers` tokenizer using `EMBEDDING_MODEL` from config (if available)
      3. Fallback to simple whitespace split
    Returns a list of token strings (or ids as strings) so len(...) gives token count.
    """
    # 1) tiktoken
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        return [str(t) for t in toks]
    except Exception:
        pass

    # 2) transformers tokenizer based on embedding model from config
    try:
        from config import EMBEDDING_MODEL
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, use_fast=True)
        toks = tok.encode(text)
        return [str(t) for t in toks]
    except Exception:
        pass

    # 3) whitespace fallback
    return text.split()


def count_tokens(text: str) -> int:
    return len(_tokenize(text))


def _find_headings(text: str, heading_regexes: Optional[List[str]] = None) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, heading_text) for heading matches.

    Default looks for lines starting with 'Session <num>' (case-insensitive).
    """
    if heading_regexes is None:
        heading_regexes = [r'^Session\s+\d+[:\-]?', r'^Session\s+\d+']

    matches = []
    for rx in heading_regexes:
        pattern = re.compile(rx, flags=re.IGNORECASE | re.MULTILINE)
        for m in pattern.finditer(text):
            # capture full line as heading
            line_end = text.find('\n', m.start())
            if line_end == -1:
                line_end = len(text)
            heading_text = text[m.start():line_end].strip()
            matches.append((m.start(), line_end, heading_text))

    # remove duplicates and sort by start
    matches = sorted({(s, e, h) for (s, e, h) in matches}, key=lambda x: x[0])
    return matches


def _split_sentences_with_spans(segment: str) -> List[Tuple[str, int, int]]:
    # split on sentence enders but preserve content. Use regex then find spans.
    sentences = re.split(r'(?<=[.!?])\s+', segment)
    spans = []
    idx = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        pos = segment.find(s, idx)
        if pos == -1:
            # fallback: skip
            continue
        spans.append((s, pos, pos + len(s)))
        idx = pos + len(s)
    return spans


def chunk_text(text: str,
               chunk_size: int = 400,
               overlap: int = 50,
               heading_regexes: Optional[List[str]] = None,
               filename: Optional[str] = None) -> List[Dict]:
    """Produce heading-aware chunks with sentence boundaries and token overlap.

    Returns list of chunk dicts with fields:
      - id, text, filename, top_level_heading, chunk_index, char_start, char_end, token_count
    """
    text = text.replace('\r\n', '\n')
    headings = _find_headings(text, heading_regexes)

    segments = []
    if headings:
        # Preamble: any course-level content before the first heading
        # (Course Description, Objectives, Learning Outcomes, Prerequisites, etc.).
        # Without this, that content is silently dropped from the index.
        first_start = headings[0][0]
        preamble = text[:first_start]
        if preamble.strip():
            segments.append((0, first_start, "Course Overview", preamble))
        for i, (s, e, h) in enumerate(headings):
            seg_start = s
            seg_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            seg_text = text[seg_start:seg_end]
            segments.append((seg_start, seg_end, h, seg_text))
    else:
        segments = [(0, len(text), filename or "document", text)]

    base_chunks = []
    for seg_idx, (seg_start, seg_end, heading_text, seg_text) in enumerate(segments):
        # Paragraph-first splitting with spans relative to the whole text
        para_splits = [p for p in re.split(r'\n\s*\n+', seg_text)]
        # find paragraph spans
        paragraphs = []
        cursor = seg_start
        for p in para_splits:
            p = p.strip()
            if not p:
                cursor += len(p)
                continue
            # find p in seg_text starting from cursor - seg_start
            rel_pos = seg_text.find(p, max(0, cursor - seg_start))
            if rel_pos == -1:
                # fallback: use cursor
                rel_pos = max(0, cursor - seg_start)
            abs_start = seg_start + rel_pos
            abs_end = abs_start + len(p)
            paragraphs.append((p, abs_start, abs_end))
            cursor = abs_end

        # compute token counts per paragraph
        para_token_counts = [count_tokens(p[0]) for p in paragraphs]

        # aggregate paragraphs into chunks
        buf = []
        buf_tokens = 0
        for idx, (para, pstart, pend) in enumerate(paragraphs):
            p_toks = para_token_counts[idx]
            if p_toks >= chunk_size:
                # paragraph too long: split by sentences
                sentences = _split_sentences_with_spans(para)
                if not sentences:
                    continue
                sent_tokens = [count_tokens(s[0]) for s in sentences]
                s_buf = []
                s_buf_tokens = 0
                for si, stoks in enumerate(sent_tokens):
                    if s_buf_tokens + stoks <= chunk_size:
                        s_buf.append(sentences[si])
                        s_buf_tokens += stoks
                        continue
                    # emit s_buf
                    chunk_text_str = ' '.join([s[0] for s in s_buf]).strip()
                    char_start = pstart + s_buf[0][1]
                    char_end = pstart + s_buf[-1][2]
                    base_chunks.append({
                        "text": chunk_text_str,
                        "char_start": char_start,
                        "char_end": char_end,
                        "top_level_heading": heading_text,
                        "paragraph_indices": [idx],
                    })
                    # reset
                    s_buf = [sentences[si]]
                    s_buf_tokens = stoks
                if s_buf:
                    chunk_text_str = ' '.join([s[0] for s in s_buf]).strip()
                    char_start = pstart + s_buf[0][1]
                    char_end = pstart + s_buf[-1][2]
                    base_chunks.append({
                        "text": chunk_text_str,
                        "char_start": char_start,
                        "char_end": char_end,
                        "top_level_heading": heading_text,
                        "paragraph_indices": [idx],
                    })
                # continue buffer untouched
                continue

            # if adding this paragraph fits, append
            if buf_tokens + p_toks <= chunk_size:
                buf.append((para, pstart, pend, idx))
                buf_tokens += p_toks
            else:
                # emit buffer as chunk
                if buf:
                    chunk_text_str = '\n\n'.join([b[0] for b in buf]).strip()
                    char_start = buf[0][1]
                    char_end = buf[-1][2]
                    para_idxs = [b[3] for b in buf]
                    base_chunks.append({
                        "text": chunk_text_str,
                        "char_start": char_start,
                        "char_end": char_end,
                        "top_level_heading": heading_text,
                        "paragraph_indices": para_idxs,
                    })
                # start new buffer with current paragraph
                buf = [(para, pstart, pend, idx)]
                buf_tokens = p_toks

        # flush remaining buffer
        if buf:
            chunk_text_str = '\n\n'.join([b[0] for b in buf]).strip()
            char_start = buf[0][1]
            char_end = buf[-1][2]
            para_idxs = [b[3] for b in buf]
            base_chunks.append({
                "text": chunk_text_str,
                "char_start": char_start,
                "char_end": char_end,
                "top_level_heading": heading_text,
                "paragraph_indices": para_idxs,
            })

    # Now apply sentence-level overlap (~`overlap` tokens) between adjacent base_chunks
    chunks_with_overlap: List[Dict] = []
    for i, bc in enumerate(base_chunks):
        text_i = bc["text"].strip()
        char_start_i = bc["char_start"]
        char_end_i = bc["char_end"]
        token_count_i = count_tokens(text_i)
        chunk_obj = {
            "text": text_i,
            "char_start": char_start_i,
            "char_end": char_end_i,
            "top_level_heading": bc.get("top_level_heading"),
            "paragraph_indices": bc.get("paragraph_indices", []),
            "token_count": token_count_i,
        }
        # if not first, prepend overlap from previous chunk's tail sentences
        if i > 0:
            prev = chunks_with_overlap[-1]
            prev_text = prev["text"]
            # split prev into sentences and pick tail until tokens >= overlap
            prev_sents = _split_sentences_with_spans(prev_text)
            tail = []
            tail_tokens = 0
            j = len(prev_sents) - 1
            while j >= 0 and tail_tokens < overlap:
                tail.insert(0, prev_sents[j][0])
                tail_tokens += count_tokens(prev_sents[j][0])
                j -= 1
            if tail:
                overlap_text = ' '.join(tail).strip()
                new_text = overlap_text + ' ' + chunk_obj["text"]
                # adjust char_start to include overlap (best-effort: take prev.char_start)
                new_char_start = min(prev["char_start"], chunk_obj["char_start"])
                new_token_count = count_tokens(new_text)
                chunk_obj.update({"text": new_text, "char_start": new_char_start, "token_count": new_token_count})

        chunks_with_overlap.append(chunk_obj)

    # Finalize chunk dicts with ids and filename
    final_chunks: List[Dict] = []
    for idx, c in enumerate(chunks_with_overlap):
        final_chunks.append({
            "id": f"{(filename or 'doc').replace(' ', '_').lower()}_{idx}",
            "text": c["text"],
            "filename": filename or 'document',
            "top_level_heading": c.get("top_level_heading"),
            "chunk_index": idx,
            "char_start": c.get("char_start"),
            "char_end": c.get("char_end"),
            "token_count": c.get("token_count"),
            "paragraph_indices": c.get("paragraph_indices", []),
        })

    return final_chunks


if __name__ == '__main__':
    import os
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent
    docs_dir = repo_root / 'documents'
    sample_file = docs_dir / 'cs110-syllabus.txt'
    if not sample_file.exists():
        print(f"Sample file not found: {sample_file}")
        exit(1)

    text = sample_file.read_text(encoding='utf-8')
    chunks = chunk_text(text, chunk_size=400, overlap=50, filename=sample_file.name)
    print(f"Produced {len(chunks)} chunks from {sample_file.name}")
    print("Sample chunks:")
    for c in chunks[:3]:
        print(json.dumps({k: c[k] for k in ('id', 'top_level_heading', 'chunk_index', 'token_count')}, indent=2))
        print(c['text'][:400])
        print('-' * 80)

    # Verification checks
    # 1) token limits
    for c in chunks:
        if c['token_count'] > 420:
            raise AssertionError(f"Chunk {c['id']} exceeds 420 tokens: {c['token_count']}")

    # 2) overlaps approx
    def tok_count(s):
        return count_tokens(s)

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]['text']
        cur = chunks[i]['text']
        # compute overlap in tokens by finding longest suffix of prev that is prefix of cur
        max_overlap = 0
        prev_words = _tokenize(prev)
        cur_words = _tokenize(cur)
        # naive token overlap by text: measure tokens in common prefix of cur and suffix of prev
        for o in range(1, min(100, len(prev_words))):
            if prev_words[-o:] == cur_words[:o]:
                max_overlap = o
        if max_overlap > 0:
            if not (40 <= max_overlap <= 60):
                print(f"Warning: chunk overlap tokens {max_overlap} not in [40,60] between {chunks[i-1]['id']} and {chunks[i]['id']}")

    print("Basic verification complete.")

import os
from typing import List, Dict, Optional

try:
    import chromadb
except Exception:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from pathlib import Path
import numpy as np

try:
    from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL
    DEFAULT_PERSIST_DIR = CHROMA_PATH
    DEFAULT_COLLECTION = CHROMA_COLLECTION
    DEFAULT_MODEL = EMBEDDING_MODEL
except Exception:
    DEFAULT_PERSIST_DIR = "./chroma_db"
    DEFAULT_COLLECTION = "syllabus"
    DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _get_model(model_name: str = DEFAULT_MODEL):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    return SentenceTransformer(model_name)


def _get_client(persist_dir: str = DEFAULT_PERSIST_DIR):
    if chromadb is None:
        raise RuntimeError("chromadb not installed")
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def embed_chunks(chunks: List[Dict], model_name: str = DEFAULT_MODEL) -> List[Dict]:
    """Add `embedding` key to each chunk using sentence-transformers."""
    model = _get_model(model_name)
    texts = [c["text"] for c in chunks]
    embs = model.encode(texts, show_progress_bar=False)
    # ensure numpy array shape
    embs = np.asarray(embs)
    out = []
    for c, v in zip(chunks, embs):
        cc = dict(c)
        cc["embedding"] = v.astype(float).tolist()
        out.append(cc)
    return out


def index_chunks(chunks: List[Dict], persist_dir: str = DEFAULT_PERSIST_DIR, collection_name: str = DEFAULT_COLLECTION, model_name: str = DEFAULT_MODEL):
    """Persist chunks (with embeddings) into a Chroma collection.

    Each chunk's metadata will include: filename, top_level_heading, chunk_index, char_start, char_end
    """
    client = _get_client(persist_dir)
    # create or get collection without embedding function; we will pass embeddings
    collection = client.get_or_create_collection(name=collection_name)

    # embed if not already embedded
    to_index = []
    for c in chunks:
        if "embedding" not in c:
            to_index.append(c)

    if to_index:
        embedded = embed_chunks(to_index, model_name=model_name)
        # merge embeddings back
        for e in embedded:
            for orig in chunks:
                if orig["id"] == e["id"]:
                    orig["embedding"] = e["embedding"]
                    break

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "filename": c.get("filename"),
            "top_level_heading": c.get("top_level_heading"),
            "chunk_index": c.get("chunk_index"),
            "char_start": c.get("char_start"),
            "char_end": c.get("char_end"),
        }
        for c in chunks
    ]
    embeddings = [c["embedding"] for c in chunks]

    # upsert so re-running ingestion is idempotent (collection.add errors on duplicate ids)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    print(f"Indexed {len(chunks)} chunks into collection '{collection_name}' at {persist_dir}")


def _cosine(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect_course_filename(query_text: str) -> Optional[str]:
    """Extract a course code (e.g. CS110) from the query and map it to a filename.

    Syllabi are structurally near-identical, so pure semantic similarity confuses
    courses. Detecting the course code lets us hard-filter to the right document.
    """
    import re
    m = re.search(r'cs\s*-?\s*(\d{3,4})', query_text, flags=re.IGNORECASE)
    if not m:
        return None
    return f"cs{m.group(1)}-syllabus.txt"


def detect_session_id(query_text: str) -> Optional[str]:
    """Extract a 'Session N' reference from the query, if present."""
    import re
    m = re.search(r'session\s*(\d+)', query_text, flags=re.IGNORECASE)
    if not m:
        return None
    return f"Session {m.group(1)}"


def retrieve(query_text: str,
             persist_dir: str = DEFAULT_PERSIST_DIR,
             collection_name: str = DEFAULT_COLLECTION,
             model_name: str = DEFAULT_MODEL,
             top_k: int = 8,
             session_id: Optional[str] = None,
             query_type: Optional[str] = None) -> List[Dict]:
    """Retrieve heading-aware results.

    Steps:
    - auto-detect course code (filename filter) and session id from the query
    - embed query
    - call chroma.query, restricted to the detected course when found
    - apply heading boost if a session is referenced
    - deduplicate near-duplicates (cosine > 0.95)
    - return reranked list of up to top_k items with metadata and combined score
    """
    client = _get_client(persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    model = _get_model(model_name)
    q_emb = model.encode([query_text], show_progress_bar=False)[0].astype(float).tolist()

    # Auto-detect course/session from the query if not explicitly provided.
    course_file = detect_course_filename(query_text)
    if session_id is None:
        session_id = detect_session_id(query_text)

    # Fetch a larger candidate pool than top_k so heading/structure boosts can pull
    # the right chunk (e.g. a specific session, or the Course Overview) into the
    # final top results even when it ranks low on raw cosine similarity.
    fetch_n = min(max(top_k * 4, 24), 60)

    def _do_query(where):
        # NB: "ids" are always returned by Chroma and are NOT a valid `include` value.
        return collection.query(
            query_embeddings=[q_emb],
            n_results=fetch_n,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    where = {"filename": course_file} if course_file else None
    results = _do_query(where)
    # Fallback (planning.md): if the course filter is too restrictive, widen to global.
    if where and len(results.get("documents", [[]])[0]) < 2:
        results = _do_query(None)
    docs = results.get("documents", [])[0]
    metas = results.get("metadatas", [])[0]
    dists = results.get("distances", [])[0]
    embds = results.get("embeddings", [])[0] if results.get("embeddings") is not None else [None] * len(docs)
    ids = results.get("ids", [])[0] if results.get("ids") is not None else [None] * len(docs)

    items = []
    for doc, meta, dist, emb, _id in zip(docs, metas, dists, embds, ids):
        # convert distance to similarity-like score
        score = 1.0 - float(dist) if dist is not None else 0.0
        items.append({
            "id": _id,
            "text": doc,
            "filename": meta.get("filename"),
            "top_level_heading": meta.get("top_level_heading"),
            "chunk_index": meta.get("chunk_index"),
            "distance": dist,
            "score": score,
            "embedding": emb,
        })

    # Heading / structure boosts. These are ADDITIVE — Chroma similarity scores can be
    # negative, so a multiplicative boost would push matches the wrong way.
    import re as _re

    def _heading_matches_session(heading: str, sid: str) -> bool:
        # Precise match so "Session 1" does not match "Session 13".
        m = _re.match(r'session\s*(\d+)', sid, flags=_re.IGNORECASE)
        if not m:
            return sid.lower() in heading.lower()
        n = m.group(1)
        return bool(_re.match(rf'session\s*{n}\b', heading, flags=_re.IGNORECASE))

    # Course-level facts (learning outcomes, objectives, prerequisites) live in the
    # "Course Overview" preamble rather than any session.
    course_level_intent = bool(_re.search(
        r'learning outcome|objective|prerequisite|course description|pre-?req',
        query_text, flags=_re.IGNORECASE))

    for it in items:
        heading = str(it.get("top_level_heading") or "")
        if session_id and _heading_matches_session(heading, session_id):
            it["score"] += 1.0
        if course_level_intent and heading.lower() == "course overview":
            it["score"] += 0.5

    # rerank by score desc
    items = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)

    # deduplicate by embedding cosine similarity
    final = []
    seen_embs = []
    for it in items:
        emb = it.get("embedding")
        if emb is None:
            final.append(it)
            continue
        dup = False
        for s in seen_embs:
            if _cosine(emb, s) > 0.95:
                dup = True
                break
        if not dup:
            final.append(it)
            seen_embs.append(emb)
        if len(final) >= top_k:
            break

    return final


if __name__ == '__main__':
    # Simple integration test: chunk cs110-syllabus, index, retrieve
    from chunker import chunk_text
    repo_root = Path(__file__).resolve().parent
    sample = repo_root / 'documents' / 'cs110-syllabus.txt'
    if not sample.exists():
        print("Sample syllabus not found for integration test; please place cs110-syllabus.txt in documents/")
        exit(0)

    text = sample.read_text(encoding='utf-8')
    chunks = chunk_text(text, chunk_size=400, overlap=50, filename=sample.name)
    index_chunks(chunks, persist_dir=DEFAULT_PERSIST_DIR, collection_name=DEFAULT_COLLECTION, model_name=DEFAULT_MODEL)
    res = retrieve('Session 3', persist_dir=DEFAULT_PERSIST_DIR, collection_name=DEFAULT_COLLECTION, model_name=DEFAULT_MODEL, top_k=8, session_id='Session 3')
    print(f"Retrieved {len(res)} items; top filenames/headings:")
    for r in res[:5]:
        print(r['filename'], r['top_level_heading'], r['chunk_index'], r['score'])

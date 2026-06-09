"""Batch ingestion: chunk every syllabus in DOCS_PATH and index into Chroma.

Run once before querying:
    python ingest.py [--reset]
"""
import argparse
from pathlib import Path

from config import DOCS_PATH, CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL
from chunker import chunk_text
from embed_index import index_chunks, _get_client


def main(reset: bool):
    docs_dir = Path(DOCS_PATH)
    files = sorted(docs_dir.glob("*.txt"))
    if not files:
        print(f"No .txt documents found in {docs_dir}")
        return

    if reset:
        client = _get_client(CHROMA_PATH)
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"Reset collection '{CHROMA_COLLECTION}'")
        except Exception:
            pass

    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=400, overlap=50, filename=f.name)
        print(f"{f.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    index_chunks(
        all_chunks,
        persist_dir=CHROMA_PATH,
        collection_name=CHROMA_COLLECTION,
        model_name=EMBEDDING_MODEL,
    )
    print(f"Done. Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk and index all syllabi into Chroma")
    ap.add_argument("--reset", action="store_true", help="Delete and rebuild the collection first")
    args = ap.parse_args()
    main(args.reset)

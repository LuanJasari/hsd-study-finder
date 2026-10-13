"""Index bauen: ``python -m embeddings.build_index``

Liest alle Modulhandbücher aus ``data/``, chunkt und embeddet sie und schreibt sie nach ChromaDB.
"""

from __future__ import annotations

import time
from collections import Counter

from config import settings
from embeddings.embedder import embed_texts
from embeddings.loader import load_all_chunks
from embeddings.vector_store import load_snapshot, query_chunks, rebuild_index


def main() -> None:
    start = time.time()
    chunks = load_all_chunks()
    if not chunks:
        raise SystemExit(f"Keine Modulhandbücher in {settings.data_dir} gefunden (PDF oder Markdown erwartet).")

    per_program = Counter(c.program_name for c in chunks)
    modules = Counter(c.program_name for c in {(c.program_id, c.module): c for c in chunks}.values())
    for name, n in per_program.items():
        print(f"  {name}: {n} Chunks, {modules[name]} Module/Abschnitte")

    print(f"Embedding mit {settings.embedding_model} …")
    embeddings = embed_texts([c.text for c in chunks], show_progress=True)

    rebuild_index(chunks, embeddings)
    snapshot = load_snapshot()
    print(f"Index geschrieben nach {settings.chroma_dir}: {len(snapshot.program_ids)} Studiengänge, "
          f"{len(snapshot.chunk_ids)} Chunks.")

    # Kurzer Plausibilitätstest des Retrievals gegen ChromaDB
    probe = embed_texts(["Programmieren, Datenbanken und Webentwicklung"])[0]
    best = query_chunks(probe, k=1)[0]
    print(f"Retrieval-Test: {best['program_name']} / {best['module']} (cos={best['similarity']:.2f})")
    print(f"Fertig in {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()

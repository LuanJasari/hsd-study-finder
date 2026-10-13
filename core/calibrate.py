"""COS_FLOOR / COS_CEIL für das aktuelle Embedding-Modell bestimmen: ``python -m core.calibrate [--write]``

Beides wird mit simulierten Nutzerprofilen gegen den gebauten Index gemessen, mit derselben Formel wie
im Live-Betrieb (Prior + Antworten, gemischter Kosinus aus Zentroid und Top-k-Chunks):

* **ceil** – mittlerer Kosinus zum eigenen Studiengang, wenn eine Person fünfmal „Ja“ zu Abschnitten
  *dieses* Studiengangs sagt. Die Abschnitte stammen aus einer Hälfte der Module, bewertet wird auf der
  anderen Hälfte, damit weder ein Chunk noch seine Nachbarn aus demselben Modul sich selbst finden.
* **floor** – mittlerer Kosinus über alle Studiengänge, wenn die Person ausschließlich zu fachfremden
  Themen („Kochen“, „Fußball“ …) „Ja“ sagt.

Mit ``--write`` werden die Werte in die ``.env`` eingetragen (sie wird bei Bedarf aus ``.env.example`` angelegt).
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from config import ROOT, settings
from core.scoring import cosine_scores
from core.similarity import normalize
from embeddings.embedder import embed_texts
from embeddings.vector_store import IndexSnapshot, load_snapshot

OFF_TOPIC = [
    "Ich koche und backe leidenschaftlich gern und probiere neue Rezepte aus.",
    "Ich spiele Fußball im Verein und gehe regelmäßig zum Training.",
    "Ich pflege gern Menschen und möchte im Krankenhaus oder Altenheim arbeiten.",
    "Ich interessiere mich für Gartenarbeit, Pflanzen und Landwirtschaft.",
    "Ich möchte Rechtswissenschaften studieren und Fälle vor Gericht verhandeln.",
    "Ich wandere gern in den Bergen und reise durch die Natur.",
    "Ich interessiere mich für Mode, Schneidern und Textildesign.",
    "Ich möchte Kindern in der Grundschule das Lesen beibringen.",
    "Ich liebe Tiere und möchte Tiermedizin oder Biologie studieren.",
    "Ich begeistere mich für Geschichte, Archäologie und Museen.",
]
N_PROBES = 5
N_TRIALS = 40
MIN_SPAN = 0.10


def _profile(snapshot: IndexSnapshot, probes: np.ndarray) -> np.ndarray:
    return normalize(settings.prior_weight * snapshot.prior + probes.sum(axis=0))


def measure_ceil(snapshot: IndexSnapshot, rng: np.random.Generator) -> np.ndarray:
    probe_idx, eval_idx = [], []
    for p in range(len(snapshot.program_ids)):
        own = snapshot.chunks_of(p)
        modules = list(dict.fromkeys(snapshot.chunk_meta[i]["module"] for i in own))
        probe_modules = set(rng.permutation(modules)[: len(modules) // 2])
        in_probe = np.array([snapshot.chunk_meta[i]["module"] in probe_modules for i in own])
        probe_idx.append(own[in_probe])
        eval_idx.append(own[~in_probe])
    evaluation = snapshot.subset(np.concatenate(eval_idx))
    values = []
    for p, pool in enumerate(probe_idx):
        for _ in range(N_TRIALS):
            probes = snapshot.chunk_embeddings[rng.choice(pool, size=min(N_PROBES, len(pool)), replace=False)]
            values.append(cosine_scores(_profile(evaluation, probes), evaluation)[p])
    return np.array(values)


def measure_floor(snapshot: IndexSnapshot, rng: np.random.Generator) -> np.ndarray:
    vectors = embed_texts(OFF_TOPIC)
    values = []
    for _ in range(N_TRIALS):
        probes = vectors[rng.choice(len(vectors), size=N_PROBES, replace=False)]
        values.append(cosine_scores(_profile(snapshot, probes), snapshot))
    return np.concatenate(values)


def suggest(snapshot: IndexSnapshot, seed: int = 42) -> tuple[float, float, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    on_topic = measure_ceil(snapshot, rng)
    off_topic = measure_floor(snapshot, rng)
    floor, ceil = float(off_topic.mean()), float(on_topic.mean())
    if ceil - floor < MIN_SPAN:  # degenerierter Index: Skala nicht kollabieren lassen
        ceil = floor + MIN_SPAN
    return round(floor, 3), round(ceil, 3), off_topic, on_topic


def write_env(floor: float, ceil: float) -> None:
    env = ROOT / ".env"
    source = env if env.exists() else ROOT / ".env.example"
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    for key, value in (("COS_FLOOR", floor), ("COS_CEIL", ceil)):
        line = f"{key}={value}"
        text, n = re.subn(rf"(?m)^{key}=.*$", line, text)
        if n == 0:
            text += ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    env.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Werte in die .env eintragen")
    args = parser.parse_args()

    snapshot = load_snapshot()
    if snapshot.is_empty:
        raise SystemExit("Der Vektorindex ist leer. Bitte zuerst `python -m embeddings.build_index` ausführen.")

    floor, ceil, off_topic, on_topic = suggest(snapshot)
    print(f"Modell: {settings.embedding_model}")
    print(f"fachfremde Profile : Ø {off_topic.mean():.3f}  (min {off_topic.min():.3f}, max {off_topic.max():.3f})")
    print(f"passende Profile   : Ø {on_topic.mean():.3f}  (min {on_topic.min():.3f}, max {on_topic.max():.3f})")
    print(f"\nVorschlag: COS_FLOOR={floor}  COS_CEIL={ceil}")
    if args.write:
        write_env(floor, ceil)
        print(f"In {ROOT / '.env'} eingetragen.")
    else:
        print("Mit --write werden die Werte in die .env geschrieben.")


if __name__ == "__main__":
    main()

"""COS_FLOOR / COS_CEIL für das aktuelle Embedding-Modell bestimmen:
``python -m core.calibrate [--probe-style chunk|feature|both] [--write]``

Beides wird mit simulierten Nutzerprofilen gegen den gebauten Index gemessen, mit derselben Formel wie
im Live-Betrieb (Prior + Antworten, gemischter Kosinus aus Zentroid und Top-k-Chunks):

* **ceil** – mittlerer Kosinus zum eigenen Studiengang, wenn eine Person fünfmal „Ja“ zu Abschnitten
  *dieses* Studiengangs sagt. Die Abschnitte stammen aus einer Hälfte der Module, bewertet wird auf der
  anderen Hälfte, damit weder ein Chunk noch seine Nachbarn aus demselben Modul sich selbst finden.
  Als „Ja“-Probe dienen wahlweise
    - ``chunk``: der ganze Chunk (lang, viel Kontext → hohe Kosinus-Werte),
    - ``feature``: nur die ersten ~14 Wörter des Chunks, also so kurz wie die Merkmalssätze, die das LLM
      im Live-Betrieb liefert (kurze Sätze erreichen kaum so hohe Werte),
    - ``both`` (Standard): beide; als ``ceil`` zählt der *niedrigere* Wert, damit die Prozentskala im
      Live-Betrieb nicht an einer unerreichbar hohen Obergrenze hängt.
* **floor** – mittlerer Kosinus über alle Studiengänge, wenn die Person ausschließlich zu fachfremden
  Themen („Kochen“, „Fußball“ …) „Ja“ sagt.

Mit ``--write`` werden die Werte in die ``.env`` eingetragen (sie wird bei Bedarf aus ``.env.example`` angelegt).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Callable

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
PROBE_STYLES = ("chunk", "feature", "both")
N_PROBES = 5
N_TRIALS = 40
SHORT_WORDS = 14
MIN_SPAN = 0.10

Embed = Callable[[list[str]], np.ndarray]


@dataclass
class Calibration:
    floor: float
    ceil: float
    off_topic: np.ndarray
    on_topic: dict[str, np.ndarray]   # Probe-Stil → gemessene Kosinus-Werte


def short_probe(chunk_text: str, words: int = SHORT_WORDS) -> str:
    """Kurzfassung eines Chunks: Wörter nach dem Modultitel, ohne Aufzählungs- und Satzzeichen am Anfang."""
    body = chunk_text.split(" – ", 1)[-1]
    body = re.sub(r"^[\s.•\-·;:,]+", "", body)
    return " ".join(body.split()[:words])


def _profile(snapshot: IndexSnapshot, probes: np.ndarray) -> np.ndarray:
    return normalize(settings.prior_weight * snapshot.prior + probes.sum(axis=0))


def measure_ceil(
    snapshot: IndexSnapshot,
    rng: np.random.Generator,
    style: str = "chunk",
    embed: Embed = embed_texts,
) -> np.ndarray:
    if style not in ("chunk", "feature"):
        raise ValueError(f"unbekannter Probe-Stil: {style}")
    probe_idx, eval_idx = [], []
    for p in range(len(snapshot.program_ids)):
        own = snapshot.chunks_of(p)
        modules = list(dict.fromkeys(snapshot.chunk_meta[i]["module"] for i in own))
        probe_modules = set(rng.permutation(modules)[: len(modules) // 2])
        in_probe = np.array([snapshot.chunk_meta[i]["module"] in probe_modules for i in own])
        probe_idx.append(own[in_probe])
        eval_idx.append(own[~in_probe])
    evaluation = snapshot.subset(np.concatenate(eval_idx))

    vectors = snapshot.chunk_embeddings
    if style == "feature":  # kurze Sätze neu einbetten (einmal je Chunk, nicht je Versuch)
        all_probe = np.concatenate(probe_idx)
        vectors = np.zeros_like(snapshot.chunk_embeddings)
        vectors[all_probe] = embed([short_probe(snapshot.chunk_texts[i]) for i in all_probe])

    values = []
    for p, pool in enumerate(probe_idx):
        for _ in range(N_TRIALS):
            probes = vectors[rng.choice(pool, size=min(N_PROBES, len(pool)), replace=False)]
            values.append(cosine_scores(_profile(evaluation, probes), evaluation)[p])
    return np.array(values)


def measure_floor(snapshot: IndexSnapshot, rng: np.random.Generator, embed: Embed = embed_texts) -> np.ndarray:
    vectors = embed(OFF_TOPIC)
    values = []
    for _ in range(N_TRIALS):
        probes = vectors[rng.choice(len(vectors), size=N_PROBES, replace=False)]
        values.append(cosine_scores(_profile(snapshot, probes), snapshot))
    return np.concatenate(values)


def calibrate(
    snapshot: IndexSnapshot,
    style: str = "both",
    seed: int = 42,
    embed: Embed = embed_texts,
) -> Calibration:
    if style not in PROBE_STYLES:
        raise ValueError(f"unbekannter Probe-Stil: {style}")
    rng = np.random.default_rng(seed)
    styles = ("chunk", "feature") if style == "both" else (style,)
    on_topic = {s: measure_ceil(snapshot, rng, s, embed) for s in styles}
    off_topic = measure_floor(snapshot, rng, embed)
    floor = float(off_topic.mean())
    ceil = min(float(values.mean()) for values in on_topic.values())
    if ceil - floor < MIN_SPAN:  # degenerierter Index: Skala nicht kollabieren lassen
        ceil = floor + MIN_SPAN
    return Calibration(round(floor, 3), round(ceil, 3), off_topic, on_topic)


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


def _stats(values: np.ndarray) -> str:
    return f"Ø {values.mean():.3f}  (min {values.min():.3f}, max {values.max():.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-style", choices=PROBE_STYLES, default="both",
                        help="Länge der simulierten „Ja“-Antworten (Standard: both → niedrigerer ceil)")
    parser.add_argument("--write", action="store_true", help="Werte in die .env eintragen")
    args = parser.parse_args()

    snapshot = load_snapshot()
    if snapshot.is_empty:
        raise SystemExit("Der Vektorindex ist leer. Bitte zuerst `python -m embeddings.build_index` ausführen.")

    result = calibrate(snapshot, args.probe_style)
    print(f"Modell: {settings.embedding_model}")
    print(f"fachfremde Profile{'':<18}: {_stats(result.off_topic)}")
    for style, values in result.on_topic.items():
        label = "ganze Chunks" if style == "chunk" else f"Kurzsätze (~{SHORT_WORDS} Wörter)"
        print(f"passende Profile, {label:<26}: {_stats(values)}")
    print(f"\nVorschlag: COS_FLOOR={result.floor}  COS_CEIL={result.ceil}")
    if args.write:
        write_env(result.floor, result.ceil)
        print(f"In {ROOT / '.env'} eingetragen.")
    else:
        print("Mit --write werden die Werte in die .env geschrieben.")


if __name__ == "__main__":
    main()

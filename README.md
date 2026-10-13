# 🎓 HSD Study-Finder

**Ein LLM-gesteuerter, probabilistischer Studiengangs-Akinator für die Hochschule Düsseldorf.**

Portfolioprojekt im Modul *Advances in Intelligent Systems* · Michael Adzase & Luan Jasari ·
Live-Demo auf der KI-Con (11.01.2027).

> ⚠️ Die Modulhandbücher in `data/programs/` sind **Beispieldaten** und keine offiziellen
> HSD-Dokumente. Wie die echten Handbücher eingebunden werden, steht in [`data/README.md`](data/README.md).

---

## Idee

Statt starrer Filter stellt der Study-Finder – wie das Spiel *Akinator* – Frage für Frage und
grenzt so die passenden Studiengänge ein. Dabei gilt:

1. **Kein Softmax.** Jeder Studiengang bekommt einen **unabhängigen** Kompatibilitätsscore
   aus der Kosinus-Ähnlichkeit zwischen Nutzerprofil und Studiengang im Embedding-Raum
   (vgl. Barz & Denzler, 2020). Mehrere Studiengänge können gleichzeitig 80 % erreichen –
   oder alle unter 20 % liegen, wenn das Angebot schlicht nicht passt.
2. **RAG.** Die Modulhandbücher werden gechunkt, eingebettet und in ChromaDB gespeichert.
   Fragen und Begründungen stützen sich auf echte Modulinhalte.
3. **Dynamischer Akinator-Loop.** Nach jeder Antwort wird das Profil neu gescort; das LLM
   bekommt die aktuellen Scores plus die trennschärfsten Modulabschnitte und formuliert
   zur Laufzeit die nächste maßgeschneiderte Entscheidungsfrage.

## Architektur

```
                    ┌──────────────────────── Offline ────────────────────────┐
 data/*.md|*.pdf ──▶│ loader.py ─▶ Chunking ─▶ embedder.py ─▶ ChromaDB          │
                    │   (Front-Matter,         (Sentence-       ├ hsd_chunks     │
                    │    Modul-Überschriften)   Transformers)   └ hsd_programs   │
                    └─────────────────────────────────────────────────────────┘
                                                                    │
                    ┌──────────────────────── Live-Loop ────────────▼─────────┐
   Antwort ────────▶│ scoring.py: u ← normalize(Σ wᵢ·e(probeᵢ))                │
   (Ja … Nein)      │            score_p = cos(u, c_p)  → linear → 0–100 %     │
                    │ question_generator.py:                                   │
                    │   1. trennscharfe Chunks (Std. der cos zu Kandidaten)    │
                    │   2. RAG-Retrieval je Kandidat (ChromaDB)                │
                    │   3. LLM formuliert Frage + Merkmalssatz (JSON)          │
                    │   4. Merkmal → Embedding = probe für die nächste Antwort │
                    └──────────────────────────────┬──────────────────────────┘
                                                   ▼
                                     app.py (Streamlit, Balkendiagramm)
```

### Mathematischer Kern

**Nutzerprofil** – ein Suchvektor, der durch den Raum wandert (Rocchio-artiges Relevance Feedback):

$$u = \mathrm{normalize}\Big(w_p\,\bar c + w_0\,e(\text{intro}) + \sum_i a_i\, e(\text{probe}_i)\Big)$$

mit $\bar c$ = neutraler Startpunkt (Mittel aller Studiengänge), $a_i \in \{1, 0.5, 0, -0.5, -1\}$
für *Ja … Nein*.

**Score je Studiengang** – unabhängig von allen anderen:

$$\cos_p = \alpha\,\cos(u, c_p) + (1-\alpha)\,\overline{\text{top-}k\ \cos(u, \text{chunk} \in p)}$$

$$\text{Passung}_p = \mathrm{clip}\Big(\tfrac{\cos_p - \text{floor}}{\text{ceil} - \text{floor}},\,0,\,1\Big)\cdot 100\,\%$$

Es gibt **keine** Normierung über die Studiengänge (Summe ≠ 100 %). `floor`/`ceil` kalibrieren den
typischen Kosinus-Bereich des Embedding-Modells (`python -m core.calibrate`).

**Fragenauswahl** – für jeden Chunk der Top-Kandidaten wird die Standardabweichung seiner
Kosinus-Ähnlichkeit zu den Kandidaten-Profilen berechnet. Hohe Streuung bedeutet: Ein „Ja“
hebt manche Kandidaten und senkt andere – die Frage trennt gut. Bereits gefragte Themen werden
über ihre Ähnlichkeit zu früheren Fragen abgewertet.

## Projektstruktur

```
hsd-study-finder/
├── app.py                     # Streamlit-Frontend (Live-Demo)
├── config.py                  # zentrale Einstellungen (per .env überschreibbar)
├── requirements.txt
├── .env.example               # LLM-Provider, API-Keys, Embedding-Modell
├── .streamlit/config.toml     # Theme für die Messe-Präsentation
├── data/
│   ├── README.md              # wie echte Modulhandbücher eingebunden werden
│   └── programs/*.md          # BEISPIEL-Modulhandbücher (9 Studiengänge)
├── embeddings/
│   ├── loader.py              # Markdown/PDF einlesen, Front-Matter, Chunking (LangChain)
│   ├── embedder.py            # Sentence-Transformers, L2-normalisiert
│   ├── vector_store.py        # ChromaDB (Kosinus), Snapshot für torch, RAG-Retrieval
│   └── build_index.py         # Index bauen: python -m embeddings.build_index
├── core/
│   ├── similarity.py          # Kosinus-Ähnlichkeit, Kosinus → Prozent (ohne Softmax)
│   ├── scoring.py             # Nutzerprofil & unabhängige Studiengangs-Scores
│   ├── question_generator.py  # trennscharfe Merkmale + RAG + LLM → nächste Frage
│   ├── akinator.py            # Session-Zustand des Frage-Antwort-Loops
│   ├── llm.py                 # Provider-Abstraktion (Claude / OpenAI / Ollama / keiner)
│   └── calibrate.py           # floor/ceil für ein Embedding-Modell bestimmen
└── tests/test_scoring.py      # Tests der Kernmathematik (u. a. "kein Softmax")
```

## Setup

Voraussetzung: Python 3.10 oder neuer.

```bash
git clone <repo-url> hsd-study-finder
cd hsd-study-finder
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# optional, aber deutlich kleiner: CPU-Version von PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env                 # LLM-Provider/API-Key eintragen (optional)
python -m embeddings.build_index     # lädt beim ersten Mal das Embedding-Modell (~470 MB)
python -m core.calibrate             # optional: floor/ceil prüfen
streamlit run app.py
```

Tests: `pytest`

### LLM-Provider

Standard ist `LLM_PROVIDER=auto`:

| Reihenfolge | Provider | Bedingung |
|---|---|---|
| 1 | Anthropic Claude (`claude-sonnet-5-5`) | `ANTHROPIC_API_KEY` gesetzt |
| 2 | OpenAI (`gpt-4o-mini`) | `OPENAI_API_KEY` gesetzt |
| 3 | Ollama lokal (`llama3.1:8b`) | Ollama läuft unter `OLLAMA_BASE_URL` |
| 4 | Regelbasierter Fallback | immer – Fragen werden aus dem trennschärfsten Modul gebaut |

Damit läuft die Demo auch ohne Internet. Für die Messe empfiehlt sich: Claude als Hauptweg und
Ollama als Offline-Backup (`ollama pull llama3.1:8b`). Welches LLM aktiv ist, zeigt die Seitenleiste.

### Embedding-Modell

Standard: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 Dimensionen, mehrsprachig,
läuft flott auf CPU). Bessere Qualität bei mehr Rechenzeit: `paraphrase-multilingual-mpnet-base-v2`.
Nach einem Modellwechsel den Index neu bauen und `python -m core.calibrate` ausführen, dann
`COS_FLOOR`/`COS_CEIL` in der `.env` setzen.

## Tipps für die Live-Demo (KI-Con)

- **Vorher einmal starten**, damit Modell und Index im Cache liegen (`HF_HUB_OFFLINE=1` erzwingt danach Offline-Betrieb).
- Browser im Vollbild (F11), Zoom 110–125 % für Lesbarkeit aus zwei Metern Abstand.
- „🔍 Warum diese Frage?“ zeigt Besucherinnen und Besuchern das Merkmal und den Provider – gut, um die Technik zu erklären.
- Der Hinweis unter dem Diagramm zeigt live, dass die Summe der Scores **nicht** 100 % ist.

## Literatur

- Barz, B. & Denzler, J. (2020): *Deep Learning on Small Datasets without Pre-Training using Cosine Loss.* WACV.
- Kaya, M. & Bilge, H. Ş. (2019): *Deep Metric Learning: A Survey.* Symmetry 11(9).
- Lewis, P. et al. (2020): *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Rocchio, J. J. (1971): *Relevance Feedback in Information Retrieval.*

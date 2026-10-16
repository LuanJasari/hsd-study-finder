"""HSD Study-Finder – Streamlit-Frontend für die Live-Demo auf der KI-Con.

Start: ``streamlit run app.py``
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config import settings
from core import llm
from core.akinator import AkinatorSession
from core.ranking import LOW_MATCH_LABEL, bar_percent, low_match_note, percent_label
from embeddings.embedder import get_model
from embeddings.vector_store import load_snapshot

st.set_page_config(page_title="HSD Study-Finder", page_icon="🎓", layout="wide")

ACCENT = "#C8102E"
MUTED = "#C9CDD3"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1400px; }}
      h1 {{ font-size: 2.6rem !important; letter-spacing: -0.02em; }}
      .question-card {{
          background: var(--secondary-background-color, #F4F5F7);
          border-left: 6px solid {ACCENT};
          border-radius: 12px; padding: 1.6rem 1.8rem; margin: 0.5rem 0 1.2rem;
      }}
      .question-card .label {{ text-transform: uppercase; font-size: .8rem;
          letter-spacing: .08em; opacity: .65; margin-bottom: .4rem; }}
      .question-card .text {{ font-size: 1.75rem; line-height: 1.3; font-weight: 600; }}
      .question-card .detail {{ font-size: 1.05rem; margin-top: .6rem; opacity: .75; }}
      .stButton > button {{ font-size: 1.15rem; padding: .7rem 0; border-radius: 10px; }}
      .hint {{ opacity: .7; font-size: .9rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- Ressourcen
@st.cache_resource(show_spinner="Lade Embedding-Modell und Vektorindex …")
def load_resources():
    get_model()  # Modell vorab laden, damit die erste Frage schnell kommt
    snapshot = load_snapshot()
    provider_label, model = llm.get_chat_model()
    return snapshot, provider_label, model is not None, llm.provider_notes()


snapshot, provider_label, llm_active, provider_notes = load_resources()
if snapshot.is_empty:
    st.error("Der Vektorindex ist leer. Bitte zuerst `python -m embeddings.build_index` ausführen.")
    st.stop()


def new_session() -> None:
    st.session_state.session = AkinatorSession.create(snapshot)
    st.session_state.started = False
    st.session_state.explanations = None


if "session" not in st.session_state:
    new_session()
session: AkinatorSession = st.session_state.session


# ------------------------------------------------------------------- Diagramm
def score_chart(scores, highlight: set[str]) -> alt.Chart:
    df = pd.DataFrame(
        {
            "Studiengang": [s.name for s in scores],
            "Passung": [round(bar_percent(s), 1) for s in scores],   # Mindestbalken: gerechnet, aber kaum Übereinstimmung
            "Label": [percent_label(s) for s in scores],
            "Hinweis": [LOW_MATCH_LABEL if s.below_floor else "" for s in scores],
            "Kosinus": [round(s.cosine, 3) for s in scores],
            "Kandidat": [s.program_id in highlight for s in scores],
        }
    )
    order = df["Studiengang"].tolist()
    base = alt.Chart(df).encode(
        y=alt.Y("Studiengang:N", sort=order, title=None,
                axis=alt.Axis(labelFontSize=15, labelLimit=320, ticks=False, domain=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": 0.72}).encode(
        # Feste Skala 0–100 %: Scores sind unabhängig, die Summe ist bewusst nicht 100 %
        x=alt.X("Passung:Q", scale=alt.Scale(domain=[0, 100]), title="Passung in %",
                axis=alt.Axis(grid=True, gridOpacity=0.3, tickCount=5, labelFontSize=13)),
        color=alt.condition("datum.Kandidat", alt.value(ACCENT), alt.value(MUTED)),
        tooltip=[
            alt.Tooltip("Studiengang:N"),
            alt.Tooltip("Label:N", title="Passung"),
            alt.Tooltip("Kosinus:Q", format=".3f", title="Kosinus-Ähnlichkeit"),
            alt.Tooltip("Hinweis:N", title=" "),
        ],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=15, fontWeight="bold").encode(x="Passung:Q", text="Label:N")
    return (bars + labels).properties(height=max(320, 46 * len(df))).configure_view(stroke=None)


# ------------------------------------------------------------ Ergebnisdarstellung
def render_ranking(scores) -> None:
    """Alle Studiengänge untereinander, jeder mit seinem unabhängigen Wert (die Summe ist nicht 100 %)."""
    for rank, s in enumerate(scores, 1):
        name_col, value_col = st.columns([3, 1])
        name_col.markdown(f"**{rank}. {s.name}**")
        value_col.markdown(f"<div style='text-align:right;font-weight:700;font-size:1.15rem'>{percent_label(s)}</div>",
                           unsafe_allow_html=True)
        st.progress(int(bar_percent(s)))
        note = low_match_note(s)
        if note:
            st.caption(f"⚠️ {note}")
    st.markdown(f'<div class="hint">Jeder Wert steht für sich – die Summe beträgt hier '
                f'{sum(s.percent for s in scores):.0f} %, nicht 100 %.</div>', unsafe_allow_html=True)


def render_explanation(ex) -> None:
    st.markdown(ex.text)
    if ex.low_match:
        st.caption(f"⚠️ {LOW_MATCH_LABEL} (Kosinus {ex.cosine:.2f})")
    if ex.pro:
        st.markdown("**Dafür sprach:**  \n" + "  \n".join(f"✅ {line}" for line in ex.pro))
    if ex.contra:
        st.markdown("**Dagegen sprach:**  \n" + "  \n".join(f"⛔ {line}" for line in ex.contra))
    if ex.modules:
        st.markdown("**Nächste Module im Modulhandbuch:**")
        for title, excerpt in ex.modules:
            st.markdown(f"- *{title}*: {excerpt}")


# --------------------------------------------------------------------- Layout
head_left, head_right = st.columns([3, 1])
with head_left:
    st.title("🎓 HSD Study-Finder")
    st.markdown("Der Studiengangs-Akinator der Hochschule Düsseldorf – "
                "beantworte ein paar Fragen, und die KI grenzt deine passenden Studiengänge ein.")
with head_right:
    st.write("")
    if st.button("🔄 Neu starten", width="stretch"):
        new_session()
        st.rerun()

left, right = st.columns([1.1, 1], gap="large")

with left:
    if not st.session_state.started:
        st.markdown('<div class="question-card"><div class="label">Los geht\'s</div>'
                    '<div class="text">Erzähl kurz, was dich interessiert – oder starte direkt mit der ersten Frage.</div></div>',
                    unsafe_allow_html=True)
        intro = st.text_area("Was machst du gern? (optional)",
                             placeholder="z. B. Ich zeichne gern, mag Technik und habe schon kleine Websites gebaut …",
                             height=110, label_visibility="collapsed")
        if st.button("▶ Start", type="primary", width="stretch"):
            with st.spinner("Analysiere Modulhandbücher und formuliere die erste Frage …"):
                session.start(intro)
            st.session_state.started = True
            st.rerun()

    elif session.question is not None:
        q = session.question
        n = len(session.turns) + 1
        st.markdown(
            f'<div class="question-card"><div class="label">Frage {n}</div>'
            f'<div class="text">{q.text}</div>'
            + (f'<div class="detail">{q.detail}</div>' if q.detail else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(settings.answer_weights))
        for col, label in zip(cols, settings.answer_weights):
            if col.button(label, width="stretch", key=f"a{n}-{label}",
                          type="primary" if label == "Ja" else "secondary"):
                with st.spinner("Verschiebe deinen Vektor im Embedding-Raum …"):
                    session.answer(label)
                st.rerun()
        st.progress(min(n / settings.max_questions, 1.0),
                    text=f"Frage {n} von maximal {settings.max_questions}")
        with st.expander("🔍 Warum diese Frage?"):
            st.write(q.rationale or "–")
            st.caption(f"Merkmal: {q.probe_text}")
            st.caption(f"Generiert von: {q.source}")

    else:
        top = session.scores[0]
        st.markdown(
            f'<div class="question-card"><div class="label">Mein Tipp</div>'
            f'<div class="text">{top.name} passt am besten zu dir ({percent_label(top)}).</div></div>',
            unsafe_allow_html=True,
        )
        st.subheader("Rangliste aller Studiengänge")
        render_ranking(session.scores)
        if st.session_state.explanations is None:
            with st.spinner("Begründe die Empfehlung mit den Modulhandbüchern …"):
                st.session_state.explanations = session.explanations()
        st.subheader(f"Warum passen diese Studiengänge? (Top {len(st.session_state.explanations)})")
        for i, ex in enumerate(st.session_state.explanations, 1):
            with st.expander(f"{i}. {ex.name} – {ex.percent_label}", expanded=(i == 1)):
                render_explanation(ex)
        if st.button("❓ Weiter fragen", width="stretch"):
            session.continue_asking()
            st.session_state.explanations = None
            st.rerun()

    if session.turns:
        with st.expander(f"📝 Verlauf ({len(session.turns)} Antworten)"):
            for i, turn in enumerate(session.turns, 1):
                st.markdown(f"**{i}. {turn.question.text}**  \n→ {turn.answer}")

with right:
    st.subheader("Aktuelle Passung")
    if not st.session_state.started:
        st.info("Die Balken erscheinen, sobald du startest.")
    else:
        candidates = {s.program_id for s in session.scores[: settings.candidate_pool]}
        st.altair_chart(score_chart(session.scores, candidates), width="stretch")
        st.markdown(
            '<div class="hint">Jeder Wert ist eine <b>unabhängige</b> Kosinus-Ähnlichkeit zwischen deinem Profil '
            "und dem Studiengang – kein Softmax. Mehrere Studiengänge können gleichzeitig hoch liegen, "
            f"die Summe ist hier {sum(s.percent for s in session.scores):.0f} %.</div>",
            unsafe_allow_html=True,
        )

with st.sidebar:
    st.header("Technik")
    st.markdown(f"**LLM:** {'🟢' if llm_active else '🟠'} {provider_label}")
    with st.expander("Provider-Details"):
        for note in provider_notes:
            st.caption(f"• {note}")
        if not llm_active:
            st.caption("Fragen werden aus dem trennschärfsten Modulabschnitt gebaut. API-Key in `.env` eintragen "
                       "oder Ollama starten, um LLM-Fragen zu nutzen (Check: `python -m core.llm`).")
    st.markdown(f"**Embeddings:** `{settings.embedding_model.split('/')[-1]}`")
    st.markdown(f"**Index:** {len(snapshot.program_ids)} Studiengänge, {len(snapshot.chunk_ids)} Chunks (ChromaDB)")
    st.caption("Datengrundlage: offizielle Modulhandbücher des Fachbereichs Medien der HSD.")

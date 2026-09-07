import os
import threading
import time

import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.getenv("VF_API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="VeriFact",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: #05080f; }

.vf-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.85rem 1.75rem;
    background: rgba(15,23,42,0.95);
    border: 1px solid #1e293b;
    border-radius: 10px;
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}
.vf-logo { display: flex; align-items: center; gap: 0.6rem; }
.vf-logo-text {
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; color: #f1f5f9;
}
.vf-logo-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22d3ee; box-shadow: 0 0 8px #22d3ee88;
}
.vf-tagline { font-size: 0.78rem; color: #475569; border-left: 1px solid #1e293b; padding-left: 0.75rem; }
.vf-pill {
    font-size: 0.7rem; font-weight: 600; padding: 0.25rem 0.65rem;
    border-radius: 20px; border: 1px solid #22d3ee44; color: #22d3ee;
    background: #0c2231; letter-spacing: 0.04em; text-transform: uppercase;
}

.stat-row { display: flex; gap: 0.85rem; margin-bottom: 1.5rem; }
.stat-box {
    flex: 1; padding: 1.1rem 1.25rem;
    background: #0d1117; border: 1px solid #1e293b; border-radius: 8px;
    transition: border-color 0.2s;
}
.stat-box:hover { border-color: #334155; }
.stat-box .num { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; line-height: 1; }
.stat-box .lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.35rem; }

.section-head {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #475569; margin: 1.5rem 0 0.85rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-head::after { content:''; flex:1; height:1px; background:#1e293b; }

.finding-card {
    background: #0d1117; border: 1px solid #1e293b; border-radius: 8px;
    margin-bottom: 1rem; overflow: hidden;
}
.finding-card.corroborated { border-left: 3px solid #10b981; }
.finding-card.contradiction { border-left: 3px solid #ef4444; }
.finding-card.resolved { border-left: 3px solid #f59e0b; }
.finding-card.failure { border-left: 3px solid #6b7280; }

.finding-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.85rem 1.1rem; border-bottom: 1px solid #1a2236;
}
.finding-body { padding: 1rem 1.1rem; }
.finding-reasoning { font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 0.75rem; }
.finding-resolution {
    background: #0a1628; border: 1px solid #1e3a5f; border-radius: 6px;
    padding: 0.6rem 0.9rem; font-size: 0.82rem; color: #7dd3fc; margin-bottom: 0.75rem;
}

.vtag {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.07em;
    padding: 0.18rem 0.55rem; border-radius: 4px;
}
.vtag-c { background: #052e16; color: #4ade80; border: 1px solid #16a34a; }
.vtag-x { background: #2d0a0a; color: #f87171; border: 1px solid #b91c1c; }
.vtag-r { background: #2d1a00; color: #fbbf24; border: 1px solid #b45309; }
.vtag-f { background: #111827; color: #9ca3af; border: 1px solid #374151; }

.src-cell {
    background: #080c13; border: 1px solid #1a2236; border-radius: 6px; padding: 0.85rem;
}
.src-file { font-size: 0.8rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.src-meta { font-size: 0.73rem; color: #64748b; margin-bottom: 0.5rem; }
.src-entity { font-size: 0.82rem; color: #94a3b8; }
.src-value { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; margin-top: 0.2rem; }
.src-quote {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem; color: #64748b; line-height: 1.5;
    border-left: 2px solid #1e293b; padding-left: 0.6rem; margin-top: 0.6rem;
    white-space: pre-wrap; word-break: break-word;
}

.fact-row {
    background: #0d1117; border: 1px solid #1e293b; border-radius: 7px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
}
.fact-row.bad { border-left: 3px solid #ef4444; }
.fact-title { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
.fact-sub { font-size: 0.78rem; color: #64748b; margin-top: 0.15rem; }

.conf-track { background: #1e293b; border-radius: 2px; height: 3px; width: 100%; margin-top: 0.5rem; }
.conf-fill { height: 3px; border-radius: 2px; }

.audit-row {
    background: #080c13; border: 1px solid #1e293b;
    border-left: 3px solid #ef4444; border-radius: 6px;
    padding: 0.7rem 0.95rem; margin-bottom: 0.5rem;
    font-size: 0.82rem; color: #94a3b8;
}

div[data-testid="stTabs"] [data-testid="stTabsBar"] {
    background: transparent; border-bottom: 1px solid #1e293b;
}
div[data-testid="stTabs"] button {
    color: #475569 !important; font-size: 0.82rem !important;
    font-weight: 600 !important; padding: 0.55rem 1rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f1f5f9 !important; border-bottom: 2px solid #22d3ee !important;
}
div[data-testid="stFileUploader"] {
    border: 1px dashed #1e293b !important; border-radius: 8px !important;
    background: #0d1117 !important;
}
.stButton > button {
    background: #0d1117 !important; color: #e2e8f0 !important;
    border: 1px solid #1e293b !important; border-radius: 6px !important;
    font-size: 0.83rem !important; font-weight: 500 !important;
}
.stButton > button:hover { border-color: #334155 !important; background: #151f2e !important; }
button[kind="primary"] {
    background: #0e7490 !important; border-color: #0891b2 !important; color: #f0fdff !important;
}
button[kind="primary"]:hover { background: #0c6a82 !important; }

p, li, span, label { color: #94a3b8 !important; }
h1,h2,h3,h4,h5,h6 { color: #f1f5f9 !important; font-weight: 600; letter-spacing: -0.02em; }
.stAlert { border-radius: 6px !important; }
.stDataFrame { border: 1px solid #1e293b !important; border-radius: 8px !important; }
hr { border-color: #1e293b !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="vf-nav">
  <div class="vf-logo">
    <div class="vf-logo-dot"></div>
    <span class="vf-logo-text">VeriFact</span>
    <span class="vf-tagline">Cross-Document Fact Extraction &amp; Reconciliation</span>
  </div>
  <span class="vf-pill">v1.0</span>
</div>
""", unsafe_allow_html=True)

VERDICT_CFG = {
    "CORROBORATED":        ("corroborated", "vtag-c", "CORROBORATED"),
    "CONTRADICTION":       ("contradiction", "vtag-x", "CONTRADICTION"),
    "RESOLVED_BY_CONTEXT": ("resolved",      "vtag-r", "RESOLVED"),
    "EXTRACTION_FAILURE":  ("failure",       "vtag-f", "FLAGGED"),
}


def api(method: str, path: str, **kw):
    try:
        r = getattr(requests, method)(f"{API_BASE}{path}", timeout=30, **kw)
        r.raise_for_status()
        return r.json(), None
    except requests.RequestException as e:
        return None, str(e)


def conf_bar(c: float) -> str:
    pct = int(c * 100)
    col = "#10b981" if c >= 0.75 else ("#f59e0b" if c >= 0.5 else "#ef4444")
    return (
        f'<div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{col};"></div></div>'
        f'<span style="font-size:0.7rem;color:#475569;">{pct}% confidence</span>'
    )


stats, err = api("get", "/stats")
if err:
    st.error(f"Cannot reach backend at `{API_BASE}` — is it running?  \n`{err}`")
    st.stop()

tab_findings, tab_facts, tab_ingest, tab_dash = st.tabs(
    ["Findings", "Facts", "Ingest", "Analytics"]
)

with tab_findings:
    st.markdown('<div class="section-head">Cross-Document Evidence</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1.5, 2, 1.5])
    with fc1:
        verdict_sel = st.selectbox(
            "Verdict", ["All", "CONTRADICTION", "RESOLVED_BY_CONTEXT", "CORROBORATED", "EXTRACTION_FAILURE"],
            key="v_sel", label_visibility="collapsed"
        )
    with fc2:
        search = st.text_input("Search", placeholder="entity, claim, or keyword...", key="v_search", label_visibility="collapsed")
    with fc3:
        src_filter = st.text_input("Source file", placeholder="filename...", key="v_src", label_visibility="collapsed")

    params = {} if verdict_sel == "All" else {"relationship_type": verdict_sel}
    rels, rel_err = api("get", "/relationships", params=params)
    if rel_err:
        st.error(rel_err)
        rels = []
    rels = rels or []

    if search:
        q = search.lower()
        rels = [r for r in rels if q in r["explanation"].lower()
                or any(q in f["entity"].lower() or q in f["metric_or_claim"].lower() for f in r.get("facts", []))
                or (r.get("resolution_context") and q in r["resolution_context"].lower())]
    if src_filter:
        sq = src_filter.lower()
        rels = [r for r in rels if any(sq in f["source_filename"].lower() for f in r.get("facts", []))]

    st.caption(f"{len(rels)} finding(s)")

    if not rels:
        st.info("No findings yet — ingest some PDFs first.")

    for rel in rels:
        rtype = rel["relationship_type"]
        css_cls, tag_cls, tag_lbl = VERDICT_CFG.get(rtype, ("failure", "vtag-f", rtype))

        st.markdown(f'<div class="finding-card {css_cls}"><div class="finding-header">'
                    f'<span class="vtag {tag_cls}">{tag_lbl}</span>'
                    f'<span style="font-size:0.7rem;color:#334155;font-family:JetBrains Mono,monospace;">'
                    f'{rel["relationship_id"]}</span></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="finding-body"><div class="finding-reasoning"><strong style="color:#e2e8f0;">Finding:</strong> {rel["explanation"]}</div>', unsafe_allow_html=True)

        if rel.get("resolution_context"):
            st.markdown(f'<div class="finding-resolution"><strong>Context:</strong> {rel["resolution_context"]}</div>', unsafe_allow_html=True)

        facts_in_rel = rel.get("facts", [])
        if facts_in_rel:
            cols = st.columns(len(facts_in_rel))
            for col, f in zip(cols, facts_in_rel):
                with col:
                    g_color = "#10b981" if f["grounded"] else "#ef4444"
                    g_lbl = "grounded" if f["grounded"] else "unverified"
                    val_str = f"{f.get('value') or ''} {f.get('unit') or ''}".strip() or "—"
                    period = f" · {f['time_period']}" if f.get("time_period") else ""
                    st.markdown(
                        f'<div class="src-cell">'
                        f'<div class="src-file">{f["source_filename"]}</div>'
                        f'<div class="src-meta">p.{f["page_number"]} · <span style="color:{g_color};">{g_lbl}</span></div>'
                        f'<div class="src-entity"><strong style="color:#cbd5e1;">{f["entity"]}</strong> — {f["metric_or_claim"]}</div>'
                        f'<div class="src-value">{val_str}{period}</div>'
                        f'<div class="src-quote">{f["exact_quote"][:280]}{"..." if len(f["exact_quote"])>280 else ""}</div>'
                        f'</div>', unsafe_allow_html=True
                    )

        st.markdown('</div></div>', unsafe_allow_html=True)


with tab_facts:
    st.markdown('<div class="section-head">Extracted Facts</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns([2, 2, 1, 1])
    with r1:
        fe_entity = st.text_input("Entity", placeholder="e.g. RBI, Delhivery...", key="fe_e", label_visibility="collapsed")
    with r2:
        fe_doc = st.text_input("Document ID", placeholder="doc_...", key="fe_d", label_visibility="collapsed")
    with r3:
        fe_ung = st.checkbox("Ungrounded", key="fe_ung")
    with r4:
        fe_low = st.checkbox("Low conf.", key="fe_low")

    fp = {}
    if fe_entity: fp["entity"] = fe_entity
    if fe_doc:    fp["document_id"] = fe_doc

    facts_data, fe_err = api("get", "/facts", params=fp)
    if fe_err:
        st.error(fe_err)
        facts_data = []
    facts_data = facts_data or []
    if fe_ung: facts_data = [f for f in facts_data if not f["grounded"]]
    if fe_low:  facts_data = [f for f in facts_data if f["confidence"] < 0.6]

    st.caption(f"{len(facts_data)} fact(s)")

    for f in facts_data:
        title = f"{f['entity']} — {f['metric_or_claim']}"
        if f.get("value"):   title += f" · {f['value']} {f.get('unit') or ''}"
        if f.get("time_period"): title += f" · {f['time_period']}"

        row_cls = "fact-row bad" if not f["grounded"] else "fact-row"
        with st.expander(title):
            if not f["grounded"]:
                st.warning("Quote not found verbatim in source text — treated as unverified.")

            left, right = st.columns([3, 2])
            with left:
                st.markdown(f"**Context:** {f['context']}")
                st.markdown(
                    f'<div class="src-quote" style="border-left:2px solid #334155;padding:0.6rem 0.9rem;margin-top:0.5rem;">{f["exact_quote"]}</div>',
                    unsafe_allow_html=True
                )
                if f.get("extra"):
                    st.json(f["extra"])
            with right:
                st.markdown(f"**Source:** `{f['source_filename']}`")
                st.markdown(f"**Page:** {f['page_number']} &nbsp;|&nbsp; **Period:** {f.get('time_period') or 'n/a'}")
                g_c = "#10b981" if f["grounded"] else "#ef4444"
                g_l = "Grounded" if f["grounded"] else "Unverified"
                st.markdown(f'<span style="color:{g_c};font-weight:600;font-size:0.8rem;">{g_l}</span>', unsafe_allow_html=True)
                st.markdown(conf_bar(f["confidence"]), unsafe_allow_html=True)
                st.caption(f"`{f['fact_id']}`")


with tab_ingest:
    st.markdown('<div class="section-head">Document Ingestion</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        st.markdown(f"**{len(uploaded)} file(s) ready:**")
        for u in uploaded:
            st.markdown(f"- `{u.name}` &nbsp; ({u.size/1024:.1f} KB)")

    if uploaded and st.button("Run Ingestion Pipeline", type="primary"):
        payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded]
        result: dict = {}

        def _upload():
            try:
                r = requests.post(f"{API_BASE}/documents/upload", files=payload, timeout=1800)
                r.raise_for_status()
                result["ok"] = r.json()
            except requests.RequestException as e:
                result["err"] = str(e)

        t = threading.Thread(target=_upload, daemon=True)
        t.start()
        bar = st.progress(0, text="Initializing...")
        elapsed = 0
        est = max(20, len(uploaded) * 30)
        while t.is_alive():
            time.sleep(0.5)
            elapsed += 0.5
            pct = min(93, int(elapsed / est * 100))
            bar.progress(pct, text=f"Extracting & reconciling... ({int(elapsed)}s)")
        t.join()
        bar.progress(100, text="Complete")

        if "err" in result:
            st.error(f"Upload failed: {result['err']}")
        else:
            for item in result["ok"]["processed"]:
                if item["status"] == "failed":
                    st.error(f"**{item['filename']}** — {item.get('error') or 'Extraction failed.'}")
                else:
                    st.success(
                        f"**{item['filename']}** — "
                        f"{item['page_count']} pages, "
                        f"{item['facts_extracted']} facts, "
                        f"{item['new_relationships']} relationships"
                    )

    st.markdown("---")
    st.markdown('<div class="section-head">Stored Documents</div>', unsafe_allow_html=True)

    docs, derr = api("get", "/documents")
    if derr:
        st.error(derr)
    elif docs:
        for d in docs:
            c1, c2, c3 = st.columns([4, 1, 1])
            with c1:
                st.markdown(f"**{d['filename']}** — {d['page_count']} pages, {d['fact_count']} facts")
            with c2:
                sc = "#10b981" if d["status"] == "indexed" else "#f59e0b"
                st.markdown(f'<span style="color:{sc};font-size:0.8rem;font-weight:600;">{d["status"]}</span>', unsafe_allow_html=True)
            with c3:
                if st.button("Delete", key=f"del_{d['document_id']}"):
                    res, derr2 = api("delete", f"/documents/{d['document_id']}")
                    if derr2: st.error(derr2)
                    else:
                        st.success("Deleted.")
                        st.rerun()
    else:
        st.info("No documents stored yet.")

    st.markdown("---")
    if st.button("Re-run Global Reconciliation"):
        with st.spinner("Re-clustering all facts..."):
            res, rerr = api("post", "/reconcile/rerun")
            if rerr: st.error(rerr)
            else:
                st.success(f"{res['facts_considered']} facts · {res['new_relationships']} new relationships found.")


with tab_dash:
    st.markdown('<div class="section-head">Overview</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="stat-row">'
        + "".join(
            f'<div class="stat-box"><div class="num">{v:,}</div><div class="lbl">{l}</div></div>'
            for v, l in [
                (stats["total_documents"],    "Documents"),
                (stats["total_facts"],        "Facts"),
                (stats["grounded_facts"],     "Grounded"),
                (stats["total_relationships"],"Relationships"),
                (stats["total_failures"],     "Flagged"),
            ]
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    ch1, ch2 = st.columns([3, 2])
    with ch1:
        st.markdown('<div class="section-head">Relationship Distribution</div>', unsafe_allow_html=True)
        rel_map = stats.get("relationships_by_type", {})
        if rel_map:
            cmap = {"CORROBORATED":"#10b981","CONTRADICTION":"#ef4444","RESOLVED_BY_CONTEXT":"#f59e0b","EXTRACTION_FAILURE":"#475569"}
            fig = go.Figure(go.Bar(
                x=list(rel_map.keys()), y=list(rel_map.values()),
                marker_color=[cmap.get(k,"#22d3ee") for k in rel_map],
                text=list(rel_map.values()), textposition="outside",
                textfont=dict(color="#64748b", size=12),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#475569", family="Inter"),
                margin=dict(l=0,r=0,t=8,b=0), height=220,
                xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#475569")),
                yaxis=dict(showgrid=True, gridcolor="#0f1923", tickfont=dict(size=10, color="#475569")),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No relationships yet.")

    with ch2:
        st.markdown('<div class="section-head">Grounding Rate</div>', unsafe_allow_html=True)
        g, u = stats["grounded_facts"], stats["ungrounded_facts"]
        if g + u > 0:
            fig2 = go.Figure(go.Pie(
                labels=["Grounded","Unverified"], values=[g, u], hole=0.62,
                marker_colors=["#10b981","#ef4444"], textfont=dict(color="#94a3b8"),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#475569", family="Inter"),
                margin=dict(l=0,r=0,t=8,b=0), height=220,
                legend=dict(font=dict(color="#64748b", size=11)),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No facts yet.")

    st.markdown('<div class="section-head">Recent Documents</div>', unsafe_allow_html=True)
    docs2, _ = api("get", "/documents")
    if docs2:
        rows = [{"File": d["filename"], "Pages": d["page_count"], "Facts": d["fact_count"],
                 "Status": d["status"], "Ingested": d.get("ingested_at","")[:19].replace("T"," ")}
                for d in docs2[:10]]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No documents ingested yet.")

    st.markdown('<div class="section-head">Extraction Issues Log</div>', unsafe_allow_html=True)
    fails, _ = api("get", "/failures")
    if fails:
        grouped = {}
        for item in fails:
            grouped.setdefault(item["source_filename"], []).append(item)
        for fname, flist in grouped.items():
            with st.expander(f"{fname} — {len(flist)} issue(s)"):
                for entry in flist:
                    pg = entry["page_number"] if entry["page_number"] != -1 else "n/a"
                    excerpt = f'<div class="src-quote" style="margin-top:0.4rem;">{entry["raw_excerpt"]}</div>' if entry.get("raw_excerpt") else ""
                    st.markdown(
                        f'<div class="audit-row"><strong>Page {pg}</strong> &nbsp;|&nbsp; '
                        f'<code>{entry["chunk_id"]}</code><br>{entry["reason"]}{excerpt}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No issues logged.")

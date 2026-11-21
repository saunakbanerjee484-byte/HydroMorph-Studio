"""Pink & blue glassmorphism theme, injected as CSS."""
import streamlit as st

_CSS = """
<style>
:root{
  --pink:#ff5fa8; --pink-soft:rgba(255,95,168,.22);
  --blue:#4a8bff; --blue-soft:rgba(74,139,255,.22);
  --ink:#1b2447; --ink-soft:#4a5580;
  --glass:rgba(255,255,255,.42); --glass-strong:rgba(255,255,255,.62);
  --edge:rgba(255,255,255,.75);
  color-scheme: light;
}
.stApp{
  background:
    radial-gradient(900px 600px at 8% 6%, rgba(255,120,185,.55), transparent 60%),
    radial-gradient(900px 700px at 96% 12%, rgba(90,150,255,.55), transparent 60%),
    radial-gradient(800px 700px at 50% 100%, rgba(190,140,255,.40), transparent 60%),
    linear-gradient(135deg,#ffe3f1 0%,#e0ecff 100%);
  background-attachment: fixed;
  color: var(--ink);
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp div[data-testid="stMarkdownContainer"]{ color: var(--ink); }
header[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top:2rem; max-width:1200px; }

/* glass surfaces */
section[data-testid="stSidebar"]{
  background: var(--glass); backdrop-filter: blur(18px) saturate(160%); -webkit-backdrop-filter: blur(18px) saturate(160%);
  border-right:1px solid var(--edge);
}
div[data-testid="stExpander"], div[data-testid="stMetric"], div[data-testid="stFileUploader"] section,
div[data-testid="stAlert"], .glass{
  background: var(--glass); backdrop-filter: blur(14px) saturate(150%); -webkit-backdrop-filter: blur(14px) saturate(150%);
  border:1px solid var(--edge); border-radius:18px;
  box-shadow: 0 8px 30px rgba(74,100,200,.14), inset 0 1px 0 rgba(255,255,255,.7);
}
/* Session Management container (st.container(key="session_mgmt")) */
.st-key-session_mgmt{
  background: linear-gradient(135deg, rgba(255,95,168,.20), rgba(74,139,255,.20));
  backdrop-filter: blur(16px) saturate(160%); -webkit-backdrop-filter: blur(16px) saturate(160%);
  border:1px solid var(--edge); border-radius:18px; padding:.9rem 1rem;
  box-shadow: 0 8px 30px rgba(74,100,200,.18), inset 0 1px 0 rgba(255,255,255,.8);
}
.st-key-session_mgmt h4{ margin:0 0 .5rem; }
div[data-testid="stExpander"] details{ border:none; background:transparent; }
div[data-testid="stMetric"]{ padding:.8rem 1rem; border-top:3px solid transparent;
  border-image: linear-gradient(90deg,var(--pink),var(--blue)) 1; border-radius:0 0 18px 18px; }
div[data-testid="stMetricValue"]{ color:var(--ink); }
.hero{ padding:1.4rem 1.6rem; margin-bottom:1rem; }
.hero h1{ margin:0; font-size:1.9rem; line-height:1.15;
  background:linear-gradient(90deg,var(--pink),var(--blue)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p{ margin:.4rem 0 0; color:var(--ink-soft); }

/* inputs */
div[data-baseweb="input"], div[data-baseweb="base-input"], .stNumberInput input{
  background: var(--glass-strong) !important; color: var(--ink) !important;
  border-radius:12px !important; border:1px solid var(--edge) !important;
}
.stNumberInput button{ background:transparent !important; color:var(--ink) !important; }
.stNumberInput input:focus{ box-shadow:0 0 0 2px var(--blue-soft) !important; }

/* buttons */
.stButton > button, .stDownloadButton > button{
  border-radius:14px; border:1px solid var(--edge); color:var(--ink); background:var(--glass-strong);
  backdrop-filter: blur(10px); font-weight:600;
}
.stButton > button[kind="primary"]{
  background: linear-gradient(90deg,var(--pink),var(--blue)); color:#fff; border:none;
  box-shadow:0 10px 26px rgba(120,110,255,.35);
}
.stButton > button:disabled{ opacity:.45; box-shadow:none; }
button[data-baseweb="tab"]{ background:transparent; color:var(--ink-soft); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
div[data-baseweb="tab-highlight"]{ background:linear-gradient(90deg,var(--pink),var(--blue)); }
div[data-testid="stRadio"] label{ background:var(--glass); border:1px solid var(--edge); border-radius:12px; padding:.35rem .7rem; margin-bottom:.3rem; }
.katex{ color:var(--ink); }
@media (max-width:640px){ .hero h1{ font-size:1.45rem; } .block-container{ padding:1rem .8rem; } }
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="glass hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

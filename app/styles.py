"""CSS for the ReadRadar Streamlit app. Imported once by app/app.py."""

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ─── base palette ─────────────────────────────────────────────────────── */
:root {
    --bg:        #f7f3ee;
    --surface:   #ffffff;
    --surface-2: #fbf6ee;
    --border:    #e6dccc;
    --border-2:  #d5c9b8;
    --ink:       #1a1210;
    --ink-2:     #3a2e26;
    --muted:     #8a7a6a;
    --muted-2:   #b0a090;
    --accent:    #b87333;
    --accent-2:  #9e6228;
    --accent-3:  #fdf6ec;
    --good:      #3a8c5c;
    --bad:       #a03030;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--ink-2) !important;
    font-family: 'Inter', 'DM Sans', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background-color: var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.75rem; max-width: 1100px; }

p, li, span, div, label { color: var(--ink-2); }
.stMarkdown p { color: var(--ink-2) !important; line-height: 1.55; }

/* ─── hero ─────────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 1.6rem 0 1.2rem;
    margin-bottom: 0.4rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.1rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--ink);
    margin: 0;
    line-height: 1.05;
}
.hero .subtitle {
    font-size: 0.72rem;
    color: var(--muted-2);
    margin-top: 0.55rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}
.radar-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    margin-right: 10px;
    vertical-align: middle;
    box-shadow: 0 0 0 0 rgba(184,115,51,0.4);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(184,115,51,0.45); }
    70%  { box-shadow: 0 0 0 8px rgba(184,115,51,0); }
    100% { box-shadow: 0 0 0 0 rgba(184,115,51,0); }
}

/* ─── pill navigation (radio restyled) ─────────────────────────────────── */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] {
    display: flex;
    justify-content: center;
    margin-bottom: 1.4rem;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: inline-flex;
    background: #ede6da;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px;
    gap: 2px;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    display: inline-flex;
    align-items: center;
    margin: 0 !important;
    padding: 0.5rem 1.4rem !important;
    border-radius: 999px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    background: transparent;
    transition: background .18s ease, color .18s ease, box-shadow .18s ease;
    white-space: nowrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] div[role="radiogroup"] label:hover { color: var(--ink); }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: var(--surface);
    color: var(--accent);
    box-shadow: 0 1px 4px rgba(26,18,16,0.08);
}

/* ─── inputs ───────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-2) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(184,115,51,0.12) !important;
}

/* ─── buttons ──────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    transition: background .18s, color .18s, border-color .18s, transform .08s !important;
    border: 1.5px solid transparent !important;
    white-space: nowrap !important;
}
/* Compact icon-button variant.
   Streamlit renders the button's label inside the button element and
   may set aria-label and/or title from either the label or the `help`
   argument depending on the version. Match any of those so the rule
   lands regardless of Streamlit release. */
.stButton > button[aria-label="✕"],
.stButton > button[title="Remove from favorites"],
.stButton > button[aria-label="Remove from favorites"] {
    padding: 0.3rem 0.55rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    font-size: 1rem !important;
    line-height: 1 !important;
    min-width: 0 !important;
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[aria-label="✕"]:hover,
.stButton > button[title="Remove from favorites"]:hover,
.stButton > button[aria-label="Remove from favorites"]:hover {
    background: #fdf2f2 !important;
    color: var(--bad) !important;
    border-color: #e2c3c3 !important;
    transform: none !important;
}
.stButton > button[kind="primary"],
.stButton > button:not([kind]):not([data-testid*="FormSubmit"]) {
    background: var(--accent) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button:not([kind]):not([data-testid*="FormSubmit"]):hover {
    background: var(--accent-2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--accent-3) !important;
}
button[data-testid="stFormSubmitButton"] {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
button[data-testid="stFormSubmitButton"]:hover { background: var(--accent-2) !important; }

/* ─── section labels ───────────────────────────────────────────────────── */
.section-label {
    font-size: 0.66rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted-2);
    font-weight: 500;
    margin: 1.3rem 0 0.7rem 0;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0.4rem 0 1rem 0;
    letter-spacing: -0.3px;
}
.result-count {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0.3rem 0 0.8rem 0;
}

/* ─── book card ────────────────────────────────────────────────────────── */
.book-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 2px rgba(26,18,16,0.03);
    transition: border-color .2s, box-shadow .2s, transform .15s;
}
.book-card:hover {
    border-color: var(--border-2);
    box-shadow: 0 4px 16px rgba(184,115,51,0.08);
    transform: translateY(-1px);
}
.bc-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.25;
    margin-bottom: 0.3rem;
}
.bc-meta {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.bc-meta .dot { opacity: 0.4; }
.bc-stars { color: #c8920a; letter-spacing: 1px; }
.bc-score {
    color: var(--accent);
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.72rem;
    padding: 0.15rem 0.55rem;
    background: var(--accent-3);
    border-radius: 999px;
    border: 1px solid #e7d4b5;
}
.bc-desc {
    font-size: 0.87rem;
    color: var(--ink-2);
    line-height: 1.55;
    margin: 0.3rem 0 0.5rem 0;
}
.bc-tags { margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.3rem; }
.bc-tag {
    display: inline-block;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: var(--ink-2);
    text-transform: uppercase;
    font-weight: 500;
}

/* ─── cover (card + modal) ─────────────────────────────────────────────── */
.cover-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 2 / 3;
    border-radius: 10px;
    overflow: hidden;
    background: #eadfce;
    box-shadow: 0 2px 8px rgba(26,18,16,0.08);
}
.cover-placeholder {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    background:
        linear-gradient(145deg, #c89764 0%, #9a6430 60%, #7a4a20 100%);
    color: #fff7e8;
    padding: 0.6rem;
}
.cover-placeholder::before {
    content: "";
    position: absolute;
    top: 0; left: 10%;
    height: 100%;
    width: 1.5px;
    background: rgba(0,0,0,0.18);
}
.cover-placeholder .ph-initials {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.cover-placeholder .ph-label {
    font-size: 0.55rem;
    text-transform: uppercase;
    letter-spacing: 0.26em;
    color: rgba(255,247,232,0.75);
}
.cover-img {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    z-index: 1;
    background: #eadfce;
}

/* ─── empty state ──────────────────────────────────────────────────────── */
.empty-state {
    background: var(--surface);
    border: 1px dashed var(--border-2);
    border-radius: 14px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: var(--muted);
    margin-bottom: 1.2rem;
}
.empty-state .es-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.6;
}
.empty-state .es-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--ink);
    margin-bottom: 0.3rem;
}
.empty-state .es-body {
    font-size: 0.86rem;
    color: var(--muted);
}

/* ─── search tips ──────────────────────────────────────────────────────── */
.tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.7rem;
    margin-top: 0.4rem;
}
.tips-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
}
.tips-card .tc-label {
    font-size: 0.66rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.tips-card .tc-example {
    display: inline-block;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.2rem 0.15rem 0;
    font-size: 0.78rem;
    color: var(--ink-2);
    font-family: 'Inter', sans-serif;
}

/* ─── expander ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-top: 0.6rem;
}
details[data-testid="stExpander"] > summary {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1rem !important;
}

/* ─── pagination ───────────────────────────────────────────────────────── */
.page-indicator {
    text-align: center;
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    padding-top: 0.6rem;
}

/* ─── detail modal ─────────────────────────────────────────────────────── */
.dm-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.2;
    margin: 0 0 0.3rem 0;
}
.dm-sub {
    font-size: 0.88rem;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.dm-section-label {
    font-size: 0.66rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted-2);
    font-weight: 600;
    margin-top: 1.1rem;
    margin-bottom: 0.45rem;
}
.dm-desc {
    font-size: 0.92rem;
    line-height: 1.6;
    color: var(--ink-2);
}
.dm-meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.5rem 1rem;
}
.dm-meta-item {
    font-size: 0.82rem;
    color: var(--ink-2);
}
.dm-meta-item .label {
    display: block;
    font-size: 0.64rem;
    color: var(--muted-2);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.1rem;
}
.dm-judgment {
    background: var(--surface-2);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--ink-2);
    margin-top: 0.4rem;
}

/* ─── misc ─────────────────────────────────────────────────────────────── */
hr { border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }
.stAlert { border-radius: 10px !important; }

.data-credit {
    text-align: center;
    font-size: 0.7rem;
    color: var(--muted-2);
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.08em;
}
.data-credit a { color: var(--muted); text-decoration: none; }
.data-credit a:hover { color: var(--accent); }
</style>
"""

"""
☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard
Single-file Streamlit app | Dark theme | Plotly interactive charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── sklearn ────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, precision_recall_curve
)
from itertools import combinations
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleepy Owl Coffee — Cross-Sell Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette — Light theme ─────────────────────────────────────────────
PRIMARY   = "#F5F7FA"   # off-white page background
CARD_BG   = "#FFFFFF"   # white cards
SIDEBAR_BG= "#E8EDF4"   # cool light sidebar
ACCENT    = "#1B6CA8"   # deep blue
ACCENT2   = "#154E7A"   # darker blue
TEAL      = "#0E8C75"   # teal
MUTED     = "#0A6B59"   # dark teal
HIGHLIGHT = "#0E8C75"   # positive highlight
NAVY2     = "#CBD5E1"   # light grey divider
RED       = "#C0392B"   # muted red
AMBER     = "#B7770D"   # muted amber
TEXT_MAIN = "#1C2B3A"   # body text
TEXT_SUB  = "#4A6580"   # secondary text

# ── Global CSS — Light theme ──────────────────────────────────────────
st.markdown(f"""
<style>
  /* ══ BASE & PAGE ═══════════════════════════════════════════════════ */
  .stApp {{
    background-color: {PRIMARY};
    color: {TEXT_MAIN};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}

  /* Kill default coloured header bar */
  header[data-testid="stHeader"] {{
    background-color: {PRIMARY} !important;
    border-bottom: 1px solid {NAVY2};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  [data-testid="stDecoration"] {{ display: none; }}
  .stToolbar {{ background-color: {PRIMARY} !important; }}

  /* ══ SIDEBAR ════════════════════════════════════════════════════════ */
  section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
    border-right: 1px solid {NAVY2};
  }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] div {{
    color: {TEXT_MAIN} !important;
  }}

  /* Radio buttons */
  [data-testid="stSidebar"] [data-baseweb="radio"] label {{
    color: {TEXT_MAIN} !important;
    font-size: 0.88rem;
    padding: 3px 0;
  }}
  [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"] {{
    border-color: {ACCENT} !important;
    background: transparent !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"] {{
    border-color: {TEAL} !important;
    background-color: {TEAL} !important;
  }}
  [data-testid="stSidebar"] [data-baseweb="radio"] [role="radio"][aria-checked="true"]::after {{
    background-color: white !important;
  }}

  /* ══ SLIDERS ════════════════════════════════════════════════════════ */
  [data-testid="stSlider"] label {{ color: {TEXT_SUB} !important; font-size: 0.85rem !important; }}
  [data-testid="stSlider"] [data-testid="stThumbValue"] {{
    background: {ACCENT} !important;
    color: white !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-weight: 600;
    font-size: 0.78rem;
  }}
  div[class*="StyledSliderTrack"] {{ background: {NAVY2} !important; }}
  div[class*="StyledSliderInnerTrack"] {{ background: {ACCENT} !important; }}
  div[class*="StyledSliderThumb"] {{
    background: {ACCENT} !important;
    border: 2px solid white !important;
    box-shadow: 0 1px 4px rgba(27,108,168,0.35) !important;
  }}

  /* ══ BUTTONS ════════════════════════════════════════════════════════ */
  .stButton > button {{
    background: {ACCENT} !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.2rem !important;
    box-shadow: 0 2px 6px rgba(27,108,168,0.25) !important;
    transition: background 0.2s, box-shadow 0.2s;
  }}
  .stButton > button:hover {{
    background: {ACCENT2} !important;
    box-shadow: 0 3px 10px rgba(27,108,168,0.35) !important;
  }}

  /* ══ SELECTBOX / MULTISELECT ════════════════════════════════════════ */
  [data-baseweb="select"] div {{
    background-color: white !important;
    border-color: {NAVY2} !important;
    color: {TEXT_MAIN} !important;
  }}
  [data-baseweb="menu"] {{
    background-color: white !important;
    border: 1px solid {NAVY2} !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
  }}
  [data-baseweb="option"]:hover {{ background-color: #EEF4FB !important; }}
  [data-baseweb="tag"] {{
    background-color: #DDEEFF !important;
    color: {ACCENT2} !important;
    border-radius: 4px !important;
  }}

  /* ══ TABS ════════════════════════════════════════════════════════════ */
  [data-baseweb="tab-list"] {{
    background-color: {CARD_BG} !important;
    border-bottom: 1px solid {NAVY2};
    gap: 2px;
  }}
  button[data-baseweb="tab"] {{
    color: {TEXT_SUB} !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.85rem !important;
    transition: all 0.15s;
  }}
  button[data-baseweb="tab"]:hover {{
    color: {ACCENT} !important;
    background: #EEF4FB !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
    font-weight: 600 !important;
    background: #F0F6FF !important;
  }}
  [data-baseweb="tab-panel"] {{
    background-color: {PRIMARY} !important;
    padding-top: 1rem !important;
  }}

  /* ══ DATAFRAMES ═════════════════════════════════════════════════════ */
  [data-testid="stDataFrame"] {{
    background: white;
    border-radius: 8px;
    border: 1px solid {NAVY2};
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  [data-testid="stDataFrame"] th {{
    background: #EEF4FB !important;
    color: {ACCENT2} !important;
    font-size: 0.8rem;
    font-weight: 700;
    border-bottom: 1px solid {NAVY2} !important;
  }}
  [data-testid="stDataFrame"] td {{
    color: {TEXT_MAIN} !important;
    font-size: 0.82rem;
  }}

  /* ══ COMPONENT CARDS ════════════════════════════════════════════════ */
  /* metric card */
  .metric-card {{
    background: {CARD_BG};
    border-left: 4px solid {TEAL};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
  }}
  .metric-card .label {{
    font-size: 0.7rem;
    color: {TEXT_SUB};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.25rem;
    font-weight: 600;
  }}
  .metric-card .value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {ACCENT};
    line-height: 1.1;
  }}
  .metric-card .delta {{
    font-size: 0.72rem;
    color: {TEXT_SUB};
    margin-top: 0.2rem;
  }}

  /* insight card */
  .insight-card {{
    background: {CARD_BG};
    border-left: 4px solid {ACCENT};
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.86rem;
    color: {TEXT_MAIN};
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .insight-card strong {{ color: {ACCENT2}; }}

  /* ARM action card */
  .action-card {{
    background: #F0FBF7;
    border: 1px solid #A8D8C8;
    border-top: 3px solid {TEAL};
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
    color: {TEXT_MAIN};
    line-height: 1.6;
  }}
  .action-card strong {{ color: {MUTED}; }}

  /* dataset info card */
  .dataset-card {{
    background: white;
    border: 1px solid {NAVY2};
    border-radius: 8px;
    padding: 0.9rem;
    font-size: 0.78rem;
    color: {TEXT_SUB};
    line-height: 2;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .dataset-card b {{ color: {ACCENT}; }}

  /* persona card */
  .persona-card {{
    background: {CARD_BG};
    border: 1px solid {NAVY2};
    border-top: 3px solid {TEAL};
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-size: 0.82rem;
    color: {TEXT_MAIN};
    line-height: 1.6;
    height: 100%;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
  }}
  .persona-card .persona-name {{
    font-size: 1rem;
    font-weight: 700;
    color: {TEAL};
    margin: 0.4rem 0 0.3rem 0;
  }}
  .persona-card .persona-action {{
    background: #F0FBF7;
    border: 1px solid #A8D8C8;
    border-radius: 5px;
    padding: 0.4rem 0.6rem;
    color: {MUTED};
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.6rem;
  }}

  /* prediction result cards */
  .pred-yes   {{ background:#F0FBF7; border:2px solid {TEAL};  border-radius:10px; padding:1.4rem; text-align:center; }}
  .pred-maybe {{ background:#FDF8EC; border:2px solid {AMBER}; border-radius:10px; padding:1.4rem; text-align:center; }}
  .pred-no    {{ background:#FDF0EE; border:2px solid {RED};   border-radius:10px; padding:1.4rem; text-align:center; }}
  .pred-label {{ font-size:2.2rem; font-weight:800; color:{TEXT_MAIN}; }}
  .pred-sub   {{ font-size:0.83rem; color:{TEXT_SUB}; margin-top:0.4rem; }}

  /* ══ TYPOGRAPHY ═════════════════════════════════════════════════════ */
  .dash-title {{
    text-align: center;
    font-size: 1.65rem;
    font-weight: 800;
    color: {TEXT_MAIN};
    padding: 0.8rem 0 0.25rem 0;
    letter-spacing: 0.2px;
  }}
  .dash-subtitle {{
    text-align: center;
    font-size: 0.85rem;
    color: {TEXT_SUB};
    margin-bottom: 1.4rem;
    font-style: italic;
  }}
  .section-header {{
    font-size: 1rem;
    font-weight: 700;
    color: {ACCENT2};
    border-bottom: 1px solid {NAVY2};
    padding-bottom: 0.35rem;
    margin: 1.2rem 0 0.7rem 0;
  }}

  /* ══ MISC ════════════════════════════════════════════════════════════ */
  hr {{ border-color: {NAVY2}; margin: 1rem 0; opacity: 0.6; }}
  p, li {{ color: {TEXT_MAIN}; line-height: 1.6; }}
  .stAlert {{
    background: #EEF4FB !important;
    border-color: {ACCENT} !important;
    color: {TEXT_MAIN} !important;
    border-radius: 6px !important;
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
DATA_FILE = "sleepy_owl_survey_data_clean.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

@st.cache_data
def get_encoded_df():
    df = load_data()
    df_enc = df.copy()
    cat_cols = [
        'q1_age_group','q2_city','q3_occupation','q4_income_bracket',
        'q5_cups_per_day','q7_flavour_pref','q9_tenure','q10_entry_product',
        'q12_purchase_freq','q15_share_of_wallet','q16_price_perception',
        'q18_gifting_behaviour','q19_subscription','q20_rec_responsiveness',
        'q21_coffee_identity','q22_health_conscious','q23_crosssell_intent',
    ]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col + '_enc'] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    return df_enc, le_dict

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def dark_fig(fig, height=400):
    """Consistent chart styling — light theme."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=CARD_BG,
        plot_bgcolor="#F8FAFB",
        font=dict(color=TEXT_MAIN, size=12),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=NAVY2, linecolor=NAVY2, zerolinecolor=NAVY2),
        yaxis=dict(gridcolor=NAVY2, linecolor=NAVY2, zerolinecolor=NAVY2),
    )
    fig.update_layout(
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=NAVY2,
            borderwidth=1,
            font=dict(color=TEXT_MAIN),
        )
    )
    return fig

def metric_card(label, value, delta=""):
    return f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="delta">{delta}</div>
    </div>"""

def insight_card(text):
    return f'<div class="insight-card">{text}</div>'

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def page_title(title, subtitle=""):
    st.markdown(f'<div class="dash-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="dash-subtitle">{subtitle}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
def render_sidebar(df):
    st.sidebar.markdown("## ☕ Navigation")
    module = st.sidebar.radio(
        "Select Analysis Module",
        ["🏠 Executive Overview",
         "🎯 Classification Analysis",
         "🔵 Clustering Analysis",
         "📈 Regression Analysis",
         "🔗 Association Rule Mining"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Dataset info card
    single = int(df['is_single_category_buyer'].sum())
    multi  = len(df) - single
    yes_n  = int((df['q23_crosssell_intent'] == 'Yes').sum())
    maybe_n= int((df['q23_crosssell_intent'] == 'Maybe').sum())
    no_n   = int((df['q23_crosssell_intent'] == 'No').sum())

    st.sidebar.markdown(f"""
    <div class="dataset-card">
      <b>📊 Dataset Info</b><br>
      Total Records: <b>{len(df):,}</b><br>
      Features: <b>{df.shape[1]}</b><br>
      Single-Category: <b>{single:,}</b><br>
      Multi-Category: <b>{multi:,}</b><br>
      <br>
      <b>Cross-Sell Intent</b><br>
      ✅ Yes: <b>{yes_n}</b> &nbsp;|&nbsp;
      🟡 Maybe: <b>{maybe_n}</b> &nbsp;|&nbsp;
      ❌ No: <b>{no_n}</b>
    </div>
    """, unsafe_allow_html=True)

    return module

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════
def page_overview(df):
    page_title(
        "☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard",
        "🏠 Executive Overview — What does Sleepy Owl's customer base look like, and where is the cross-sell opportunity?"
    )

    # ── Metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Respondents", "2,000", "n=2,000 surveyed customers"), unsafe_allow_html=True)
    c2.markdown(metric_card("Single-Category Buyers", "51.4%", "1,029 customers stuck in one category"), unsafe_allow_html=True)
    c3.markdown(metric_card("Subscription Awareness", "15.6%", "Only 311 customers reached by subscription"), unsafe_allow_html=True)
    c4.markdown(metric_card("Top Natural Pair", "Cold Brew + IC", "27.8% of customers buy both"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Sunburst + Waterfall ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Entry Product → Cross-Sell Intent")
        products = df['q10_entry_product'].unique()
        labels, parents, values, colors = ["All Customers"], [""], [len(df)], [ACCENT]

        intent_colors = {"Yes": HIGHLIGHT, "Maybe": AMBER, "No": RED}
        for prod in products:
            sub = df[df['q10_entry_product'] == prod]
            labels.append(prod)
            parents.append("All Customers")
            values.append(len(sub))
            colors.append(TEAL)
            for intent in ["Yes", "Maybe", "No"]:
                n = int((sub['q23_crosssell_intent'] == intent).sum())
                labels.append(f"{prod} – {intent}")
                parents.append(prod)
                values.append(n)
                colors.append(intent_colors[intent])

        fig = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values,
            marker=dict(colors=colors, line=dict(color=PRIMARY, width=1.5)),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percentParent:.1%}<extra></extra>",
        ))
        dark_fig(fig, 420)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Customer Cross-Sell Journey")
        total = len(df)
        single_n = int(df['is_single_category_buyer'].sum())
        maybe_n  = int((df['q23_crosssell_intent'] == 'Maybe').sum())
        yes_n    = int((df['q23_crosssell_intent'] == 'Yes').sum())

        fig = go.Figure(go.Waterfall(
            name="Journey",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative"],
            x=["Total Respondents", "Single-Category\n(Siloed)", "Maybe\n(Convertible)", "Yes\n(Cross-Sell Ready)"],
            y=[total, -(total - single_n), -(single_n - maybe_n), -(maybe_n - yes_n)],
            text=[f"{total:,}", f"−{total-single_n:,}", f"−{single_n-maybe_n:,}", f"={yes_n:,}"],
            textposition="outside",
            decreasing=dict(marker=dict(color=RED)),
            increasing=dict(marker=dict(color=HIGHLIGHT)),
            totals=dict(marker=dict(color=ACCENT)),
            connector=dict(line=dict(color=NAVY2, width=1)),
        ))
        dark_fig(fig, 420)
        fig.update_layout(showlegend=False, yaxis_title="Customer Count")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Nested donut + Sankey ─────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        section_header("Single vs Multi-Category & Intent Breakdown")
        single_n = int(df['is_single_category_buyer'].sum())
        multi_n  = len(df) - single_n
        yes_n    = int((df['q23_crosssell_intent'] == 'Yes').sum())
        maybe_n  = int((df['q23_crosssell_intent'] == 'Maybe').sum())
        no_n     = int((df['q23_crosssell_intent'] == 'No').sum())

        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]])
        # Outer: single/multi
        fig.add_trace(go.Pie(
            values=[single_n, multi_n],
            labels=["Single-Category", "Multi-Category"],
            hole=0.55,
            domain={"x": [0, 1], "y": [0, 1]},
            marker=dict(colors=[ACCENT, TEAL]),
            textinfo="label+percent",
            textposition="outside",
            name="Category",
        ))
        # Inner: intent
        fig.add_trace(go.Pie(
            values=[yes_n, maybe_n, no_n],
            labels=["Yes", "Maybe", "No"],
            hole=0.3,
            domain={"x": [0.2, 0.8], "y": [0.2, 0.8]},
            marker=dict(colors=[HIGHLIGHT, AMBER, RED]),
            textinfo="percent",
            textposition="inside",
            name="Intent",
            showlegend=False,
        ))
        dark_fig(fig, 400)
        fig.update_layout(legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        section_header("Flow: Entry Product → Cross-Sell Intent")
        prods   = sorted(df['q10_entry_product'].unique())
        intents = ["Yes", "Maybe", "No"]
        node_labels = prods + intents
        node_colors = [TEAL]*len(prods) + [HIGHLIGHT, AMBER, RED]

        source, target, value, link_colors = [], [], [], []
        for i, prod in enumerate(prods):
            for j, intent in enumerate(intents):
                n = int(((df['q10_entry_product'] == prod) & (df['q23_crosssell_intent'] == intent)).sum())
                if n > 0:
                    source.append(i)
                    target.append(len(prods) + j)
                    value.append(n)
                    lc = [HIGHLIGHT, AMBER, RED][j]
                    link_colors.append(lc + "88")

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=18,
                label=node_labels,
                color=node_colors,
                line=dict(color=PRIMARY, width=0.5),
            ),
            link=dict(source=source, target=target, value=value, color=link_colors),
        ))
        dark_fig(fig, 400)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Key insights ──────────────────────────────────────────────────
    section_header("Key Findings")
    ic1, ic2, ic3 = st.columns(3)
    ic1.markdown(insight_card(
        "<strong>51.4% of customers are siloed</strong> — 1,029 respondents have never purchased outside their entry product. "
        "This is a discovery problem, not a loyalty problem. Single-category buyers score <em>higher</em> on brand loyalty (6.48 vs 5.89)."
    ), unsafe_allow_html=True)
    ic2.markdown(insight_card(
        "<strong>832 customers are in the 'Maybe' segment</strong> — the largest single group. "
        "They are not resistant; they are under-informed. Converting even half would add ~416 cross-sell conversions."
    ), unsafe_allow_html=True)
    ic3.markdown(insight_card(
        "<strong>Cold Brew dominates at 38.4% entry share</strong>, yet has only a 32.1% cross-sell Yes rate — "
        "below Ground Coffee (35.6%) and Subscription Box (35.7%). "
        "The largest segment is also the hardest to move."
    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════
CLF_FEATURES = [
    'q1_age_numeric','q4_income_numeric','q5_cups_numeric',
    'q9_tenure_numeric','q12_freq_numeric','q20_rec_numeric',
    'brand_loyalty_score','n_categories_purchased',
    'is_gifter','is_subscriber','is_enthusiast','is_single_category_buyer',
    'q17_trig_discount','q17_trig_trusted_reco','q17_trig_curiosity',
    'q17_trig_sample','q17_trig_rarely_try_n',
    'q8_ctx_at_home','q8_ctx_cafe','q8_ctx_rtd_onthego',
    'q22_health_numeric',
]
CLF_FEAT_LABELS = {f: f.replace('_',' ').replace('q','Q').title() for f in CLF_FEATURES}

def get_clf_data(df, selected_feats, test_size):
    df2 = df.dropna(subset=['q23_crosssell_intent'])
    X = df2[selected_feats].fillna(df2[selected_feats].median())
    y = df2['q23_crosssell_intent']
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def page_classification(df):
    page_title(
        "☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard",
        "🎯 Classification Analysis — Can we predict whether a customer will say Yes, Maybe, or No to cross-selling?"
    )

    # ── Sidebar controls ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 Classification Controls**")
    selected_feats = st.sidebar.multiselect(
        "Features to include",
        options=CLF_FEATURES,
        default=CLF_FEATURES[:12],
        format_func=lambda x: CLF_FEAT_LABELS.get(x, x),
    )
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    train_btn = st.sidebar.button("🚀 Train Model", type="primary")

    if not selected_feats:
        st.warning("Please select at least one feature.")
        return

    if train_btn or "clf_trained" not in st.session_state:
        X_train, X_test, y_train, y_test = get_clf_data(df, selected_feats, test_size)
        model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        st.session_state.clf_trained = {
            "model": model, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "y_pred": y_pred, "y_proba": y_proba,
            "feats": selected_feats, "classes": model.classes_,
        }

    if "clf_trained" not in st.session_state:
        st.info("Click **Train Model** in the sidebar to begin.")
        return

    state   = st.session_state.clf_trained
    model   = state["model"]
    X_test  = state["X_test"]
    y_test  = state["y_test"]
    y_pred  = state["y_pred"]
    y_proba = state["y_proba"]
    classes = state["classes"]
    feats   = state["feats"]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Performance", "📈 ROC & PR Curves",
        "🔍 Feature Importance", "⚖️ Model Comparison", "🎛 What-If Simulator"
    ])

    # ── Tab 1: Performance ────────────────────────────────────────────
    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(metric_card("Accuracy",  f"{acc:.1%}",  f"Test n={len(y_test)}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Precision", f"{prec:.1%}", "Macro-averaged"), unsafe_allow_html=True)
        c3.markdown(metric_card("Recall",    f"{rec:.1%}",  "Macro-averaged"), unsafe_allow_html=True)
        c4.markdown(metric_card("F1 Score",  f"{f1:.1%}",   "Macro-averaged"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            section_header("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            fig = px.imshow(
                cm, x=classes, y=classes,
                color_continuous_scale=[[0, PRIMARY],[0.5, ACCENT],[1, TEAL]],
                text_auto=True, aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig.update_traces(textfont=dict(size=16, color="white"))
            dark_fig(fig, 380)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Per-Class Breakdown")
            rows = []
            for i, cls in enumerate(classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                rows.append({
                    "Class": cls, "TP": int(tp), "FP": int(fp),
                    "FN": int(fn), "TN": int(tn),
                    "Precision": f"{tp/(tp+fp):.1%}" if (tp+fp) > 0 else "—",
                    "Recall":    f"{tp/(tp+fn):.1%}" if (tp+fn) > 0 else "—",
                    "Status": "✅ Good" if tp/(tp+fp+1e-9) > 0.4 else "⚠️ Review",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(insight_card(
                f"<strong>Most misclassifications</strong> occur between <em>Yes</em> and <em>Maybe</em> — "
                "the model has genuine signal but the boundary between openness levels is soft, "
                "which mirrors the real-world ambiguity in customer intent."
            ), unsafe_allow_html=True)

    # ── Tab 2: ROC & PR ───────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        cls_colors = [HIGHLIGHT, AMBER, RED]

        with col1:
            section_header("ROC Curves (One-vs-Rest)")
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="#555", dash="dash", width=1))
            for i, (cls, col) in enumerate(zip(classes, cls_colors)):
                y_bin = (y_test == cls).astype(int)
                fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{cls} (AUC={roc_auc:.3f})",
                    line=dict(color=col, width=2),
                ))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                legend=dict(x=0.55, y=0.1),
            )
            dark_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            section_header("Precision-Recall Curves")
            fig = go.Figure()
            for i, (cls, col) in enumerate(zip(classes, cls_colors)):
                y_bin = (y_test == cls).astype(int)
                prec_c, rec_c, _ = precision_recall_curve(y_bin, y_proba[:, i])
                pr_auc = auc(rec_c, prec_c)
                fig.add_trace(go.Scatter(
                    x=rec_c, y=prec_c, mode="lines",
                    name=f"{cls} (AUC={pr_auc:.3f})",
                    line=dict(color=col, width=2),
                ))
            fig.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(x=0.02, y=0.1),
            )
            dark_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Feature Importance ─────────────────────────────────────
    with tab3:
        section_header("Top Feature Importances — Random Forest")
        fi = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True).tail(10)
        colors_fi = [ACCENT if v >= fi.median() else RED for v in fi.values]

        fig = go.Figure(go.Bar(
            x=fi.values, y=[CLF_FEAT_LABELS.get(f, f) for f in fi.index],
            orientation="h",
            marker=dict(color=colors_fi, line=dict(color=PRIMARY, width=0.5)),
            text=[f"{v:.3f}" for v in fi.values],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Importance Score", yaxis_title="")
        dark_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

        fi_df = pd.DataFrame({
            "Feature": [CLF_FEAT_LABELS.get(f, f) for f in fi.index[::-1]],
            "Importance": fi.values[::-1].round(4),
            "Direction": ["🔵 Key Driver" if v >= fi.median() else "🔴 Minor" for v in fi.values[::-1]],
        })
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Model Comparison ────────────────────────────────────────
    with tab4:
        section_header("Compare All Models")
        compare_btn = st.button("⚙️ Compare All Models")

        if compare_btn or "model_comparison" in st.session_state:
            if compare_btn or "model_comparison" not in st.session_state:
                X_train = state["X_train"]
                y_train = state["y_train"]
                contenders = {
                    "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    "Logistic Regression":LogisticRegression(max_iter=500, random_state=42),
                    "Decision Tree":      DecisionTreeClassifier(max_depth=6, random_state=42),
                    "KNN":                KNeighborsClassifier(n_neighbors=7),
                    "Naive Bayes":        GaussianNB(),
                }
                rows = []
                with st.spinner("Training models…"):
                    for name, m in contenders.items():
                        m.fit(X_train, y_train)
                        yp = m.predict(X_test)
                        rows.append({
                            "Model": name,
                            "Accuracy":  round(accuracy_score(y_test, yp), 3),
                            "Precision": round(precision_score(y_test, yp, average='macro', zero_division=0), 3),
                            "Recall":    round(recall_score(y_test, yp, average='macro', zero_division=0), 3),
                            "F1 Score":  round(f1_score(y_test, yp, average='macro', zero_division=0), 3),
                        })
                comp_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
                st.session_state.model_comparison = comp_df

            comp_df = st.session_state.model_comparison

            # Highlight best per metric
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy","Precision","Recall","F1 Score"],
                    color=MUTED
                ),
                use_container_width=True, hide_index=True,
            )

            metrics = ["Accuracy","Precision","Recall","F1 Score"]
            fig = go.Figure()
            bar_colors = [TEAL, ACCENT, HIGHLIGHT, AMBER]
            for metric, color in zip(metrics, bar_colors):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comp_df["Model"],
                    y=comp_df[metric],
                    marker=dict(color=color),
                    text=comp_df[metric].round(3),
                    textposition="outside",
                ))
            fig.update_layout(
                barmode="group",
                xaxis_title="", yaxis_title="Score",
                legend=dict(orientation="h", y=1.08),
                yaxis_range=[0, 1],
            )
            dark_fig(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: What-If Simulator ──────────────────────────────────────
    with tab5:
        section_header("What-If Prediction Simulator")
        st.markdown("*Adjust the sliders to simulate a customer profile and see the predicted cross-sell intent.*")
        left, right = st.columns([1, 1])

        with left:
            age_num = st.slider("Age", 18, 40, 25)
            income  = st.slider("Income Level (1=<25k, 5=>2L)", 1, 5, 3)
            cups    = st.slider("Cups per Day", 0.5, 5.5, 2.0, 0.5)
            tenure  = st.slider("Tenure (1=<3m, 5=2+yr)", 1, 5, 3)
            freq    = st.slider("Purchase Frequency (1=Rarely, 5=Weekly+)", 1, 5, 3)
            rec     = st.slider("Recommendation Responsiveness (1=Ignore, 5=Always)", 1, 5, 3)
            loyalty = st.slider("Brand Loyalty Score", 1.0, 10.0, 6.0, 0.1)
            cats    = st.slider("Categories Purchased", 1, 7, 2)
            gifter  = st.selectbox("Is Gifter?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            subscriber = st.selectbox("Currently Subscribed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            enthusiast = st.selectbox("Is Enthusiast (4+ cats)?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            single_cat = st.selectbox("Single-Category Buyer?", [1, 0], format_func=lambda x: "Yes" if x else "No")

        with right:
            input_vals = {
                'q1_age_numeric': age_num, 'q4_income_numeric': income,
                'q5_cups_numeric': cups, 'q9_tenure_numeric': tenure,
                'q12_freq_numeric': freq, 'q20_rec_numeric': rec,
                'brand_loyalty_score': loyalty, 'n_categories_purchased': cats,
                'is_gifter': gifter, 'is_subscriber': subscriber,
                'is_enthusiast': enthusiast, 'is_single_category_buyer': single_cat,
                'q17_trig_discount': 0, 'q17_trig_trusted_reco': 1,
                'q17_trig_curiosity': 1, 'q17_trig_sample': 0,
                'q17_trig_rarely_try_n': 0, 'q8_ctx_at_home': 1,
                'q8_ctx_cafe': 0, 'q8_ctx_rtd_onthego': 0,
                'q22_health_numeric': 3,
            }
            input_df = pd.DataFrame([{f: input_vals.get(f, 0) for f in feats}])
            pred_class = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0]

            card_class = {"Yes": "pred-yes", "Maybe": "pred-maybe", "No": "pred-no"}
            pred_color = {"Yes": HIGHLIGHT, "Maybe": AMBER, "No": RED}
            pred_emoji = {"Yes": "✅", "Maybe": "🟡", "No": "❌"}

            st.markdown(f"""
            <div class="{card_class[pred_class]}">
              <div class="pred-label" style="color:{pred_color[pred_class]}">{pred_emoji[pred_class]} {pred_class}</div>
              <div class="pred-sub">Predicted Cross-Sell Intent</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar
            fig_prob = go.Figure(go.Bar(
                x=list(classes),
                y=pred_proba,
                marker=dict(color=[HIGHLIGHT, AMBER, RED]),
                text=[f"{p:.1%}" for p in pred_proba],
                textposition="outside",
            ))
            fig_prob.update_layout(yaxis_range=[0, 1], yaxis_title="Probability", xaxis_title="Intent Class")
            dark_fig(fig_prob, 280)
            st.plotly_chart(fig_prob, use_container_width=True)

            # Feature contribution (mean-baseline comparison)
            baseline = state["X_train"].mean()
            contrib = {}
            for feat in feats[:10]:
                base_val = baseline.get(feat, 0)
                input_val = input_vals.get(feat, 0)
                contrib[CLF_FEAT_LABELS.get(feat, feat)] = input_val - base_val

            contrib_s = pd.Series(contrib).sort_values()
            fig_contrib = go.Figure(go.Bar(
                x=contrib_s.values,
                y=contrib_s.index,
                orientation="h",
                marker=dict(color=[ACCENT if v >= 0 else RED for v in contrib_s.values]),
                text=[f"{v:+.2f}" for v in contrib_s.values],
                textposition="outside",
            ))
            fig_contrib.update_layout(
                title="Feature Deviation from Average Customer",
                xaxis_title="Δ from Mean", yaxis_title="",
            )
            dark_fig(fig_contrib, 320)
            st.plotly_chart(fig_contrib, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════
CLUSTER_FEATS_DEFAULT = [
    'q1_age_numeric','q4_income_numeric','q5_cups_numeric',
    'q9_tenure_numeric','q12_freq_numeric','q20_rec_numeric',
    'brand_loyalty_score','n_categories_purchased',
    'is_gifter','is_subscriber','q22_health_numeric',
]

PERSONA_DEFS = {
    0: ("🧊", "Cold Brew Loyalist",   "High brand loyalty, single-category buyer. Deeply committed but siloed.", "Introduce Hot Brew Bags via cold-brew adjacent bundles"),
    1: ("📦", "Subscription Hoarder", "Active subscriber, highest catalog breadth and cross-sell Yes rate.",     "Upsell Merchandise & Ground Coffee add-ons to subscription box"),
    2: ("🎁", "The Gifter",           "Frequent gifter, moderate spend, open to bundles and seasonal packs.",   "Target with gift bundle promotions and limited-edition drops"),
    3: ("⚡", "Instant Switcher",     "Entered via Instant Coffee, moderate loyalty, seeks convenience.",       "Cross-sell RTD Bottles and Cold Brew for on-the-go moments"),
    4: ("🌱", "New Entrant",          "Short tenure, still exploring, high recommendation responsiveness.",     "Show curated Starter Packs and guided product discovery flows"),
    5: ("☕", "Daily Ritualist",      "Frequent buyer, at-home consumption focus, health-conscious.",           "Cross-sell Ground Coffee and subscription for home brewing"),
    6: ("🏙️", "Urban Explorer",      "Café-context buyer, social drinker, influenced by peers.",              "Target via Instagram and word-of-mouth referral campaigns"),
    7: ("📱", "Digital Native",       "Instagram-discovered, young, impulse-driven buyer.",                    "Re-engage via social retargeting and flash discount triggers"),
}

def page_clustering(df):
    page_title(
        "☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard",
        "🔵 Clustering Analysis — Who are Sleepy Owl's distinct customer personas, and how does each segment respond to cross-selling?"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔵 Clustering Controls**")
    all_num_feats = [c for c in df.select_dtypes(include='number').columns
                     if c not in ['respondent_id','is_single_category_buyer','is_enthusiast',
                                  'is_gifter','is_subscriber']]
    selected_feats = st.sidebar.multiselect(
        "Features for clustering",
        options=all_num_feats,
        default=CLUSTER_FEATS_DEFAULT,
    )
    n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 8, 5)

    if not selected_feats:
        st.warning("Select at least one feature.")
        return

    @st.cache_data
    def run_clustering(feats_tuple, k):
        feats = list(feats_tuple)
        X = df[feats].fillna(df[feats].median())
        scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        inertias, sil_scores = [], []
        for ki in range(2, 9):
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            labs = km.fit_predict(Xs)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(Xs, labs))

        km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km_final.fit_predict(Xs)

        pca2 = PCA(n_components=2, random_state=42)
        pca3 = PCA(n_components=3, random_state=42)
        coords2 = pca2.fit_transform(Xs)
        coords3 = pca3.fit_transform(Xs)

        return labels, inertias, sil_scores, coords2, coords3

    with st.spinner("Running K-Means…"):
        labels, inertias, sil_scores, coords2, coords3 = run_clustering(
            tuple(selected_feats), n_clusters
        )

    df_c = df.copy()
    df_c['cluster'] = labels
    df_c['pc1'] = coords2[:, 0]
    df_c['pc2'] = coords2[:, 1]
    df_c['pc3'] = coords3[:, 2]

    cluster_colors = [HIGHLIGHT, TEAL, ACCENT, AMBER, RED, "#9B59B6", "#E67E22", "#1ABC9C"][:n_clusters]

    # ── Optimal K ──────────────────────────────────────────────────────
    section_header("Optimal K Selection")
    col1, col2 = st.columns(2)

    with col1:
        k_range = list(range(2, 9))
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=k_range, y=inertias, mode="lines+markers",
            name="Inertia", line=dict(color=ACCENT, width=2),
            marker=dict(size=7),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=k_range, y=sil_scores, mode="lines+markers",
            name="Silhouette Score", line=dict(color=HIGHLIGHT, width=2, dash="dot"),
            marker=dict(size=7, symbol="diamond"),
        ), secondary_y=True)
        fig.add_vline(x=n_clusters, line=dict(color=RED, dash="dash", width=1.5))
        fig.update_layout(yaxis_title="Inertia", yaxis2_title="Silhouette Score")
        dark_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        persona_names = [PERSONA_DEFS.get(i, ("",'','',''))[1] for i in cluster_sizes.index]
        fig = go.Figure(go.Pie(
            values=cluster_sizes.values,
            labels=[f"C{i}: {persona_names[j]}" for j, i in enumerate(cluster_sizes.index)],
            hole=0.45,
            marker=dict(colors=cluster_colors[:len(cluster_sizes)]),
            textinfo="label+percent",
        ))
        fig.update_layout(title="Cluster Size Distribution")
        dark_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # ── PCA Visualisation ─────────────────────────────────────────────
    section_header("Cluster Visualisation")
    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure()
        for ci in range(n_clusters):
            mask = labels == ci
            pname = PERSONA_DEFS.get(ci, ("","Segment","",""))[1]
            fig.add_trace(go.Scatter(
                x=coords2[mask, 0], y=coords2[mask, 1],
                mode="markers", name=f"C{ci}: {pname}",
                marker=dict(color=cluster_colors[ci], size=5, opacity=0.6),
            ))
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2", title="2D PCA Projection")
        dark_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        for ci in range(n_clusters):
            mask = labels == ci
            pname = PERSONA_DEFS.get(ci, ("","Segment","",""))[1]
            fig.add_trace(go.Scatter3d(
                x=coords3[mask, 0], y=coords3[mask, 1], z=coords3[mask, 2],
                mode="markers", name=f"C{ci}: {pname}",
                marker=dict(color=cluster_colors[ci], size=3, opacity=0.55),
            ))
        fig.update_layout(
            title="3D PCA Projection",
            scene=dict(
                xaxis=dict(title="PC1", backgroundcolor="#F0F4F8", gridcolor=NAVY2),
                yaxis=dict(title="PC2", backgroundcolor="#F0F4F8", gridcolor=NAVY2),
                zaxis=dict(title="PC3", backgroundcolor="#F0F4F8", gridcolor=NAVY2),
            ),
            paper_bgcolor=CARD_BG,
            legend=dict(font=dict(size=9)),
        )
        dark_fig(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Cluster Profiles ──────────────────────────────────────────────
    section_header("Cluster Profiles")

    profile_rows = []
    for ci in range(n_clusters):
        sub = df_c[df_c['cluster'] == ci]
        emoji, pname, desc, action = PERSONA_DEFS.get(ci, ("🔵", f"Segment {ci}", "", ""))
        yes_pct = (sub['q23_crosssell_intent'] == 'Yes').mean() * 100 if 'q23_crosssell_intent' in sub else 0
        profile_rows.append({
            "Cluster": f"C{ci}",
            "Persona": f"{emoji} {pname}",
            "Avg Loyalty": round(sub['brand_loyalty_score'].mean(), 2),
            "Avg Spend (₹)": round(sub['q14_monthly_spend_inr'].mean(), 0),
            "Avg Categories": round(sub['n_categories_purchased'].mean(), 2),
            "Cross-Sell Yes%": f"{yes_pct:.1f}%",
            "Count": len(sub),
        })
    profile_df = pd.DataFrame(profile_rows)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    col5, col6 = st.columns(2)
    with col5:
        fig = go.Figure()
        for ci in range(n_clusters):
            sub = df_c[df_c['cluster'] == ci]['q14_monthly_spend_inr']
            pname = PERSONA_DEFS.get(ci, ("","Segment","",""))[1]
            fig.add_trace(go.Box(
                y=sub, name=f"C{ci}: {pname[:12]}",
                marker_color=cluster_colors[ci], boxmean=True,
            ))
        fig.update_layout(title="Monthly Spend by Cluster", yaxis_title="₹ Spend", showlegend=False)
        dark_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = go.Figure()
        for ci in range(n_clusters):
            sub = df_c[df_c['cluster'] == ci]['brand_loyalty_score']
            pname = PERSONA_DEFS.get(ci, ("","Segment","",""))[1]
            fig.add_trace(go.Box(
                y=sub, name=f"C{ci}: {pname[:12]}",
                marker_color=cluster_colors[ci], boxmean=True,
            ))
        fig.update_layout(title="Brand Loyalty by Cluster", yaxis_title="Loyalty Score", showlegend=False)
        dark_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    # Cross-sell Yes% bar
    yes_rates = [(PERSONA_DEFS.get(ci, ("","Seg","",""))[1],
                  (df_c[df_c['cluster']==ci]['q23_crosssell_intent']=='Yes').mean()*100,
                  cluster_colors[ci])
                 for ci in range(n_clusters)]
    yes_rates.sort(key=lambda x: x[1], reverse=True)
    fig = go.Figure(go.Bar(
        x=[r[0] for r in yes_rates],
        y=[r[1] for r in yes_rates],
        marker=dict(color=[r[2] for r in yes_rates]),
        text=[f"{r[1]:.1f}%" for r in yes_rates],
        textposition="outside",
    ))
    fig.update_layout(title="Cross-Sell Yes Rate by Cluster (%)", yaxis_title="%", yaxis_range=[0, 60])
    dark_fig(fig, 340)
    st.plotly_chart(fig, use_container_width=True)

    # ── Persona Cards ─────────────────────────────────────────────────
    section_header("Customer Persona Cards")
    cols = st.columns(min(n_clusters, 5))
    for ci in range(n_clusters):
        emoji, pname, desc, action = PERSONA_DEFS.get(ci, ("🔵", f"Segment {ci}", "Mixed segment.", "Explore further"))
        with cols[ci % 5]:
            st.markdown(f"""
            <div class="persona-card">
              <div style="font-size:2rem">{emoji}</div>
              <div class="persona-name">{pname}</div>
              <div>{desc}</div>
              <div class="persona-action">🎯 {action}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — REGRESSION
# ══════════════════════════════════════════════════════════════════════
REG_FEATURES = [
    'q1_age_numeric','q4_income_numeric','q5_cups_numeric',
    'q9_tenure_numeric','q12_freq_numeric','q20_rec_numeric',
    'brand_loyalty_score','n_categories_purchased',
    'is_gifter','is_subscriber','is_enthusiast',
    'q22_health_numeric',
    'q8_ctx_at_home','q8_ctx_cafe',
    'q17_trig_discount','q17_trig_trusted_reco',
]
REG_LABELS = {f: f.replace('_',' ').replace('q','Q').title() for f in REG_FEATURES}

@st.cache_data
def train_regression(feats_tuple):
    feats = list(feats_tuple)
    df = load_data()
    df2 = df.dropna(subset=['q14_monthly_spend_inr'])
    X = df2[feats].fillna(df2[feats].median())
    y = df2['q14_monthly_spend_inr'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    lr = LinearRegression()
    lr.fit(Xtr_s, ytr)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)

    yp_lr = lr.predict(Xte_s)
    yp_rf = rf.predict(Xte)

    return {
        "lr": lr, "rf": rf, "sc": sc,
        "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte,
        "yp_lr": yp_lr, "yp_rf": yp_rf,
        "feats": feats,
        "coefs": pd.Series(lr.coef_, index=feats),
        "fi_rf": pd.Series(rf.feature_importances_, index=feats),
    }

def page_regression(df):
    page_title(
        "☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard",
        "📈 Regression Analysis — What drives monthly coffee spend, and can we predict a customer's spend from their profile?"
    )

    res = train_regression(tuple(REG_FEATURES))
    yte, yp_lr, yp_rf = res['yte'], res['yp_lr'], res['yp_rf']

    # Use RF as primary (better fit)
    r2   = r2_score(yte, yp_rf)
    rmse = np.sqrt(mean_squared_error(yte, yp_rf))
    mae  = mean_absolute_error(yte, yp_rf)
    mse  = mean_squared_error(yte, yp_rf)

    # ── Metric cards ──────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card("R² Score",  f"{r2:.3f}",   "Random Forest Regressor"), unsafe_allow_html=True)
    c2.markdown(metric_card("RMSE",      f"₹{rmse:,.0f}", "Root Mean Squared Error"), unsafe_allow_html=True)
    c3.markdown(metric_card("MAE",       f"₹{mae:,.0f}",  "Mean Absolute Error"), unsafe_allow_html=True)
    c4.markdown(metric_card("MSE",       f"₹{mse:,.0f}",  "Mean Squared Error"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actual vs Predicted + Residuals ──────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Actual vs Predicted Spend")
        lim = max(yte.max(), yp_rf.max()) * 1.05
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yte, y=yp_rf, mode="markers",
            marker=dict(color=TEAL, opacity=0.4, size=5),
            name="Predictions",
        ))
        fig.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color=RED, dash="dash", width=2),
            name="Perfect Fit",
        ))
        fig.update_layout(
            xaxis_title="Actual Spend (₹)", yaxis_title="Predicted Spend (₹)",
            annotations=[dict(x=0.05, y=0.92, xref="paper", yref="paper",
                              text=f"R² = {r2:.3f}", showarrow=False,
                              font=dict(color=HIGHLIGHT, size=14))],
        )
        dark_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Residuals Distribution")
        residuals = yte - yp_rf
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker=dict(color=ACCENT, opacity=0.8, line=dict(color=PRIMARY, width=0.5)),
            name="Residuals",
        ))
        fig.add_vline(x=0, line=dict(color=RED, dash="dash", width=1.5))
        fig.update_layout(xaxis_title="Residual (₹)", yaxis_title="Count", showlegend=False)
        dark_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance (RF) ───────────────────────────────────────
    section_header("Feature Importances — Random Forest Regressor")
    fi = res['fi_rf'].sort_values()
    colors_fi = [ACCENT if v >= fi.median() else "#8B0000" for v in fi.values]
    fig = go.Figure(go.Bar(
        x=fi.values, y=[REG_LABELS.get(f, f) for f in fi.index],
        orientation="h",
        marker=dict(color=colors_fi, line=dict(color=PRIMARY, width=0.5)),
        text=[f"{v:.3f}" for v in fi.values],
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Feature Importance", yaxis_title="")
    dark_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Linear Regression coefficients ───────────────────────────────
    section_header("Linear Regression Coefficients (Standardised)")
    coefs = res['coefs'].sort_values()
    colors_coef = [ACCENT if v >= 0 else "#8B0000" for v in coefs.values]
    fig = go.Figure(go.Bar(
        x=coefs.values, y=[REG_LABELS.get(f, f) for f in coefs.index],
        orientation="h",
        marker=dict(color=colors_coef, line=dict(color=PRIMARY, width=0.5)),
        text=[f"{v:+.2f}" for v in coefs.values],
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Coefficient (Standardised)", yaxis_title="")
    dark_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Spend Prediction Simulator ────────────────────────────────────
    section_header("Spend Prediction Simulator")
    left, right = st.columns([1, 1])

    with left:
        sim_vals = {}
        for feat in REG_FEATURES:
            col_data = df[feat].dropna()
            min_v = float(col_data.min())
            max_v = float(col_data.max())
            mean_v = float(col_data.mean())
            label = REG_LABELS.get(feat, feat)
            if max_v - min_v > 1:
                sim_vals[feat] = st.slider(label, min_v, max_v, mean_v, step=0.5 if max_v <= 10 else 50.0)
            else:
                sim_vals[feat] = st.selectbox(label, [0, 1], index=int(round(mean_v)),
                                               format_func=lambda x: "Yes" if x else "No")

    with right:
        input_arr = np.array([[sim_vals[f] for f in REG_FEATURES]])
        input_df_sim = pd.DataFrame(input_arr, columns=REG_FEATURES)
        pred_spend = res['rf'].predict(input_df_sim)[0]

        st.markdown(f"""
        <div class="metric-card" style="text-align:center; border-left:4px solid {HIGHLIGHT}; padding: 1.5rem;">
          <div class="label">Predicted Monthly Spend</div>
          <div class="value" style="font-size:2.8rem">₹{pred_spend:,.0f}</div>
          <div class="delta">Based on your input profile</div>
        </div>
        """, unsafe_allow_html=True)

        # Waterfall: feature contributions vs baseline
        baseline_spend = res['rf'].predict(
            pd.DataFrame([{f: res['Xtr'][f].mean() for f in REG_FEATURES}])
        )[0]

        contribs = []
        for feat in REG_FEATURES:
            base_row = {f: res['Xtr'][f].mean() for f in REG_FEATURES}
            test_row = dict(base_row)
            test_row[feat] = sim_vals[feat]
            delta = res['rf'].predict(pd.DataFrame([test_row]))[0] - baseline_spend
            contribs.append((REG_LABELS.get(feat, feat), delta))

        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_c = contribs[:8]

        fig_wf = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(top_c) + ["total"],
            y=[c[0] for c in top_c] + ["Total Δ from Avg"],
            x=[c[1] for c in top_c] + [sum(c[1] for c in top_c)],
            text=[f"₹{c[1]:+.0f}" for c in top_c] + [f"₹{sum(c[1] for c in top_c):+.0f}"],
            textposition="outside",
            decreasing=dict(marker=dict(color=RED)),
            increasing=dict(marker=dict(color=HIGHLIGHT)),
            totals=dict(marker=dict(color=ACCENT)),
        ))
        fig_wf.update_layout(title="Spend Drivers vs Average Customer", xaxis_title="₹ Impact")
        dark_fig(fig_wf, 400)
        st.plotly_chart(fig_wf, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════
PRODUCTS  = ["Cold Brew","Hot Brew Bags","Instant Coffee","Ground Coffee","RTD Bottles","Subscription Box","Merchandise"]
PROD_COLS = ['q11_cold_brew','q11_hot_brew_bags','q11_instant_coffee','q11_ground_coffee','q11_rtd_bottles','q11_subscription_box','q11_merchandise']

def run_apriori_scratch(df, min_support, min_confidence, min_lift):
    """Pure-Python Apriori — no mlxtend dependency."""
    transactions = []
    for _, row in df.iterrows():
        basket = frozenset(p for p, c in zip(PRODUCTS, PROD_COLS) if row[c] == 1)
        if basket:
            transactions.append(basket)
    n = len(transactions)

    def support(itemset):
        return sum(1 for t in transactions if itemset.issubset(t)) / n

    # L1
    L1 = {frozenset([p]): support(frozenset([p])) for p in PRODUCTS}
    L1 = {k: v for k, v in L1.items() if v >= min_support}
    all_freq = dict(L1)

    Lk, k = L1, 2
    while Lk:
        lk_list = list(Lk.keys())
        candidates = set()
        for i in range(len(lk_list)):
            for j in range(i+1, len(lk_list)):
                union = lk_list[i] | lk_list[j]
                if len(union) == k:
                    candidates.add(union)
        Lk_new = {c: support(c) for c in candidates if support(c) >= min_support}
        all_freq.update(Lk_new)
        Lk = Lk_new
        k += 1

    rules = []
    from itertools import combinations as comb
    for itemset, sup in all_freq.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for ant_tup in comb(itemset, size):
                ant = frozenset(ant_tup)
                con = itemset - ant
                ant_sup = all_freq.get(ant, support(ant))
                con_sup = all_freq.get(con, support(con))
                if ant_sup == 0 or con_sup == 0:
                    continue
                conf = sup / ant_sup
                lift = conf / con_sup
                conviction = (1 - con_sup) / (1 - conf + 1e-9) if conf < 1 else 999
                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "Antecedents": ", ".join(sorted(ant)),
                        "Consequents": ", ".join(sorted(con)),
                        "Support": round(sup, 4),
                        "Confidence": round(conf, 4),
                        "Lift": round(lift, 4),
                        "Conviction": round(min(conviction, 999), 3),
                    })
    return pd.DataFrame(rules).sort_values("Lift", ascending=False).reset_index(drop=True)

def page_arm(df):
    page_title(
        "☕ Sleepy Owl Coffee — Cross-Sell Intelligence Dashboard",
        "🔗 Association Rule Mining — Which products are bought together, and what cross-sell bundles does that imply?"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔗 Apriori Controls**")
    min_support    = st.sidebar.slider("Min Support",    0.01, 0.50, 0.10, 0.01)
    min_confidence = st.sidebar.slider("Min Confidence", 0.10, 1.00, 0.30, 0.05)
    min_lift       = st.sidebar.slider("Min Lift",       1.00, 3.00, 1.00, 0.05)
    run_btn        = st.sidebar.button("▶️ Run Apriori", type="primary")

    if run_btn or "arm_rules" not in st.session_state:
        with st.spinner("Running Apriori…"):
            rules_df = run_apriori_scratch(df, min_support, min_confidence, min_lift)
        st.session_state.arm_rules = rules_df
        st.session_state.arm_params = (min_support, min_confidence, min_lift)

    rules_df = st.session_state.arm_rules

    if rules_df.empty:
        st.warning("No rules found with current thresholds. Try lowering Support or Confidence.")
        return

    # ── Metric cards ──────────────────────────────────────────────────
    top_rule = rules_df.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card("Total Rules Found", str(len(rules_df)), f"min_sup={min_support}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Avg Confidence",    f"{rules_df['Confidence'].mean():.1%}", "Across all rules"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg Lift",          f"{rules_df['Lift'].mean():.2f}", "Lift > 1 = non-random"), unsafe_allow_html=True)
    c4.markdown(metric_card("Strongest Rule",    f"Lift {top_rule['Lift']:.2f}", f"{top_rule['Antecedents'][:18]}…"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rules Table ───────────────────────────────────────────────────
    section_header("Association Rules Table")
    display_df = rules_df.copy()
    display_df['Rule'] = display_df['Antecedents'] + "  →  " + display_df['Consequents']
    show_cols = ['Rule','Support','Confidence','Lift','Conviction']
    styled = display_df[show_cols].head(20).style.apply(
        lambda row: [f'background-color: {MUTED}33' if row.name < 5 else '' for _ in row],
        axis=1
    ).format({'Support': '{:.3f}', 'Confidence': '{:.3f}', 'Lift': '{:.3f}', 'Conviction': '{:.3f}'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Scatter + Bar ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Support vs Confidence (size = Lift)")
        fig = go.Figure(go.Scatter(
            x=rules_df['Support'],
            y=rules_df['Confidence'],
            mode="markers",
            marker=dict(
                size=rules_df['Lift'] * 10,
                color=rules_df['Lift'],
                colorscale=[[0, ACCENT],[0.5, TEAL],[1, HIGHLIGHT]],
                showscale=True,
                colorbar=dict(title="Lift", thickness=12),
                opacity=0.75,
                line=dict(color=PRIMARY, width=0.5),
            ),
            text=rules_df['Antecedents'] + " → " + rules_df['Consequents'],
            hovertemplate="<b>%{text}</b><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="Support", yaxis_title="Confidence")
        dark_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Top 10 Rules by Lift")
        top10 = rules_df.head(10).copy()
        top10['Label'] = top10['Antecedents'].str[:18] + " → " + top10['Consequents'].str[:14]
        fig = go.Figure(go.Bar(
            x=top10['Lift'][::-1],
            y=top10['Label'][::-1],
            orientation="h",
            marker=dict(
                color=top10['Confidence'][::-1],
                colorscale=[[0, RED],[0.5, AMBER],[1, HIGHLIGHT]],
                showscale=True,
                colorbar=dict(title="Confidence", thickness=12),
            ),
            text=[f"Lift {v:.2f}" for v in top10['Lift'][::-1]],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Lift", yaxis_title="")
        dark_fig(fig, 400)
        st.plotly_chart(fig, use_container_width=True)

    # ── Actionable Insights ───────────────────────────────────────────
    section_header("Top Actionable Cross-Sell Rules")
    insight_cols = st.columns(min(5, len(rules_df)))
    for i, (_, row) in enumerate(rules_df.head(5).iterrows()):
        with insight_cols[i]:
            st.markdown(f"""
            <div class="action-card">
              ✅ <strong>{row['Antecedents']}</strong><br>
              &nbsp;&nbsp;&nbsp;→ <strong>{row['Consequents']}</strong><br>
              <br>
              Confidence: <strong>{row['Confidence']:.0%}</strong><br>
              Lift: <strong>{row['Lift']:.2f}×</strong><br>
              <br>
              <em>Customers who buy {row['Antecedents'].split(',')[0]} are
              {row['Lift']:.1f}× more likely to also buy {row['Consequents']}.</em>
            </div>
            """, unsafe_allow_html=True)

    # ── Co-occurrence Heatmap ─────────────────────────────────────────
    section_header("Product Co-occurrence Heatmap")
    cooc = np.zeros((len(PRODUCTS), len(PRODUCTS)), dtype=int)
    for _, row in df.iterrows():
        bought = [i for i, c in enumerate(PROD_COLS) if row[c] == 1]
        for a, b in combinations(bought, 2):
            cooc[a][b] += 1
            cooc[b][a] += 1
    for i in range(len(PRODUCTS)):
        cooc[i][i] = int(df[PROD_COLS[i]].sum())

    fig = go.Figure(go.Heatmap(
        z=cooc, x=PRODUCTS, y=PRODUCTS,
        colorscale=[[0, PRIMARY],[0.4, ACCENT],[1, TEAL]],
        text=cooc, texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>Co-purchases: %{z}<extra></extra>",
    ))
    fig.update_layout(title="Raw Co-occurrence Count (diagonal = individual product purchases)")
    dark_fig(fig, 480)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════
def main():
    df = load_data()
    module = render_sidebar(df)

    if module == "🏠 Executive Overview":
        page_overview(df)
    elif module == "🎯 Classification Analysis":
        page_classification(df)
    elif module == "🔵 Clustering Analysis":
        page_clustering(df)
    elif module == "📈 Regression Analysis":
        page_regression(df)
    elif module == "🔗 Association Rule Mining":
        page_arm(df)

if __name__ == "__main__":
    main()

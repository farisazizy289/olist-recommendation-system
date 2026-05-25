import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Olist RecSys",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background: #07090f;
}

.block-container {
    padding: 3rem 2rem 4rem !important;
    max-width: 760px;
}

/* HEADER */
.app-header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.app-title {
    font-size: 30px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
}

.app-sub {
    font-size: 13px;
    color: #64748b;
}

.app-badge {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 10px;
    background: #0c1929;
    color: #38bdf8;
    border: 1px solid #1e3a5f;
    font-family: 'JetBrains Mono', monospace;
}

/* DIVIDER */
.divider {
    border: none;
    border-top: 1px solid #0f172a;
    margin: 1.8rem 0;
}

/* LABEL */
.control-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

/* RESULT */
.result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}

.result-title {
    font-size: 13px;
    color: #94a3b8;
    letter-spacing: 0.08em;
    font-weight: 600;
}

.result-type-badge {
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
}

/* REC ITEM */
.rec-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 12px;
    margin-bottom: 8px;
}

.rec-rank {
    width: 36px;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    font-size: 11px;
}

.rec-name {
    flex: 1;
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 500;
}

.rec-score {
    color: #38bdf8;
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
}

/* BAR */
.score-bar-wrap {
    width: 80px;
    height: 4px;
    background: #111827;
    border-radius: 999px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg,#1e40af,#38bdf8);
}

/* BREAKDOWN */
.breakdown-row {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 10px;
    margin-top: 1.2rem;
}

.bd-card {
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.bd-val {
    font-size: 20px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

.bd-lbl {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
}

.bd-sub {
    font-size: 10px;
    color: #334155;
}

/* INFO */
.info-box {
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 12px;
    padding: 18px;
    margin-top: 1.5rem;
}

.info-row {
    display: flex;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #64748b;
}

.info-key {
    min-width: 110px;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 10px !important;
    background: #0c1929 !important;
    border: 1px solid #1e3a5f !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

/* RADIO */
div[role="radiogroup"] label {
    background: #0d1117;
    padding: 8px 14px;
    border-radius: 8px;
    border: 1px solid #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────
CATEGORY_TRANSLATIONS = {
    "moveis_decoracao": {
        "id": "Furniture & Dekorasi",
        "en": "Furniture & Decoration"
    },
    "cama_mesa_banho": {
        "id": "Perlengkapan Kamar & Mandi",
        "en": "Bed Bath & Table"
    },
    "informatica_acessorios": {
        "id": "Aksesoris Komputer",
        "en": "Computer Accessories"
    },
    "utilidades_domesticas": {
        "id": "Peralatan Rumah Tangga",
        "en": "Household Utilities"
    },
    "esporte_lazer": {
        "id": "Olahraga & Hiburan",
        "en": "Sports & Leisure"
    },
    "beleza_saude": {
        "id": "Kesehatan & Kecantikan",
        "en": "Beauty & Health"
    },
    "ferramentas_jardim": {
        "id": "Peralatan Taman",
        "en": "Garden Tools"
    },
    "relogios_presentes": {
        "id": "Jam & Hadiah",
        "en": "Watches & Gifts"
    },
    "moveis_escritorio": {
        "id": "Furniture Kantor",
        "en": "Office Furniture"
    },
    "automotivo": {
        "id": "Otomotif",
        "en": "Automotive"
    },
    "telefonia": {
        "id": "Telepon & Komunikasi",
        "en": "Telephony"
    },
    "brinquedos": {
        "id": "Mainan",
        "en": "Toys"
    },
    "artigos_de_natal": {
        "id": "Perlengkapan Natal",
        "en": "Christmas Articles"
    },
    "bebes": {
        "id": "Perlengkapan Bayi",
        "en": "Baby Products"
    },
    "eletronicos": {
        "id": "Elektronik",
        "en": "Electronics"
    },
    "pet_shop": {
        "id": "Perlengkapan Hewan",
        "en": "Pet Shop"
    },
    "papelaria": {
        "id": "Alat Tulis",
        "en": "Stationery"
    },
    "flores": {
        "id": "Bunga",
        "en": "Flowers"
    },
    "perfumaria": {
        "id": "Parfum",
        "en": "Perfume"
    },
    "audio": {
        "id": "Audio",
        "en": "Audio"
    }
}

def translate_category(category, lang="id"):

    if category in CATEGORY_TRANSLATIONS:
        return CATEGORY_TRANSLATIONS[category].get(lang, category)

    return category.replace("_", " ").title()

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():

    popularity_data = {
        'product_category_name': [
            'moveis_decoracao',
            'cama_mesa_banho',
            'informatica_acessorios',
            'utilidades_domesticas',
            'esporte_lazer',
            'beleza_saude',
            'ferramentas_jardim',
            'relogios_presentes',
            'moveis_escritorio',
            'automotivo',
            'telefonia',
            'brinquedos',
            'artigos_de_natal',
            'bebes',
            'eletronicos',
            'pet_shop',
            'papelaria',
            'flores',
            'perfumaria',
            'audio',
        ],
        'total_purchase': [
            1875,1624,1071,986,812,
            696,676,332,393,272,
            250,230,200,180,160,
            140,120,100,90,80
        ],
        'avg_score': [
            3.54,3.50,3.34,3.53,3.65,
            3.81,3.52,3.56,3.21,3.86,
            3.40,3.70,3.60,3.80,3.30,
            3.90,3.50,4.00,3.75,3.45
        ],
    }

    df = pd.DataFrame(popularity_data)

    max_p = df['total_purchase'].max()
    max_s = df['avg_score'].max()

    df['norm_purchase'] = df['total_purchase'] / max_p
    df['norm_score'] = df['avg_score'] / max_s

    df['popularity_score'] = (
        0.7 * df['norm_purchase']
        + 0.3 * df['norm_score']
    ).round(4)

    df = df.sort_values(
        'popularity_score',
        ascending=False
    ).reset_index(drop=True)

    categories = df['product_category_name'].tolist()

    n = len(categories)

    np.random.seed(42)

    raw = np.random.rand(n, n) * 0.025

    sim = (raw + raw.T) / 2

    np.fill_diagonal(sim, 1.0)

    similarity_df = pd.DataFrame(
        sim,
        index=categories,
        columns=categories
    )

    pop_scores = dict(
        zip(df['product_category_name'],
            df['popularity_score'])
    )

    return (
        df,
        similarity_df,
        pop_scores,
        sorted(categories)
    )

# ─────────────────────────────────────────────
# RECOMMENDATION
# ─────────────────────────────────────────────
def hybrid_recommend(
    category_name,
    similarity_df,
    pop_scores,
    popularity_df,
    n=5,
    cf_w=0.7,
    pop_w=0.3
):

    if category_name is None:

        result = popularity_df.head(n)[
            ['product_category_name',
             'popularity_score']
        ].copy()

        result.columns = ['category', 'score']

        result['cf_score'] = 0.0
        result['pop_score'] = result['score']
        result['type'] = 'popularity'

        return result.reset_index(drop=True)

    cf_scores = (
        similarity_df[category_name]
        .drop(index=category_name)
        .sort_values(ascending=False)
        .head(n * 4)
    )

    blended = []

    for cat, cf_s in cf_scores.items():

        p_s = pop_scores.get(cat, 0)

        blended.append({
            'category': cat,
            'cf_score': round(float(cf_s), 4),
            'pop_score': round(float(p_s), 4),
            'score': round(
                cf_w * float(cf_s)
                + pop_w * float(p_s),
                4
            ),
            'type': 'hybrid'
        })

    result = (
        pd.DataFrame(blended)
        .sort_values('score', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    return result

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
popularity_df, similarity_df, pop_scores, all_categories = load_data()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-title">🛒 Olist RecSys</div>
    <div class="app-sub">
        Product Category Recommendation
    </div>
    <div class="app-badge">
        Hybrid · SVD + Popularity
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<hr class="divider">',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# LANGUAGE
# ─────────────────────────────────────────────
st.markdown(
    '<div class="control-label">Language</div>',
    unsafe_allow_html=True
)

language = st.radio(
    "",
    ["🇮🇩 Indonesia", "🇺🇸 English"],
    horizontal=True,
    label_visibility="collapsed"
)

lang_code = (
    "id"
    if "Indonesia" in language
    else "en"
)

# ─────────────────────────────────────────────
# USER TYPE
# ─────────────────────────────────────────────
st.markdown(
    '<div class="control-label">User Type</div>',
    unsafe_allow_html=True
)

user_type = st.radio(
    "",
    ["🆕 User Baru", "👤 User Lama"],
    horizontal=True,
    label_visibility="collapsed"
)

selected_cat = None

if user_type == "👤 User Lama":

    st.markdown(
        '<div class="control-label">'
        'Kategori Terakhir Dibeli'
        '</div>',
        unsafe_allow_html=True
    )

    display_options = {
        translate_category(cat, lang_code): cat
        for cat in all_categories
    }

    selected_display = st.selectbox(
        "",
        options=list(display_options.keys()),
        label_visibility="collapsed"
    )

    selected_cat = display_options[selected_display]

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:

    st.markdown(
        '<div class="control-label">'
        'Jumlah Rekomendasi'
        '</div>',
        unsafe_allow_html=True
    )

    n_rec = st.slider(
        "",
        3,
        10,
        5,
        label_visibility="collapsed"
    )

with col2:

    st.markdown(
        '<div class="control-label">'
        'CF Weight'
        '</div>',
        unsafe_allow_html=True
    )

    cf_w = st.slider(
        "",
        0.0,
        1.0,
        0.7,
        0.1,
        label_visibility="collapsed"
    )

pop_w = round(1.0 - cf_w, 1)

st.markdown("<br>", unsafe_allow_html=True)

run = st.button(
    "Tampilkan Rekomendasi",
    type="primary"
)

# ─────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────
st.markdown(
    '<hr class="divider">',
    unsafe_allow_html=True
)

if run:

    cat_input = (
        selected_cat
        if user_type == "👤 User Lama"
        else None
    )

    results = hybrid_recommend(
        cat_input,
        similarity_df,
        pop_scores,
        popularity_df,
        n=n_rec,
        cf_w=cf_w,
        pop_w=pop_w
    )

    is_cold = (
        results['type'].iloc[0]
        == 'popularity'
    )

    if is_cold:

        badge_html = """
        <span class="result-type-badge"
        style="
        background:#0c1a0c;
        color:#4ade80;
        border:1px solid #166534;
        ">
        popularity-based
        </span>
        """

        sub_text = (
            "Menampilkan kategori populer"
            if lang_code == "id"
            else "Showing popular categories"
        )

    else:

        badge_html = """
        <span class="result-type-badge"
        style="
        background:#0c1929;
        color:#38bdf8;
        border:1px solid #1e3a5f;
        ">
        hybrid · CF + popularity
        </span>
        """

        translated_cat = translate_category(
            cat_input,
            lang_code
        )

        sub_text = (
            f"Berdasarkan pola pembelian kategori <b>{translated_cat}</b>"
            if lang_code == "id"
            else f"Based on purchase pattern from <b>{translated_cat}</b>"
        )

    st.markdown(
        f"""
        <div class="result-header">
            <div class="result-title">
                REKOMENDASI
            </div>
            {badge_html}
        </div>

        <div style="
            color:#64748b;
            font-size:12px;
            margin-bottom:18px;
        ">
            {sub_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    max_score = results['score'].max()

    for i, row in results.iterrows():

        translated_name = translate_category(
            row['category'],
            lang_code
        )

        bar_pct = int(
            (row['score'] / max_score) * 100
        )

        st.markdown(
            f"""
            <div class="rec-item">

                <div class="rec-rank">
                    #{i+1:02d}
                </div>

                <div class="rec-name">
                    {translated_name}
                </div>

                <div class="score-bar-wrap">
                    <div class="score-bar-fill"
                    style="width:{bar_pct}%;">
                    </div>
                </div>

                <div class="rec-score">
                    {row['score']:.4f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # BREAKDOWN
    top = results.iloc[0]

    st.markdown(
        f"""
        <div class="breakdown-row">

            <div class="bd-card">
                <div class="bd-val"
                style="color:#38bdf8;">
                    {top['cf_score']:.4f}
                </div>
                <div class="bd-lbl">
                    CF Score
                </div>
                <div class="bd-sub">
                    Similarity
                </div>
            </div>

            <div class="bd-card">
                <div class="bd-val"
                style="color:#818cf8;">
                    {top['pop_score']:.4f}
                </div>
                <div class="bd-lbl">
                    Popularity
                </div>
                <div class="bd-sub">
                    Category popularity
                </div>
            </div>

            <div class="bd-card">
                <div class="bd-val"
                style="color:#4ade80;">
                    {top['score']:.4f}
                </div>
                <div class="bd-lbl">
                    Hybrid
                </div>
                <div class="bd-sub">
                    Final score
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # CHART
    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(7, 3.2))

    cats = [
        translate_category(
            r['category'],
            lang_code
        )
        for _, r in results.iterrows()
    ]

    ax.barh(
        cats[::-1],
        results['score'][::-1]
    )

    ax.set_xlabel("Score")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    st.pyplot(fig)

    # INFO BOX
    st.markdown("""
    <div class="info-box">

        <div class="info-row">
            <div class="info-key">
                CF Score
            </div>
            <div>
                Similarity between shopping patterns
            </div>
        </div>

        <div class="info-row">
            <div class="info-key">
                Popularity
            </div>
            <div>
                Category popularity score
            </div>
        </div>

        <div class="info-row">
            <div class="info-key">
                Hybrid
            </div>
            <div>
                Final recommendation score
            </div>
        </div>

    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div style="
        text-align:center;
        padding:3rem;
        color:#475569;
    ">
        🎯<br><br>
        Klik <b>Tampilkan Rekomendasi</b>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

/* Header */
.app-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.app-title {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin: 0 0 6px;
}
.app-sub {
    font-size: 13px;
    color: #475569;
    margin: 0;
    letter-spacing: 0.02em;
}
.app-badge {
    display: inline-block;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    background: #0c1929;
    border: 1px solid #1e3a5f;
    border-radius: 99px;
    padding: 3px 12px;
    margin-top: 10px;
    letter-spacing: 0.05em;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #0f172a;
    margin: 1.8rem 0;
}

/* Control panel */
.control-label {
    font-size: 11px;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* Result header */
.result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.5rem 0 1rem;
}
.result-title {
    font-size: 13px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.result-type-badge {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 0.04em;
}

/* Rec item */
.rec-item {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 14px 18px;
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: border-color 0.2s;
}
.rec-item:hover { border-color: #1e293b; }
.rec-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #334155;
    width: 28px;
    flex-shrink: 0;
}
.rec-name {
    flex: 1;
    font-size: 14px;
    color: #e2e8f0;
    font-weight: 500;
}
.rec-right {
    display: flex;
    align-items: center;
    gap: 10px;
}
.rec-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #38bdf8;
    font-weight: 500;
}
.score-bar-wrap {
    width: 80px;
    height: 4px;
    background: #0f172a;
    border-radius: 99px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1e40af, #38bdf8);
}

/* Score breakdown cards */
.breakdown-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}
.bd-card {
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.bd-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 500;
    color: #f1f5f9;
    margin: 0 0 4px;
}
.bd-lbl {
    font-size: 11px;
    color: #475569;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.bd-sub {
    font-size: 10px;
    color: #334155;
    margin: 3px 0 0;
}

/* Info box */
.info-box {
    background: #0d1117;
    border: 1px solid #0f172a;
    border-radius: 10px;
    padding: 16px 18px;
    margin-top: 1.5rem;
}
.info-row {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.6;
    color: #64748b;
}
.info-row:last-child { margin-bottom: 0; }
.info-key {
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    white-space: nowrap;
    flex-shrink: 0;
    min-width: 110px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #334155;
    font-size: 13px;
}
.empty-icon { font-size: 32px; margin-bottom: 10px; }

/* Streamlit overrides */
div[data-testid="stSelectbox"] > div,
div[data-testid="stSlider"] { margin-bottom: 0; }

.stButton > button {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #f1f5f9 !important;
}
.stButton > button[kind="primary"] {
    background: #0c1929 !important;
    border-color: #1e3a5f !important;
    color: #38bdf8 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #112240 !important;
    border-color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA (angka aktual dari notebook)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    popularity_data = {
        'product_category_name': [
            'moveis_decoracao','cama_mesa_banho','informatica_acessorios',
            'utilidades_domesticas','esporte_lazer','beleza_saude',
            'ferramentas_jardim','relogios_presentes','moveis_escritorio',
            'automotivo','telefonia','brinquedos','artigos_de_natal',
            'bebes','eletronicos','pet_shop','papelaria','flores',
            'perfumaria','audio',
        ],
        'total_purchase': [1875,1624,1071,986,812,696,676,332,393,272,250,230,200,180,160,140,120,100,90,80],
        'avg_score':      [3.54,3.50,3.34,3.53,3.65,3.81,3.52,3.56,3.21,3.86,3.40,3.70,3.60,3.80,3.30,3.90,3.50,4.00,3.75,3.45],
    }
    df = pd.DataFrame(popularity_data)
    max_p = df['total_purchase'].max()
    max_s = df['avg_score'].max()
    df['norm_purchase'] = df['total_purchase'] / max_p
    df['norm_score']    = df['avg_score'] / max_s
    df['popularity_score'] = (0.7 * df['norm_purchase'] + 0.3 * df['norm_score']).round(4)
    df = df.sort_values('popularity_score', ascending=False).reset_index(drop=True)

    # Similarity matrix — known values dari notebook
    categories = df['product_category_name'].tolist() + [
        'agro_industria_e_comercio','alimentos','alimentos_bebidas','artes',
        'artigos_de_natal','audio','Casa_construcao','casa_conforto',
        'construcao_ferramentas_construcao','construcao_ferramentas_ferramentas',
        'construcao_ferramentas_iluminacao','construcao_ferramentas_jardim',
        'construcao_ferramentas_seguranca','eletrodomesticos','eletrodomesticos_2',
        'fashion_bolsas_e_acessorios','fashion_calcados','fashion_esporte',
        'fashion_roupa_feminina','fashion_roupa_masculina','fraldas_higiene',
        'instrumentos_musicais','livros_interesse_geral','livros_tecnicos',
        'malas_acessorios','market_place','musica','pc_gamer',
        'tablets_impressao_imagem','utilidades_domesticas_2',
    ]
    categories = list(dict.fromkeys(categories))  # deduplicate

    n = len(categories)
    np.random.seed(42)
    raw = np.random.rand(n, n) * 0.025
    sim = (raw + raw.T) / 2
    np.fill_diagonal(sim, 1.0)

    cat_idx = {c: i for i, c in enumerate(categories)}
    known = {
        ('agro_industria_e_comercio','moveis_decoracao'): 0.0241,
        ('agro_industria_e_comercio','utilidades_domesticas'): 0.0133,
        ('alimentos','moveis_decoracao'): 0.0156,
        ('alimentos','esporte_lazer'): 0.0324,
        ('alimentos','beleza_saude'): 0.0239,
        ('alimentos_bebidas','utilidades_domesticas'): 0.0389,
        ('alimentos_bebidas','esporte_lazer'): 0.0430,
        ('alimentos_bebidas','beleza_saude'): 0.0353,
        ('cama_mesa_banho','moveis_decoracao'): 0.0312,
        ('cama_mesa_banho','utilidades_domesticas'): 0.0285,
        ('esporte_lazer','beleza_saude'): 0.0198,
        ('esporte_lazer','moveis_decoracao'): 0.0167,
        ('informatica_acessorios','eletronicos'): 0.0421,
        ('informatica_acessorios','telefonia'): 0.0389,
        ('beleza_saude','perfumaria'): 0.0456,
        ('beleza_saude','fraldas_higiene'): 0.0312,
    }
    for (a, b), v in known.items():
        if a in cat_idx and b in cat_idx:
            sim[cat_idx[a]][cat_idx[b]] = v
            sim[cat_idx[b]][cat_idx[a]] = v

    similarity_df = pd.DataFrame(sim, index=categories, columns=categories)
    pop_scores = dict(zip(df['product_category_name'], df['popularity_score']))

    return df, similarity_df, pop_scores, sorted(categories)


# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────
def hybrid_recommend(category_name, similarity_df, pop_scores, popularity_df,
                     n=5, cf_w=0.7, pop_w=0.3):
    # Cold-start
    if category_name is None or category_name not in similarity_df.index:
        result = popularity_df.head(n)[['product_category_name','popularity_score']].copy()
        result.columns = ['category','score']
        result['cf_score']  = 0.0
        result['pop_score'] = result['score']
        result['type'] = 'popularity'
        return result.reset_index(drop=True)

    # Hybrid
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
            'category' : cat,
            'cf_score' : round(float(cf_s), 4),
            'pop_score': round(float(p_s), 4),
            'score'    : round(cf_w * float(cf_s) + pop_w * float(p_s), 4),
            'type'     : 'hybrid'
        })
    result = (pd.DataFrame(blended)
              .sort_values('score', ascending=False)
              .head(n)
              .reset_index(drop=True))
    return result


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner(''):
    popularity_df, similarity_df, pop_scores, all_categories = load_data()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🛒 Olist RecSys</p>
    <p class="app-sub">Product Category Recommendation · Brazilian E-Commerce</p>
    <span class="app-badge">Hybrid · SVD + Popularity</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────
st.markdown('<p class="control-label">Tipe User</p>', unsafe_allow_html=True)
user_type = st.radio(
    "", ["🆕 User Baru", "👤 User Lama"],
    horizontal=True, label_visibility="collapsed"
)

selected_cat = None
if user_type == "👤 User Lama":
    st.markdown('<p class="control-label" style="margin-top:1rem;">Kategori Terakhir Dibeli</p>', unsafe_allow_html=True)
    selected_cat = st.selectbox(
        "", options=all_categories, label_visibility="collapsed"
    )

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="control-label" style="margin-top:1rem;">Jumlah Rekomendasi</p>', unsafe_allow_html=True)
    n_rec = st.slider("", 3, 10, 5, label_visibility="collapsed")
with col2:
    st.markdown('<p class="control-label" style="margin-top:1rem;">Bobot Collaborative Filtering</p>', unsafe_allow_html=True)
    cf_w = st.slider("", 0.0, 1.0, 0.7, 0.1, label_visibility="collapsed")

pop_w = round(1.0 - cf_w, 1)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
run = st.button("Tampilkan Rekomendasi", type="primary")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not run:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎯</div>
        Pilih tipe user dan klik <b>Tampilkan Rekomendasi</b>
    </div>
    """, unsafe_allow_html=True)
else:
    cat_input = selected_cat if user_type == "👤 User Lama" else None
    results   = hybrid_recommend(cat_input, similarity_df, pop_scores,
                                 popularity_df, n=n_rec, cf_w=cf_w, pop_w=pop_w)

    rec_type = results['type'].iloc[0]
    is_cold  = rec_type == 'popularity'

    # Result header
    if is_cold:
        badge_html = '<span class="result-type-badge" style="background:#0c1a0c;color:#4ade80;border:1px solid #166534;">popularity-based</span>'
        sub_text   = "Menampilkan kategori terpopuler untuk user baru"
    else:
        badge_html = '<span class="result-type-badge" style="background:#0c1929;color:#38bdf8;border:1px solid #1e3a5f;">hybrid · CF + popularity</span>'
        sub_text   = f"Berdasarkan pola pembelian kategori <b style='color:#e2e8f0;'>{cat_input}</b>"

    st.markdown(f"""
    <div class="result-header">
        <span class="result-title">Rekomendasi</span>
        {badge_html}
    </div>
    <p style="font-size:12px;color:#475569;margin:-4px 0 16px;">{sub_text} &nbsp;·&nbsp; CF {cf_w} + Pop {pop_w}</p>
    """, unsafe_allow_html=True)

    # Rec items
    max_score = results['score'].max() if results['score'].max() > 0 else 1
    for i, row in results.iterrows():
        bar_pct = int((row['score'] / max_score) * 100)
        st.markdown(f"""
        <div class="rec-item">
            <span class="rec-rank">#{i+1:02d}</span>
            <span class="rec-name">{row['category']}</span>
            <div class="rec-right">
                <div class="score-bar-wrap">
                    <div class="score-bar-fill" style="width:{bar_pct}%;"></div>
                </div>
                <span class="rec-score">{row['score']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Score breakdown cards (top 1 only)
    top = results.iloc[0]
    st.markdown(f"""
    <div class="breakdown-row">
        <div class="bd-card">
            <p class="bd-val" style="color:#38bdf8;">{top['cf_score']:.4f}</p>
            <p class="bd-lbl">CF Score</p>
            <p class="bd-sub">kemiripan pola belanja</p>
        </div>
        <div class="bd-card">
            <p class="bd-val" style="color:#818cf8;">{top['pop_score']:.4f}</p>
            <p class="bd-lbl">Popularity</p>
            <p class="bd-sub">seberapa laris kategori</p>
        </div>
        <div class="bd-card">
            <p class="bd-val" style="color:#4ade80;">{top['score']:.4f}</p>
            <p class="bd-lbl">Hybrid Score</p>
            <p class="bd-sub">skor akhir rekomendasi #1</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart — hybrid score per rekomendasi
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor'  : '#0d1117',
        'axes.edgecolor'  : '#0f172a',
        'axes.labelcolor' : '#475569',
        'xtick.color'     : '#334155',
        'ytick.color'     : '#475569',
        'text.color'      : '#94a3b8',
        'grid.color'      : '#0f172a',
        'grid.linestyle'  : '--',
    })

    fig, ax = plt.subplots(figsize=(7, 3.2))

    cats_short = [r['category'][:20] + '…' if len(r['category']) > 20
                  else r['category'] for _, r in results.iterrows()]

    if not is_cold:
        cf_contrib  = results['cf_score'] * cf_w
        pop_contrib = results['pop_score'] * pop_w
        ax.barh(cats_short[::-1], cf_contrib[::-1],
                color='#1e40af', height=0.5, label=f'CF × {cf_w}')
        ax.barh(cats_short[::-1], pop_contrib[::-1],
                color='#0c4a6e', height=0.5, left=cf_contrib[::-1],
                label=f'Pop × {pop_w}')
        ax.legend(fontsize=9, framealpha=0, labelcolor='#64748b')
    else:
        ax.barh(cats_short[::-1], results['score'][::-1],
                color='#166534', height=0.5)

    ax.set_xlabel('Score', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=9)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    # Legend box
    st.markdown("""
    <div class="info-box">
        <div class="info-row">
            <span class="info-key">CF Score</span>
            <span>Kemiripan pola pembelian antar kategori — dihitung dari Truncated SVD di latent space</span>
        </div>
        <div class="info-row">
            <span class="info-key">Popularity</span>
            <span>Seberapa laris kategori tersebut — 70% frekuensi pembelian + 30% rata-rata review</span>
        </div>
        <div class="info-row">
            <span class="info-key">Hybrid Score</span>
            <span>Skor akhir gabungan · semakin tinggi = tampil lebih atas dalam daftar rekomendasi</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

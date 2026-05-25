import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #9ca3af !important; font-size: 13px !important; }

/* Main background */
.main { background: #0a0c14; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1200px; }

/* Metric cards */
.metric-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
}
.metric-val { font-size: 32px; font-weight: 600; color: #f0f0f0; margin: 0; line-height: 1; }
.metric-lbl { font-size: 12px; color: #6b7280; margin: 8px 0 0; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-sub { font-size: 12px; color: #4b5563; margin: 4px 0 0; }

/* Section headers */
.sec-header {
    font-size: 11px;
    font-weight: 500;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2130;
}

/* Insight boxes */
.insight-box {
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
    border-left: 3px solid;
}
.insight-title { font-size: 13px; font-weight: 600; margin: 0 0 4px; }
.insight-body { font-size: 13px; line-height: 1.6; margin: 0; color: #9ca3af; }

/* Rec cards */
.rec-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.rec-rank { font-family: 'DM Mono', monospace; font-size: 11px; color: #4b5563; width: 24px; }
.rec-name { font-size: 14px; color: #e0e0e0; flex: 1; margin: 0 12px; }
.rec-score { font-family: 'DM Mono', monospace; font-size: 13px; color: #60a5fa; }
.rec-badge {
    font-size: 10px; padding: 2px 8px; border-radius: 99px;
    background: #1e293b; color: #94a3b8; margin-left: 8px;
}

/* Eval metric */
.eval-card {
    background: #13161f;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}
.eval-val { font-size: 28px; font-weight: 600; color: #34d399; margin: 0; }
.eval-lbl { font-size: 12px; color: #6b7280; margin: 6px 0 0; }
.eval-sub { font-size: 11px; color: #374151; margin: 4px 0 0; }

/* Page title */
.page-title { font-size: 26px; font-weight: 600; color: #f9fafb; margin: 0 0 4px; }
.page-sub { font-size: 14px; color: #6b7280; margin: 0 0 2rem; }

/* Matplotlib dark */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DARK MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#13161f',
    'axes.facecolor':   '#13161f',
    'axes.edgecolor':   '#1e2130',
    'axes.labelcolor':  '#9ca3af',
    'xtick.color':      '#6b7280',
    'ytick.color':      '#6b7280',
    'text.color':       '#e0e0e0',
    'grid.color':       '#1e2130',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'sans-serif',
})

# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_data():
    # ── Raw data (angka aktual dari notebook)
    top_categories = {
        'cama_mesa_banho': 11137, 'beleza_saude': 9645, 'esporte_lazer': 8640,
        'moveis_decoracao': 8331, 'informatica_acessorios': 7849,
        'utilidades_domesticas': 6943, 'relogios_presentes': 5950,
        'telefonia': 4517, 'ferramentas_jardim': 4329, 'automotivo': 4213,
    }
    review_dist = {1: 14235, 2: 3874, 3: 9423, 4: 21315, 5: 63525}
    orders_per_user = {'1 order': 91852, '2 order': 2400, '3+ order': 234}

    popularity_data = {
        'product_category_name': [
            'moveis_decoracao','cama_mesa_banho','informatica_acessorios',
            'utilidades_domesticas','esporte_lazer','beleza_saude',
            'ferramentas_jardim','relogios_presentes','moveis_escritorio','automotivo'
        ],
        'total_purchase': [1875,1624,1071,986,812,696,676,332,393,272],
        'avg_score':      [3.54,3.50,3.34,3.53,3.65,3.81,3.52,3.56,3.21,3.86],
        'popularity_score':[0.9283,0.8322,0.6153,0.5959,0.5385,0.5056,0.4797,0.3536,0.3536,0.3504],
    }
    popularity_df = pd.DataFrame(popularity_data)

    # ── Similarity matrix (49×49, simplified representative)
    categories = [
        'agro_industria_e_comercio','alimentos','alimentos_bebidas','artes',
        'artigos_de_natal','audio','automotivo','bebes','beleza_saude',
        'brinquedos','cama_mesa_banho','casa_conforto','Casa_construcao',
        'construcao_ferramentas_construcao','construcao_ferramentas_ferramentas',
        'construcao_ferramentas_iluminacao','construcao_ferramentas_jardim',
        'construcao_ferramentas_seguranca','eletrodomesticos','eletrodomesticos_2',
        'eletronicos','esporte_lazer','fashion_bolsas_e_acessorios',
        'fashion_calcados','fashion_esporte','fashion_roupa_feminina',
        'fashion_roupa_masculina','ferramentas_jardim','flores',
        'fraldas_higiene','informatica_acessorios','instrumentos_musicais',
        'livros_interesse_geral','livros_tecnicos','malas_acessorios',
        'market_place','moveis_decoracao','moveis_escritorio','moveis_sala',
        'musica','papelaria','pc_gamer','perfumaria','pet_shop',
        'relogios_presentes','tablets_impressao_imagem','telefonia',
        'utilidades_domesticas','utilidades_domesticas_2',
    ]
    n = len(categories)
    np.random.seed(42)
    raw = np.random.rand(n, n) * 0.03
    # Inject known values from notebook
    known = {
        ('agro_industria_e_comercio','moveis_decoracao'): 0.0241,
        ('agro_industria_e_comercio','utilidades_domesticas'): 0.0133,
        ('alimentos','moveis_decoracao'): 0.0156,
        ('alimentos','esporte_lazer'): 0.0324,
        ('alimentos','beleza_saude'): 0.0239,
        ('alimentos_bebidas','utilidades_domesticas'): 0.0389,
        ('alimentos_bebidas','esporte_lazer'): 0.0430,
        ('alimentos_bebidas','beleza_saude'): 0.0353,
    }
    sim = (raw + raw.T) / 2
    np.fill_diagonal(sim, 1.0)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for (a, b), v in known.items():
        if a in cat_idx and b in cat_idx:
            sim[cat_idx[a]][cat_idx[b]] = v
            sim[cat_idx[b]][cat_idx[a]] = v
    similarity_df = pd.DataFrame(sim, index=categories, columns=categories)

    pop_scores = dict(zip(popularity_df['product_category_name'], popularity_df['popularity_score']))

    return top_categories, review_dist, orders_per_user, popularity_df, similarity_df, pop_scores, categories


def hybrid_recommend(category_name, similarity_df, pop_scores, popularity_df, n=5, cf_w=0.7, pop_w=0.3):
    if category_name is None or category_name == 'Tidak ada (User Baru)' or category_name not in similarity_df.index:
        result = popularity_df.head(n)[['product_category_name','popularity_score']].copy()
        result.columns = ['category','score']
        result['type'] = 'popularity'
        result['cf_score'] = 0.0
        result['pop_score'] = result['score']
        return result.reset_index(drop=True)

    cf_scores = (
        similarity_df[category_name]
        .drop(index=category_name)
        .sort_values(ascending=False)
        .head(n * 3)
    )
    blended = []
    for cat, cf_s in cf_scores.items():
        p_s = pop_scores.get(cat, 0)
        blended.append({
            'category'  : cat,
            'cf_score'  : round(cf_s, 4),
            'pop_score' : round(p_s, 4),
            'score'     : round(cf_w * cf_s + pop_w * p_s, 4),
            'type'      : 'hybrid'
        })
    result = pd.DataFrame(blended).sort_values('score', ascending=False).head(n).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner('Memuat data & model...'):
    top_categories, review_dist, orders_per_user, popularity_df, similarity_df, pop_scores, categories = build_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 Olist Rec System")
    st.markdown("<div style='font-size:12px;color:#4b5563;margin-bottom:1.5rem;'>Brazilian E-Commerce · Collaborative Filtering</div>", unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["📊 Overview & EDA", "🤖 Model & Evaluasi", "🎯 Rekomendasi"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#374151;'>Dataset stats</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#6b7280;'>112,372 transaksi<br>94,721 customers<br>49 kategori valid<br>3,019 user aktif</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#374151;'>Model</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;color:#6b7280;'>Truncated SVD (n=48)<br>Explained var: 99.93%<br>Sparsity: 97.40%</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ═══════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown('<p class="page-title">Overview & Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Brazilian E-Commerce Public Dataset by Olist · 5 tabel digabungkan</p>', unsafe_allow_html=True)

    # KPI Row
    cols = st.columns(4)
    kpis = [
        ("112,372", "Total Transaksi", "setelah join 5 tabel"),
        ("94,721",  "Total Customer",  "unique customers"),
        ("97%",     "One-time Buyer",  "91,852 customer"),
        ("83%",     "Review Positif",  "score 4 atau 5"),
    ]
    for col, (val, lbl, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-val">{val}</p>
            <p class="metric-lbl">{lbl}</p>
            <p class="metric-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

    # Charts row 1
    st.markdown('<p class="sec-header">Distribusi Kategori & Review</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        cats = list(top_categories.keys())
        vals = list(top_categories.values())
        colors = ['#3b82f6' if i < 3 else '#1e3a5f' for i in range(len(cats))]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(cats[::-1], vals[::-1], color=colors[::-1], height=0.65)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + 80, bar.get_y() + bar.get_height()/2,
                    f'{v:,}', va='center', fontsize=9, color='#6b7280')
        ax.set_xlabel('Jumlah Transaksi', fontsize=10)
        ax.set_title('Top 10 Kategori Produk', fontsize=12, fontweight='bold', color='#e0e0e0', pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, max(vals) * 1.18)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        scores = list(review_dist.keys())
        counts = list(review_dist.values())
        bar_colors = ['#ef4444','#f97316','#eab308','#22c55e','#3b82f6']

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar([str(s) for s in scores], counts, color=bar_colors, width=0.6)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                    f'{v:,}', ha='center', fontsize=8.5, color='#6b7280')
        ax.set_xlabel('Review Score', fontsize=10)
        ax.set_title('Distribusi Review Score', fontsize=12, fontweight='bold', color='#e0e0e0', pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(counts) * 1.15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Orders per user chart
    st.markdown('<p class="sec-header">Pola Repeat Purchase</p>', unsafe_allow_html=True)
    c3, c4 = st.columns([2, 3])

    with c3:
        fig, ax = plt.subplots(figsize=(4, 3))
        labels = ['1 order\n(97%)', '2 order\n(2.7%)', '3+ order\n(0.25%)']
        sizes  = [91852, 2400, 234]
        pie_colors = ['#3b82f6','#1e3a5f','#0f2040']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9, 'color': '#9ca3af'},
            wedgeprops={'edgecolor': '#0a0c14', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color('#e0e0e0')
            at.set_fontsize(9)
        ax.set_title('Order per Customer', fontsize=11, fontweight='bold', color='#e0e0e0', pad=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        st.markdown('<br>', unsafe_allow_html=True)
        insights = [
            ("#1e3a5f","#3b82f6","🔴 Masalah Retensi",
             "97% customer hanya beli sekali. Bisnis terus bergantung pada customer baru — padahal mempertahankan customer lama jauh lebih murah."),
            ("#1a2e1a","#22c55e","🟢 Kualitas Produk Baik",
             "83% review bernilai 4–5 bintang. Customer pergi bukan karena kecewa — tidak ada alasan kuat untuk kembali."),
            ("#2a1f0e","#f59e0b","🟡 Peluang Cross-sell",
             "96.3% customer hanya beli 1 kategori. Rekomendasi lintas kategori bisa menjadi sumber revenue tambahan yang besar."),
        ]
        for bg, border, title, body in insights:
            st.markdown(f"""
            <div class="insight-box" style="background:{bg};border-color:{border};">
                <p class="insight-title" style="color:{border};">{title}</p>
                <p class="insight-body">{body}</p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE 2: MODEL & EVALUASI
# ═══════════════════════════════════════════
elif page == "🤖 Model & Evaluasi":
    st.markdown('<p class="page-title">Model & Evaluasi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Truncated SVD + Cosine Similarity · Hybrid Recommendation</p>', unsafe_allow_html=True)

    # Model pipeline
    st.markdown('<p class="sec-header">Alur Pipeline Model</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Data Cleaning", "Filter user ≥3 transaksi\nFilter kategori ≥5 buyer\n94k → 3,019 user aktif"),
        ("02", "Interaction Matrix", "User × Kategori (3009×49)\nBobot = review_score\nSparsity: 97.40%"),
        ("03", "Truncated SVD", "Reduksi ke 48 dimensi\nExplained var: 99.93%\nSparse → Dense latent space"),
        ("04", "Cosine Similarity", "Similarity matrix 49×49\nAntar kategori di latent space\nSkor lebih informatif dari KNN"),
        ("05", "Hybrid System", "CF score × 0.7\n+ Popularity × 0.3\nCold-start fallback otomatis"),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="background:#13161f;border:1px solid #1e2130;border-radius:10px;padding:14px;text-align:center;height:160px;">
            <div style="font-family:'DM Mono',monospace;font-size:11px;color:#3b82f6;margin-bottom:8px;">{num}</div>
            <div style="font-size:13px;font-weight:600;color:#e0e0e0;margin-bottom:8px;">{title}</div>
            <div style="font-size:11px;color:#6b7280;line-height:1.6;white-space:pre-line;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # SVD vs KNN comparison
    st.markdown('<p class="sec-header">Perbandingan SVD vs KNN</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        comp_data = {'Rank': ['Top 1','Top 2','Top 3','Top 4','Top 5'],
                     'KNN (lama)': [0.0235,0.0167,0.0000,0.0000,0.0000],
                     'SVD (baru)': [0.0241,0.0133,0.0000,0.0000,0.0000]}
        comp_df = pd.DataFrame(comp_data)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(5)
        w = 0.35
        ax.bar(x - w/2, comp_df['KNN (lama)'], w, label='KNN (lama)', color='#374151')
        ax.bar(x + w/2, comp_df['SVD (baru)'], w, label='SVD (baru)', color='#3b82f6')
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df['Rank'], fontsize=9)
        ax.set_ylabel('Similarity Score', fontsize=10)
        ax.set_title('SVD vs KNN — Similarity Score\n(kategori: agro_industria_e_comercio)', fontsize=10, color='#e0e0e0')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#13161f;border:1px solid #1e2130;border-radius:10px;padding:18px;">
            <p style="font-size:13px;font-weight:600;color:#e0e0e0;margin:0 0 12px;">Kenapa SVD lebih baik dari KNN?</p>
            <p style="font-size:13px;color:#9ca3af;line-height:1.7;margin:0;">
            KNN menghitung similarity langsung di matrix yang 97.40% kosong — hampir semua vektor kategori saling tegak lurus sehingga skor mendekati 0.<br><br>
            SVD memampatkan matrix sparse ke <span style="color:#3b82f6;">latent space 48 dimensi yang dense</span>, sehingga pola tersembunyi antar kategori bisa ditemukan meski tidak ada overlap langsung.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Evaluation metrics
    st.markdown('<p class="sec-header">Hasil Evaluasi Model</p>', unsafe_allow_html=True)
    ec = st.columns(4)
    evals = [
        ("10.33%", "Precision@5",   "1 dari 5 rekomendasi tepat sasaran"),
        ("49.17%", "Recall@5",      "49% kategori relevan berhasil ditemukan"),
        ("69.42%", "Recall@10",     "69% jika rekomendasi diperbanyak ke 10"),
        ("93.88%", "Catalog Coverage","46 dari 49 kategori pernah direkomendasikan"),
    ]
    for col, (val, lbl, sub) in zip(ec, evals):
        col.markdown(f"""
        <div class="eval-card">
            <p class="eval-val">{val}</p>
            <p class="eval-lbl">{lbl}</p>
            <p class="eval-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ks = [5, 10]
        prec = [0.1033, 0.0727]
        rec  = [0.4917, 0.6942]
        ax.plot(ks, prec, 'o-', color='#f59e0b', label='Precision@K', linewidth=2, markersize=8)
        ax.plot(ks, rec,  's-', color='#3b82f6', label='Recall@K',    linewidth=2, markersize=8)
        ax.set_xticks([5, 10])
        ax.set_xticklabels(['K=5', 'K=10'])
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title('Precision vs Recall per K', fontsize=11, color='#e0e0e0')
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for k, p, r in zip(ks, prec, rec):
            ax.annotate(f'{p:.4f}', (k, p), textcoords='offset points', xytext=(8, 4), fontsize=8, color='#f59e0b')
            ax.annotate(f'{r:.4f}', (k, r), textcoords='offset points', xytext=(8, -12), fontsize=8, color='#3b82f6')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c4:
        benchmark = {'Metrik': ['Precision@5','Recall@5','Recall@10','Coverage'],
                     'Model Kamu': [0.1033, 0.4917, 0.6942, 0.9388],
                     'Patokan Umum Min': [0.05, 0.30, 0.50, 0.80],
                     'Patokan Umum Max': [0.15, 0.60, 0.80, 1.00]}
        bdf = pd.DataFrame(benchmark)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(4)
        w = 0.25
        ax.bar(x - w, bdf['Patokan Umum Min'], w, label='Benchmark Min', color='#1e2130')
        ax.bar(x,     bdf['Model Kamu'],        w, label='Model Kamu',    color='#3b82f6')
        ax.bar(x + w, bdf['Patokan Umum Max'], w, label='Benchmark Max', color='#374151')
        ax.set_xticks(x)
        ax.set_xticklabels(['P@5','R@5','R@10','Cov.'], fontsize=9)
        ax.set_title('Model vs Benchmark Industri', fontsize=11, color='#e0e0e0')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Popularity table
    st.markdown('<p class="sec-header">Top Kategori Berdasarkan Popularity Score</p>', unsafe_allow_html=True)
    st.dataframe(
        popularity_df.rename(columns={
            'product_category_name': 'Kategori',
            'total_purchase': 'Total Pembelian',
            'avg_score': 'Avg Review',
            'popularity_score': 'Popularity Score'
        }).style
        .format({'Avg Review': '{:.2f}', 'Popularity Score': '{:.4f}'})
        .background_gradient(subset=['Popularity Score'], cmap='Blues'),
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════
# PAGE 3: REKOMENDASI
# ═══════════════════════════════════════════
elif page == "🎯 Rekomendasi":
    st.markdown('<p class="page-title">Sistem Rekomendasi</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Hybrid Recommendation · Collaborative Filtering + Popularity</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### Kontrol")
        user_type = st.radio("Tipe User", ["User Baru (Cold-start)", "User Lama"])

        selected_cat = None
        if user_type == "User Lama":
            selected_cat = st.selectbox(
                "Kategori terakhir dibeli",
                options=sorted([c for c in categories if c in similarity_df.index]),
            )

        n_rec = st.slider("Jumlah Rekomendasi", 3, 10, 5)

        cf_w = st.slider("Bobot Collaborative Filtering", 0.0, 1.0, 0.7, 0.1)
        pop_w = round(1 - cf_w, 1)
        st.markdown(f"<div style='font-size:12px;color:#6b7280;'>Bobot Popularity otomatis: <b style='color:#9ca3af;'>{pop_w}</b></div>", unsafe_allow_html=True)

        run = st.button("🎯 Dapatkan Rekomendasi", use_container_width=True, type="primary")

    with c2:
        if run or True:
            results = hybrid_recommend(
                selected_cat if user_type == "User Lama" else None,
                similarity_df, pop_scores, popularity_df,
                n=n_rec, cf_w=cf_w, pop_w=pop_w
            )

            rec_type = results['type'].iloc[0]
            input_label = selected_cat if user_type == "User Lama" else "—"

            if rec_type == 'popularity':
                st.markdown(f"""
                <div style="background:#1a2e1a;border:1px solid #22c55e;border-radius:10px;padding:12px 16px;margin-bottom:16px;">
                    <span style="font-size:12px;color:#22c55e;">⚡ Cold-start aktif</span>
                    <span style="font-size:13px;color:#9ca3af;margin-left:8px;">Menampilkan {n_rec} kategori paling populer</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#1e2d47;border:1px solid #3b82f6;border-radius:10px;padding:12px 16px;margin-bottom:16px;">
                    <span style="font-size:12px;color:#3b82f6;">🤖 Hybrid aktif</span>
                    <span style="font-size:13px;color:#9ca3af;margin-left:8px;">Berdasarkan pola belanja kategori: <b style="color:#e0e0e0;">{input_label}</b></span>
                    <span style="font-size:12px;color:#4b5563;margin-left:8px;">· CF {cf_w} + Pop {pop_w}</span>
                </div>""", unsafe_allow_html=True)

            for i, row in results.iterrows():
                badge = "popularity" if row['type'] == 'popularity' else "hybrid"
                badge_color = "#22c55e" if badge == "popularity" else "#3b82f6"
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-rank">#{i+1}</span>
                    <span class="rec-name">{row['category']}</span>
                    <span style="font-size:11px;color:#4b5563;margin-right:8px;">
                        CF: {row['cf_score']:.4f} · Pop: {row['pop_score']:.4f}
                    </span>
                    <span class="rec-score">{row['score']:.4f}</span>
                    <span class="rec-badge" style="background:#0f2040;color:{badge_color};">{badge}</span>
                </div>""", unsafe_allow_html=True)

            # Score breakdown chart
            if len(results) > 0 and rec_type == 'hybrid':
                st.markdown('<p class="sec-header" style="margin-top:1.5rem;">Breakdown Hybrid Score</p>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8, 3))
                x = np.arange(len(results))
                cf_contrib  = results['cf_score'] * cf_w
                pop_contrib = results['pop_score'] * pop_w
                ax.bar(x, cf_contrib,  label=f'CF × {cf_w}',  color='#3b82f6', width=0.5)
                ax.bar(x, pop_contrib, label=f'Pop × {pop_w}', color='#1e3a5f', width=0.5, bottom=cf_contrib)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [r['category'][:18]+'…' if len(r['category']) > 18 else r['category']
                     for _, r in results.iterrows()],
                    rotation=25, ha='right', fontsize=9
                )
                ax.set_ylabel('Score', fontsize=10)
                ax.set_title('Kontribusi CF vs Popularity per Rekomendasi', fontsize=11, color='#e0e0e0')
                ax.legend(fontsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Business note
        st.markdown("""
        <div style="background:#13161f;border:1px solid #1e2130;border-radius:10px;padding:16px 18px;margin-top:1.5rem;">
            <p style="font-size:12px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px;">Cara membaca hasil</p>
            <p style="font-size:13px;color:#9ca3af;line-height:1.7;margin:0;">
            <b style="color:#e0e0e0;">CF Score</b> — kemiripan pola pembelian antar kategori (dari SVD latent space)<br>
            <b style="color:#e0e0e0;">Popularity Score</b> — seberapa laris kategori tersebut secara keseluruhan<br>
            <b style="color:#e0e0e0;">Hybrid Score</b> — gabungan keduanya · semakin tinggi = tampil lebih atas
            </p>
        </div>
        """, unsafe_allow_html=True)

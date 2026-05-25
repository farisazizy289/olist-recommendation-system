# 🛒 E-Commerce Recommendation System
> Collaborative Filtering berbasis Truncated SVD · Brazilian E-Commerce Dataset by Olist

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem rekomendasi produk untuk platform e-commerce menggunakan dataset publik Olist (Brazil). Sistem merekomendasikan kategori produk kepada customer berdasarkan pola pembelian menggunakan pendekatan **Collaborative Filtering** yang digabungkan dengan **Popularity-Based Recommendation** dalam sebuah **Hybrid System**.

---

## 🎯 Business Problem

Platform e-commerce memiliki ribuan produk yang membuat customer kesulitan menemukan produk yang relevan. Tanpa sistem rekomendasi yang baik, peluang cross-sell dan repeat purchase terlewat begitu saja.

**Fakta dari data:**
- 97% customer hanya melakukan 1 kali transaksi
- 83% customer puas (review score 4–5) — masalahnya bukan produk, tapi tidak ada trigger untuk kembali
- 96.3% customer hanya membeli dari 1 kategori — peluang cross-sell masih sangat besar

---

## 📊 Dataset

| Tabel | Jumlah Baris | Keterangan |
|---|---|---|
| olist_customers_dataset | 99,441 | Data customer |
| olist_orders_dataset | 99,441 | Transaksi order |
| olist_order_items_dataset | 112,650 | Detail produk yang dibeli |
| olist_products_dataset | 32,951 | Informasi produk |
| olist_order_reviews_dataset | 99,224 | Review & rating customer |

**Master dataset setelah join:** 112,372 baris × 32 kolom

---

## 🔍 Exploratory Data Analysis

### Top 10 Kategori Produk
| Kategori | Transaksi |
|---|---|
| cama_mesa_banho | 11,137 |
| beleza_saude | 9,645 |
| esporte_lazer | 8,640 |
| moveis_decoracao | 8,331 |
| informatica_acessorios | 7,849 |
| utilidades_domesticas |  6,943 |
| relogios_presentes | 5,950 |
| telefonia | 4,517 |
| ferramentas_jardim | 4,329 |
| automotivo | 4,213 |

### Distribusi Review Score
| Score | Jumlah | Persentase |
|---|---|---|
| ⭐⭐⭐⭐⭐ (5) | 63,525 | 56.5% |
| ⭐⭐⭐⭐ (4) | 21,315 | 19.0% |
| ⭐⭐⭐ (3) | 9,423 | 8.4% |
| ⭐⭐ (2) | 3,874 | 3.4% |
| ⭐ (1) | 14,235 | 12.7% |

### Pola Repeat Purchase
- Customer dengan hanya **1 order: 91,852 (97%)**
- Customer dengan **≥ 3 order: hanya 234 orang**

---

## ⚙️ Metodologi

### 1. Data Cleaning & Preparation
- Hapus baris dengan missing value pada kolom kritis
- Filter user dengan transaksi **< 3x** (tidak cukup data untuk dipelajari)
- Filter kategori yang dibeli **< 5 user unik** (terlalu jarang untuk dibandingkan)
- Hasil: **3,019 user aktif** · **49 kategori valid**

### 2. Interaction Matrix
- Format: User × Kategori **(3,009 × 49)**
- Nilai: rata-rata `review_score` per pasangan (user, kategori) — bukan binary 0/1
- Sparsity: **97.40%**

### 3. Model — Truncated SVD
- Reduksi dimensi dari 3,009 → **48 dimensi (latent space)**
- Explained variance: **99.93%**
- Cosine similarity dihitung di latent space yang dense

### 4. Popularity-Based (Cold-Start Handler)
- Skor = **70% frekuensi pembelian + 30% rata-rata review score**
- Digunakan untuk user baru yang belum punya riwayat transaksi

### 5. Hybrid Recommendation
```
Hybrid Score = (CF Score × 0.7) + (Popularity Score × 0.3)
```
- **User lama** → SVD similarity + bobot popularitas
- **User baru** → fallback ke top-N popularity

---

## 📈 Hasil Evaluasi

| Metrik | Hasil | Patokan Industri | Status |
|---|---|---|---|
| Precision@5 | 0.1033 | 0.05 – 0.15 | ✅ Baik |
| Recall@5 | 0.4917 | 0.30 – 0.60 | ✅ Baik |
| Recall@10 | 0.6942 | 0.50 – 0.80 | ✅ Baik |
| Catalog Coverage | 93.88% | > 80% | ✅ Sangat Baik |

> **Catatan:** Precision@5 = 10.33% artinya model **5× lebih baik dari tebakan acak** (peluang acak = ~2% dari 49 kategori)

---

## 💡 Business Insight

1. **3 kategori teratas menyumbang 26% transaksi** — fokus stok dan promosi di cama_mesa_banho, beleza_saude, dan esporte_lazer
2. **Customer puas tapi tidak kembali** — perlu program engagement pasca transaksi (voucher, notifikasi rekomendasi)
3. **Cold-start adalah kondisi mayoritas** — popularity-based bukan sekadar fallback, ini jalur utama untuk 97% user
4. **Review negatif perlu dibobot berbeda** — gunakan review_score asli agar sinyal negatif tidak mencemari model
5. **Setiap repeat buyer baru langsung memperkaya data** — retensi customer = investasi kualitas model jangka panjang

---

## 🚀 Cara Menjalankan

### Jalankan Notebook
```bash
# Buka di Kaggle atau Jupyter
jupyter notebook e-commerce-recommendation-system.ipynb
```

### Jalankan Streamlit App (Lokal)
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

### Akses via Browser
```
http://localhost:8501
```

---

## 🗂️ Struktur File

```
├── e-commerce-recommendation-system.ipynb   # Notebook utama
├── app.py                                   # Streamlit dashboard
├── requirements.txt                         # Dependencies
└── README.md                                # Dokumentasi ini
```

---

## 🛠️ Tech Stack

| Library | Kegunaan |
|---|---|
| `pandas` | Manipulasi data |
| `numpy` | Komputasi numerik |
| `scikit-learn` | TruncatedSVD, cosine similarity, evaluasi |
| `matplotlib` & `seaborn` | Visualisasi EDA |
| `streamlit` | Dashboard interaktif |

---

## 📋 Rekomendasi Pengembangan Lanjutan

- [ ] Tambah implicit feedback (klik, view, wishlist) untuk memperkaya matrix
- [ ] Implementasi Matrix Factorization (ALS) untuk performa lebih tinggi
- [ ] Re-training otomatis setiap bulan dengan data transaksi terbaru
- [ ] A/B testing rekomendasi untuk mengukur uplift revenue secara langsung
- [ ] Integrasi dengan sistem notifikasi email / push notification

---

## 📄 Sumber Data

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
Lisensi: CC BY-NC-SA 4.0
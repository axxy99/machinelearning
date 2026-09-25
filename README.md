# Deteksi Spam Email dengan Machine Learning

Tugas Praktikum Mata Kuliah Kecerdasan Buatan — Bidang: **Machine Learning**
Studi Kasus: Deteksi Spam Email pada Sistem Layanan Akademik dan Teknologi Informasi Kampus

## Deskripsi

Proyek ini mendemonstrasikan penerapan Machine Learning (Multinomial Naive Bayes dan
Logistic Regression) untuk mengklasifikasikan pesan/email sebagai **spam** atau **ham**
(bukan spam), menggunakan dataset publik **SMS Spam Collection** (5.572 pesan berlabel).

Model dilatih menggunakan representasi fitur **TF-IDF** dan dievaluasi dengan metrik
akurasi, precision, recall, dan F1-score.

## Hasil Utama

| Model | Akurasi | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 97,31% | 98,37% | 81,21% | 88,97% |
| Logistic Regression | 96,77% | 100,00% | 75,84% | 86,26% |

## Struktur Repositori

```
deteksi-spam-email-ml/
├── README.md
├── notebook/
│   └── Deteksi_Spam_Email_ML.ipynb   # Notebook lengkap + hasil eksekusi
├── dataset/
│   └── dataset_spam_sms.csv          # Dataset SMS Spam Collection (5.572 baris)
├── laporan/
│   └── Laporan_ML_Deteksi_Spam_Email.pdf
├── slide/
│   └── Presentasi_Deteksi_Spam_Email_ML.pptx
├── diagram/
│   └── arsitektur_sistem.drawio      # (opsional) dibuat di draw.io/diagrams.net
└── requirements.txt
```

## Cara Menjalankan Notebook

1. Buka `notebook/Deteksi_Spam_Email_ML.ipynb` di Google Colab atau Jupyter lokal.
2. Pastikan `dataset/dataset_spam_sms.csv` berada di folder yang sama dengan notebook
   (atau sesuaikan path pada sel pertama).
3. Jalankan seluruh sel secara berurutan (Run All).

### Instalasi dependency (jika menjalankan lokal)

```bash
pip install -r requirements.txt
```

## Sumber Dataset

SMS Spam Collection Dataset — Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011).
Dataset publik yang umum digunakan untuk riset dan pembelajaran deteksi spam.

## Bidang AI & Alasan Pemilihan

Machine Learning dipilih karena permasalahan deteksi spam bersifat dinamis, berbasis pola
statistik teks, dan membutuhkan kemampuan generalisasi terhadap variasi pesan baru — sesuatu
yang sulit dicapai dengan pendekatan berbasis aturan statis. Penjelasan lengkap tersedia pada
`laporan/Laporan_ML_Deteksi_Spam_Email.pdf` (Bab II dan III).

## Penulis

- Nama: [Diisi]
- NIM: [Diisi]
- Program Studi: Sistem Informasi / Teknologi Informasi
- Mata Kuliah: Kecerdasan Buatan

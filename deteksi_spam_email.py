# # Deteksi Spam Email/SMS Menggunakan Machine Learning
# ### Tugas Praktikum Kecerdasan Buatan — Bidang: Machine Learning
# ### Studi Kasus: Deteksi Spam pada Layanan Akademik dan TI Kampus
# 
# Notebook ini mendemonstrasikan penerapan **Machine Learning (Supervised Learning)** untuk
# mengklasifikasikan pesan (email/SMS) ke dalam dua kelas: **spam** dan **ham (bukan spam)**.
# 
# Dataset yang digunakan adalah **SMS Spam Collection Dataset** (dataset publik yang umum
# digunakan untuk riset dan pembelajaran deteksi spam), terdiri dari 5.572 pesan berlabel.
# Struktur permasalahan (klasifikasi teks pendek menjadi spam/bukan spam) merepresentasikan
# kasus nyata deteksi spam pada layanan email/notifikasi kampus.
# 
# **Algoritma yang digunakan:**
# 1. Multinomial Naive Bayes
# 2. Logistic Regression (sebagai model pembanding)
# 
# **Alur kerja notebook:**
# 1. Memuat dataset
# 2. Text preprocessing
# 3. Ekstraksi fitur TF-IDF
# 4. Pelatihan model
# 5. Evaluasi model (akurasi, precision, recall, F1-score, confusion matrix)
# 6. Pengujian pada contoh pesan baru
# 

# ## 1. Import Library

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

pd.set_option('display.max_colwidth', 80)


# ## 2. Memuat Dataset
# 
# Dataset `dataset_spam_sms.csv` berisi dua kolom: `label` (ham/spam) dan `text` (isi pesan).
# File ini berada pada folder yang sama dengan notebook ini.
# 

df = pd.read_csv("dataset_spam_sms.csv")
print("Jumlah data:", df.shape[0])
print("\nDistribusi label:")
print(df["label"].value_counts())
df.head(10)


# ## 3. Text Preprocessing
# 
# Sebelum diekstraksi menjadi fitur numerik, teks pesan dibersihkan melalui beberapa langkah:
# - Mengubah menjadi huruf kecil (case folding)
# - Menghapus URL
# - Menghapus angka dan simbol/tanda baca
# - Merapikan spasi berlebih
# 

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(preprocess_text)
df[["text", "clean_text"]].head(10)


# ## 4. Pembagian Data dan Ekstraksi Fitur TF-IDF
# 
# Data dibagi menjadi data latih (80%) dan data uji (20%) dengan stratifikasi berdasarkan label
# agar proporsi kelas spam/ham tetap terjaga pada kedua subset. Representasi teks diubah menjadi
# vektor numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).
# 

X = df["clean_text"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Ukuran data latih :", X_train_tfidf.shape)
print("Ukuran data uji    :", X_test_tfidf.shape)


# ## 5. Pelatihan Model

model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_tfidf, y_train)

print("Model Naive Bayes dan Logistic Regression selesai dilatih.")


# ## 6. Evaluasi Model
# 
# Model dievaluasi menggunakan data uji yang tidak dilihat selama pelatihan, dengan metrik:
# akurasi, precision, recall, dan F1-score.
# 

results = {}
for nama, model in [("Naive Bayes", model_nb), ("Logistic Regression", model_lr)]:
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[nama] = [acc, prec, rec, f1]
    print(f"===== {nama} =====")
    print(f"Akurasi   : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print()

hasil_df = pd.DataFrame(results, index=["Akurasi", "Precision", "Recall", "F1-Score"]).T
hasil_df


# ### Confusion Matrix — Multinomial Naive Bayes

y_pred_nb = model_nb.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Multinomial Naive Bayes")
plt.tight_layout()
plt.savefig("confusion_matrix_nb.png", dpi=120)
plt.show()


print(classification_report(y_test, y_pred_nb, target_names=["Ham", "Spam"]))


# ## 7. Pengujian pada Contoh Pesan Baru
# 
# Model diuji dengan beberapa contoh pesan baru yang merepresentasikan konteks kampus
# (pengumuman akademik) dan pesan mencurigakan (indikasi spam/phishing).
# 

contoh_pesan = [
    "Congratulations! You have WON a free prize, click this link now to claim!",
    "Reminder: Your final exam schedule has been published on the academic portal.",
    "URGENT: Your account will be suspended, verify your password immediately at this link.",
    "The lecture for tomorrow is moved to room 301 at 9 AM.",
    "You have been selected for a cash reward of $5000, reply now to claim your prize."
]

contoh_bersih = [preprocess_text(t) for t in contoh_pesan]
contoh_tfidf = vectorizer.transform(contoh_bersih)
hasil_prediksi = model_nb.predict(contoh_tfidf)
hasil_proba = model_nb.predict_proba(contoh_tfidf)

for teks, label, proba in zip(contoh_pesan, hasil_prediksi, hasil_proba):
    kelas = "SPAM" if label == 1 else "HAM"
    print(f"[{kelas:4s}] (prob. spam={proba[1]:.3f})  {teks}")


# ## 8. Kesimpulan Demonstrasi
# 
# Berdasarkan hasil pelatihan dan pengujian pada dataset SMS Spam Collection (5.572 pesan berlabel),
# model **Multinomial Naive Bayes** dan **Logistic Regression** berhasil mengklasifikasikan pesan
# spam dan ham dengan tingkat akurasi tinggi (lihat tabel hasil pada Bagian 6). Model juga berhasil
# mengenali pola pesan baru yang belum pernah dilihat sebelumnya, termasuk pola umum phishing
# ("klik link", "verifikasi akun", "hadiah/reward") dan pesan akademik yang sah.
# 
# Hasil ini mendukung gagasan bahwa pendekatan Machine Learning dapat diterapkan secara efektif
# untuk membangun sistem penyaring spam pada layanan email/notifikasi kampus, sebagaimana dibahas
# pada laporan tugas praktikum yang menyertai notebook ini.
# 
# **Catatan:** Dataset yang digunakan berbahasa Inggris (dataset publik standar untuk riset spam).
# Untuk implementasi nyata pada layanan kampus berbahasa Indonesia, disarankan mengumpulkan dan
# melabeli dataset email/SMS berbahasa Indonesia agar model lebih representatif terhadap konteks
# lokal.
# 

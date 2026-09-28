# Prediksi Indeks Kualitas Air (IKA) Kabupaten/Kota di Indonesia

Proyek machine learning untuk memprediksi **Indeks Kualitas Air (IKA) tahun 2025** setiap kabupaten/kota di Indonesia berdasarkan riwayat IKA tahun 2022–2024 dan provinsinya.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-orange)
![Status](https://img.shields.io/badge/status-selesai-brightgreen)

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Metodologi](#metodologi)
- [Hasil](#hasil)
- [Struktur Repositori](#struktur-repositori)
- [Cara Menjalankan](#cara-menjalankan)
- [Contoh Penggunaan Model](#contoh-penggunaan-model)
- [Keterbatasan](#keterbatasan)
- [Pengembangan Lanjutan](#pengembangan-lanjutan)

---

## Latar Belakang

IKA adalah salah satu komponen Indeks Kualitas Lingkungan Hidup (IKLH) dengan skala 0–100; semakin tinggi nilainya, semakin baik kualitas air. Memperkirakan IKA tahun berikutnya dapat membantu memprioritaskan wilayah yang perlu perhatian lebih awal.

Proyek ini membangun model regresi untuk *forecasting* satu tahun ke depan, lalu membandingkannya dengan baseline sederhana agar terlihat apakah machine learning benar-benar memberi nilai tambah.

## Dataset

**File:** `Indeks_Kualitas_Air__IKA__Tahun_2022_-_2025.csv` (552 baris, 8 kolom)

| Kolom | Keterangan |
|---|---|
| `kedalaman` | Level wilayah: `KABKOTA` (514 baris) atau `PROV` (38 baris rekap provinsi) |
| `kode_referensi` | Kode wilayah |
| `kabkota` | Nama kabupaten/kota (atau provinsi untuk baris `PROV`) |
| `provinsi` | Nama provinsi (38 provinsi) |
| `ika_2022` … `ika_2025` | Nilai IKA per tahun (0–100) |

**Catatan kualitas data:**
- Terdapat nilai kosong pada kolom IKA (`ika_2022`: 107, `ika_2023`: 26, `ika_2024`: 11, `ika_2025`: 54).
- Baris level provinsi dipisahkan dari kabupaten/kota agar tidak terjadi pencampuran level dan perhitungan ganda.
- Baris dengan `ika_2025` kosong dibuang (target tidak boleh ditebak), sehingga **460 kabupaten/kota** dipakai untuk pemodelan.

## Metodologi

1. **Pembersihan data** — pisahkan level `PROV` dan `KABKOTA`, buang baris tanpa target.
2. **Eksplorasi data (EDA)** — distribusi IKA per tahun, tren rata-rata, korelasi antar tahun, dan IKA per provinsi.
3. **Rekayasa fitur**
   - `ika_2022`, `ika_2023`, `ika_2024`
   - `rata_riwayat`: rata-rata IKA yang tersedia
   - `tren`: selisih `ika_2024 − ika_2023`
   - `provinsi` (kategori)
4. **Split data** — 80% latih (368) dan 20% uji (92).
5. **Preprocessing (dalam pipeline)** — imputasi median + indikator nilai kosong + standarisasi untuk fitur numerik; imputasi modus + One-Hot Encoding untuk provinsi.
6. **Pemilihan model** — validasi silang 5-fold dengan metrik MAE untuk Ridge Regression, Random Forest, dan Gradient Boosting, dibandingkan dengan baseline *"IKA 2025 = IKA 2024"*.
7. **Evaluasi** — model terbaik diuji sekali pada data uji (MAE, RMSE, R²) dan dikonversi ke kategori IKA.
8. **Interpretasi & penyimpanan** — pengaruh fitur, lalu model disimpan dalam format `.joblib`.

## Hasil

**Validasi silang 5-fold (data latih, MAE — makin kecil makin baik):**

| Model | MAE |
|---|---|
| Baseline (IKA 2024) | 3,83 |
| **Ridge Regression** | **3,55** |
| Gradient Boosting | 3,68 |
| Random Forest | 3,77 |

**Evaluasi pada data uji (model terpilih: Ridge Regression):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (IKA 2024) | 3,78 | 4,98 | 0,37 |
| **Ridge Regression** | **3,35** | **4,24** | **0,54** |

Akurasi kategori IKA (Sangat Kurang / Kurang / Sedang / Baik / Sangat Baik) pada data uji: **≈ 74%**.

**Interpretasi singkat:** model mengalahkan baseline, dengan rata-rata kesalahan sekitar 3,3 poin IKA. Ridge unggul atas model berbasis pohon karena datanya kecil dan hubungan antar tahun cenderung linear. Angka pasti dapat sedikit berbeda pada mesin Anda.

> Ambang kategori yang dipakai (asumsi berdasarkan acuan KLHK): Sangat Baik ≥ 90, Baik 70–<90, Sedang 50–<70, Kurang 25–<50, Sangat Kurang < 25.

## Struktur Repositori

```
.
├── model_kualitas_air.ipynb                          # Notebook utama (analisis + model)
├── Indeks_Kualitas_Air__IKA__Tahun_2022_-_2025.csv   # Dataset
├── README.md
└── hasil_model/                                      # Dibuat otomatis saat notebook dijalankan
    ├── model_ika.joblib
    ├── distribusi_ika.png
    ├── korelasi_ika.png
    ├── aktual_vs_prediksi.png
    ├── confusion_matrix_kategori.png
    └── pengaruh_fitur.png
```

## Cara Menjalankan

**1. Clone repositori**

```bash
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>
```

**2. Instal dependensi**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

**3. Jalankan notebook**

```bash
jupyter notebook model_kualitas_air.ipynb
```

Pilih *Run All*. Pastikan file CSV berada di folder yang sama dengan notebook (atau ubah `DATA_PATH` pada sel pertama).

**Google Colab:** unggah `model_kualitas_air.ipynb` dan file CSV ke Colab, lalu jalankan semua sel.

## Contoh Penggunaan Model

Di dalam notebook (bagian *Prediksi Data Baru*) tersedia fungsi `prediksi_ika_2025`. Nilai riwayat yang tidak diketahui boleh diisi `None`.

```python
nilai, kategori = prediksi_ika_2025(70.1, 72.3, 71.5, "Jawa Barat")
print(nilai, kategori)   # contoh keluaran: 71.83 Baik

nilai, kategori = prediksi_ika_2025(None, 68.0, 69.5, "Jawa Timur")
print(nilai, kategori)   # contoh keluaran: 70.97 Baik
```

Nama provinsi harus sesuai penulisan pada dataset (contoh: `Jawa Barat`, `Nanggroe Aceh Darussalam`).

## Keterbatasan

- Hanya ada **empat titik waktu** per wilayah, dan tidak ada variabel penyebab (curah hujan, industri, kepadatan penduduk, tutupan lahan, dll.).
- IKA dapat berubah cukup besar antar tahun, misalnya karena perubahan titik pemantauan, sehingga R² model baru sekitar 0,5.
- Wilayah dengan `ika_2025` kosong dikeluarkan dari pelatihan, sehingga model belum tentu mewakili wilayah yang datanya sering tidak tersedia.
- Prediksi sebaiknya dipakai sebagai perkiraan awal, **bukan pengganti pengukuran lapangan**.

## Pengembangan Lanjutan

- Menambahkan variabel eksternal (curah hujan, tutupan lahan, jumlah penduduk, aktivitas industri).
- Memakai pendekatan *time series* jika data historis lebih panjang tersedia.
- *Hyperparameter tuning* (`GridSearchCV` / `RandomizedSearchCV`).
- Membuat aplikasi sederhana (Streamlit) agar model mudah dicoba tanpa membuka notebook.

## Lisensi & Sumber Data

Tambahkan lisensi kode (mis. MIT) dan cantumkan sumber resmi dataset IKA di sini sebelum dipublikasikan.

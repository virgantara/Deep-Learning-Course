## Asesmen
Kemampuan yang diharapkan:
1. Sub-CPMK01-03 - Mahasiswa mampu menjelaskan fungsi aktivasi dan peranannya dalam deep learning

### Judul Tugas
**Tugas Kelompok: Klasifikasi Bunga Iris dengan Multilayer Perceptron (MLP)**

### Deskripsi Tugas
Proyek ini merupakan bagian dari tugas kelompok untuk mata kuliah *[Nama Mata Kuliah]* dengan tujuan untuk mengimplementasikan algoritma **Multilayer Perceptron (MLP)** menggunakan **PyTorch** untuk melakukan klasifikasi bunga Iris berdasarkan dataset Iris dari Scikit-learn.

Kelompok diminta untuk melakukan eksplorasi data, membangun model MLP, melatih model, mengevaluasi performa, serta membandingkannya dengan model sederhana seperti Logistic Regression.

### Tools
- Python 3.8+
- PyTorch
- Scikit-learn
- Matplotlib
- Pandas
- Jupyter Notebook

### Struktur Proyek

```plaintext
.
├── model.py               # Arsitektur MLP
├── train.py               # Skrip pelatihan model
├── predict.py             # Skrip untuk inference/prediksi data baru
├── model.pth              # Model hasil pelatihan
├── iris_logreg.py         # Perbandingan dengan Logistic Regression
├── laporan_akhir.ipynb    # Laporan utama (Notebook)
├── laporan_akhir.pdf      # Laporan dalam bentuk PDF
├── README.md              # Dokumentasi ini
```


## Tujuan Pembelajaran
- Menjelaskan fungsi aktivasi dalam deep learning.
- Mengimplementasikan MLP secara langsung menggunakan PyTorch.
- Membandingkan hasil model deep learning dengan baseline sederhana.
- Menyusun laporan ilmiah berbasis eksperimen kode dan analisis hasil.

1. Memuat dataset Iris dan melakukan eksplorasi data.
2. Membuat arsitektur MLP dengan setidaknya 1 hidden layer menggunakan PyTorch.
3. Melatih model menggunakan dataset Iris dengan optimasi dan fungsi loss yang sesuai.
4. Melakukan evaluasi model menggunakan metrik accuracy.
5. Membuat visualisasi loss dan akurasi selama training.
6. Menganalisis hasil: Menjelaskan bagaimana performa model dan kemungkinan perbaikannya.
7. Bandingkan hasil model MLP yang Anda buat dengan model sederhana seperti Logistic Regression. Jelaskan mengapa MLP lebih unggul atau justru lebih buruk.

### Instruksi Teknis
1. Gunakan Python dan PyTorch untuk membangun model.
2. Gunakan Scikit-learn untuk memuat dataset.
3. Latih model selama minimal 100 epoch dengan optimizer Adam atau SGD.
4. Gunakan fungsi aktivasi ReLU pada hidden layer dan Softmax pada output layer.
5. Simpan model yang telah dilatih dan buat script untuk melakukan prediksi pada data baru.
6. Tambahkan di laporan bagaimana model yang dibuat dapat diterapkan dalam dunia nyata dan apa saja tantangan yang mungkin muncul
7. Laporan dikumpulkan dalam format notebook (.ipynb) dan PDF.

### Rubrik Penilaian Tugas Kelompok

| No | Kriteria Penilaian                  | Deskripsi Penilaian                                                                 | Bobot (%) |
|----|-------------------------------------|--------------------------------------------------------------------------------------|-----------|
| 1  | **Eksplorasi Dataset**              | Dataset dimuat dengan benar dan eksplorasi data dilakukan menyeluruh                | 10        |
|    | - Memuat dataset Iris dengan benar  |                                                                                      | 5         |
|    | - Analisis jumlah kelas, distribusi fitur, dan insight dari data               | 5         |
| 2  | **Implementasi Arsitektur MLP**     | Arsitektur model MLP dibangun dengan baik menggunakan PyTorch                       | 10        |
|    | - Terdapat input, hidden, dan output layer                                     | 5         |
|    | - Penggunaan fungsi aktivasi yang sesuai (ReLU & Softmax)                      | 5         |
| 3  | **Proses Pelatihan Model**          | Model dilatih menggunakan konfigurasi dan optimasi yang sesuai                      | 10        |
|    | - Penggunaan loss function dan optimizer yang tepat                            | 5         |
|    | - Model disimpan dengan benar (`model.pth`)                                    | 5         |
| 4  | **Evaluasi Model dan Perbandingan** | Model dievaluasi dengan akurasi dan dibandingkan dengan model sederhana             | 10        |
|    | - Menghitung akurasi model                                                  | 5         |
|    | - Membandingkan dengan Logistic Regression + analisis hasil                   | 5         |
| 5  | **Visualisasi Hasil Pelatihan**     | Grafik loss dan akurasi disajikan dengan baik selama training                       | 10        |
|    | - Menampilkan grafik loss dan akurasi per epoch                               | 10        |
| 6  | **Analisis dan Refleksi Kelompok** | Evaluasi performa, refleksi terhadap kelebihan/kekurangan, dan penerapan nyata     | 15        |
|    | - Analisis performa model & hasil evaluasi                                     | 5         |
|    | - Kelebihan dan kekurangan MLP dibanding pendekatan lain                      | 5         |
|    | - Tantangan penerapan model dalam dunia nyata                                 | 5         |
| 7  | **Kualitas dan Struktur Laporan**   | Kerapihan, format, dan dokumentasi kontribusi anggota                               | 15        |
|    | - Struktur laporan sistematis dan mudah dibaca                                | 5         |
|    | - Format file sesuai (Notebook + PDF) + refleksi kontribusi anggota            | 5         |
|    | - Penjelasan kerja sama dalam tim                                             | 5         |
| **Total**                                 |                                                                                      | **100**   |

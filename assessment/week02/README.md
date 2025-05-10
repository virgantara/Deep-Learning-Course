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


| No | Nama                                | Kriteria                                                                 | Nama En                          | Bobot |
|----|-------------------------------------|---------------------------------------------------------------------------|----------------------------------|--------|
| 1  | Dokumentasi & Laporan               | Laporan disusun sistematis, lengkap, dan sesuai format (.ipynb & .pdf)   | Documentation & Reporting        | 20     |
| 2  | Pemahaman Konsep & Penerapan       | Menunjukkan pemahaman arsitektur MLP, fungsi aktivasi, training dan evaluasi | Concept Understanding & Application | 20     |
| 3  | Kompleksitas & Kreativitas Solusi  | Model dikembangkan dengan baik, dibandingkan dengan baseline, ada analisis dan visualisasi | Solution Complexity & Creativity | 20     |
| 4  | Kolaborasi & Peran Tim             | Tugas dibagi secara adil, kontribusi anggota jelas, refleksi individu disertakan | Collaboration & Team Role        | 40     |
|    | **Total**                           |                                                                           |                                  | **100** |

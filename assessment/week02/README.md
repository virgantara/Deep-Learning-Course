## Asesmen
Kemampuan yang diharapkan:
1. Sub-CPMK01-03 - Mahasiswa mampu menjelaskan fungsi aktivasi dan peranannya dalam deep learning

### Judul Tugas
**Tugas Individu: Klasifikasi Bunga Iris dengan Multilayer Perceptron (MLP) menggunakan PyTorch**

### Deskripsi Tugas
Pada tugas ini, mahasiswa diminta untuk mengimplementasikan Multilayer Perceptron (MLP) menggunakan PyTorch untuk melakukan klasifikasi bunga Iris berdasarkan dataset Iris dari Scikit-learn. Mahasiswa harus:

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

### Rubrik Penilaian

| No | Kriteria Penilaian | Bobot (%) |
|----|--------------------|-----------|
| 1  | **Eksplorasi Dataset** | 10 |
|    | - Memuat dataset Iris dengan benar | 5 |
|    | - Melakukan eksplorasi data (jumlah kelas, distribusi fitur, insight dari data) | 5 |
| 2  | **Implementasi Arsitektur MLP** | 10 |
|    | - Membangun model dengan PyTorch (input, hidden, output layer) | 5 |
|    | - Menggunakan fungsi aktivasi yang sesuai (ReLU, Softmax) | 5 |
| 3  | **Proses Pelatihan Model** | 10 |
|    | - Melakukan training dengan loss function dan optimizer yang sesuai | 5 |
|    | - Menyimpan model yang telah dilatih | 5 |
| 4  | **Evaluasi Model** | 10 |
|    | - Menghitung akurasi model dengan benar | 5 |
|    | - Membandingkan akurasi dengan model sederhana (misal: Logistic Regression) | 5 |
|    | - Menjelaskan faktor yang memengaruhi hasil akurasi | 5 |
| 5  | **Visualisasi Hasil** | 10 |
|    | - Menampilkan grafik loss dan akurasi selama training | 10 |
| 6  | **Analisis dan Kesimpulan** | 15 |
|    | - Menjelaskan performa model berdasarkan hasil evaluasi | 5 |
|    | - Menjelaskan kelebihan dan kelemahan model dibandingkan pendekatan lain | 5 |
|    | - Menyampaikan kemungkinan tantangan dan perbaikan model di dunia nyata | 5 |
| 7  | **Kualitas Laporan** | 15 |
|    | - Struktur laporan jelas dan sistematis | 5 |
|    | - Format file sesuai instruksi (Notebook + PDF) | 5 |
| **Total** |  | **100** |

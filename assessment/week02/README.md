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


| No | Aspek                          | Kriteria Penilaian                                                                                                      | A (≥ 80.01)                                                                                   | B (65.01–80.00)                                                                                 | C (50.01–65.00)                                                                                  | D (≤ 50.00)                                                                                      | Bobot |
|----|--------------------------------|--------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------|
| 1  | **Dokumentasi & Laporan**      | Struktur laporan, format file, penulisan kode, dan dokumentasi                                                          | Laporan lengkap, sistematis, sangat rapi, komentar kode jelas dan informatif                | Laporan cukup sistematis, dokumentasi kode ada meskipun bisa ditingkatkan                     | Laporan kurang rapi, struktur tidak logis, komentar minim                                       | Laporan tidak sistematis, format salah, tidak ada komentar, kode sulit dipahami                 | 20     |
| 2  | **Pemahaman Konsep & Penerapan** | Implementasi MLP (arsitektur, aktivasi), training, dan evaluasi                                                         | Implementasi tepat, sesuai konsep, penjelasan teori mendalam dan akurat                     | Implementasi cukup tepat, penjelasan konsep ada namun belum mendalam                          | Implementasi kurang tepat, penjelasan konsep terbatas                                           | Banyak kesalahan implementasi, tidak menunjukkan pemahaman konsep dasar                        | 20     |
| 3  | **Kompleksitas & Kreativitas Solusi** | Perbandingan dengan model sederhana, visualisasi metrik, dan analisis hasil                                             | Ada perbandingan valid, visualisasi jelas, analisis kritis dan menyeluruh                   | Ada perbandingan dan visualisasi, analisis cukup                                               | Visualisasi sederhana, tidak ada perbandingan, analisis terbatas                               | Tidak ada visualisasi atau analisis, solusi terlalu sederhana                                  | 20     |
| 4  | **Kolaborasi & Peran Tim**     | Pembagian tugas, refleksi kontribusi individu, dan kekompakan hasil kerja                                               | Tugas terbagi adil, refleksi kontribusi lengkap, kerja tim sangat baik                      | Ada pembagian tugas dan refleksi kontribusi, kerja tim cukup terkoordinasi                    | Pembagian tugas tidak merata, refleksi minim, kolaborasi kurang terasa                         | Tidak ada refleksi, hanya satu anggota yang aktif, kerja tim lemah                             | 40     |
|    | **Total**                      |                                                                                                                          |                                                                                               |                                                                                                  |                                                                                                   |                                                                                                   | **100** |


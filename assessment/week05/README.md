# Tugas Individu
# Deskripsi

Setiap mahasiswa diminta membangun model klasifikasi teks menggunakan arsitektur RNN (misalnya: LSTM, GRU) untuk memisahkan dua kelas teks pilihan sendiri. Contoh kasus:

1. opini positif vs negatif,
1. spam vs non-spam,
1. berita politik vs olahraga.

### Fokus utama tugas ini bukan hanya hasil akhir, melainkan proses berpikir, eksplorasi ide, dan refleksi mandiri.

# Ketentuan Dataset
1. Dataset minimal 200 data teks, terdiri dari 100 data untuk masing-masing kelas.
1. Data dapat dikumpulkan sendiri atau disusun dari sumber terbuka.
1. Wajib ada variasi panjang teks, gaya bahasa, dan penjelasan pemilihan data.

# Output Individu
1. Notebook Jupyter (.ipynb) berisi komentar kritis: mengapa melakukan langkah tertentu, bukan sekadar eksekusi.
1. Laporan PDF (maksimal 10 halaman) berisi dokumentasi proses, eksplorasi, dan refleksi pribadi.
1. `README.md` ringkasan tugas dan cara menjalankan kode.
1. Tautan GitHub individu dan link GitHub tugas individu di SIAKAD

## Contoh struktur laporan

# Laporan Tugas Individu: Klasifikasi Teks Menggunakan RNN

## 1. Pendahuluan
Jelaskan latar belakang tugas ini, mengapa klasifikasi teks penting, serta tujuan dan ruang lingkup tugas Anda.

## 2. Dataset
- **Sumber Data**: Sebutkan dari mana data Anda diperoleh atau bagaimana Anda mengumpulkannya.
- **Deskripsi Dataset**: Jumlah data per kelas, contoh teks, variasi panjang/gaya teks.
- **Alasan Pemilihan Dataset**: Mengapa Anda memilih topik atau jenis data ini?

## 3. Implementasi Model
### 3.1 Arsitektur RNN
- Jenis model (LSTM, GRU, BiLSTM, dll)
- Penjelasan singkat arsitektur dan parameter

### 3.2 Preprocessing
- Tokenisasi, padding, normalisasi, dll.

### 3.3 Pengaturan Eksperimen
- Jumlah epoch, batch size, optimizer, loss function, dsb.

### 3.4 Log Eksperimen
Ceritakan proses eksplorasi dan iterasi model Anda, misalnya:
- Apa konfigurasi awal Anda? Apa masalahnya?
- Apa yang Anda ubah? Mengapa?
- Bagaimana performa berubah?

Contoh tabel log eksperimen:

| Percobaan | Model | Dropout | Optimizer | Akurasi Validasi | Catatan |
|-----------|--------|----------|-----------|------------------|---------|
| #1        | LSTM(2) | 0        | Adam      | 64.2%            | Overfitting sejak epoch 3 |
| #2        | LSTM(2) | 0.5      | Adam      | 72.5%            | Lebih stabil |

## 4. Evaluasi Hasil
- Metrik akurasi, loss, confusion matrix.
- Visualisasi learning curve (jika ada).
- Analisis performa model dan kemungkinan penyebab hasil.

## 5. Refleksi Pribadi
- Apa tantangan utama yang Anda hadapi?
- Apa solusi yang Anda coba dan bagaimana hasilnya?
- Jika Anda menggunakan ChatGPT atau AI lain, bagian mana yang dibantu dan bagaimana Anda memverifikasinya?
- Apa pelajaran paling penting dari tugas ini?

## 6. Kesimpulan dan Saran
- Ringkasan hasil terbaik
- Rekomendasi untuk pengembangan selanjutnya (misal: ganti model, tambah data, pretraining, dll)

## 7. Referensi
Gunakan format referensi standar untuk menyebutkan sumber pustaka, artikel, atau tutorial.

---



# Rubrik Penilaian Individu

| No | Aspek                            | Kriteria Penilaian | A (≥ 80.01) | B (65.01–80.00) | C (50.01–65.00) | D (≤ 50.00) | Bobot |
|----|----------------------------------|---------------------|-------------|------------------|------------------|-------------|--------|
| 1  | Dokumentasi & Laporan           | Struktur laporan, penjelasan keputusan, komentar dalam kode | Laporan rapi, ada argumen setiap keputusan teknis, kode terkomentar baik | Cukup rapi, penjelasan sebagian langkah | Penjelasan kurang logis atau tidak utuh | Tidak jelas dan sulit dipahami | 25 |
| 2  | Kualitas Dataset & Justifikasi  | Jumlah, variasi, dan alasan memilih data | 200 data bervariasi, ada argumen kuat pemilihan data | Cukup lengkap, justifikasi terbatas | Kurang bervariasi atau justifikasi lemah | Dataset acak/tidak dijelaskan | 20 |
| 3  | Implementasi RNN & Eksplorasi   | Pemilihan model, arsitektur, eksperimen, fine-tuning | Implementasi rapi, eksplorasi 2+ konfigurasi, evaluasi tiap percobaan | Implementasi benar, tapi eksplorasi terbatas | Hanya 1 pendekatan tanpa eksplorasi | Banyak kesalahan atau asal pakai | 20 |
| 4  | Visualisasi & Evaluasi          | Akurasi/loss, confusion matrix, penjelasan hasil | Visualisasi lengkap, evaluasi tajam & disertai argumen | Visualisasi ada tapi kurang dalam | Visualisasi terbatas, analisis minim | Tanpa evaluasi atau copy-paste | 20 |
| 5  | Refleksi Pribadi                | Insight, tantangan, pembelajaran | Refleksi dalam, menyebutkan kegagalan & solusi konkrit | Cukup dalam, insight terbatas | Umum dan datar | Hanya deskripsi teknis | 15 |

**Total Bobot: 100**

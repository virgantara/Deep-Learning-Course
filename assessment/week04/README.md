# Penugasan: Klasifikasi Dua Jenis Objek dengan Transfer Learning

Tugas ini terdiri dari dua bagian: **Tugas Individu** dan **Tugas Kelompok**. Mahasiswa diminta untuk mengerjakan klasifikasi citra dua objek menggunakan transfer learning, lalu mendiskusikan dan membandingkan hasil di tingkat kelompok.

---

## A. Tugas Individu

### Deskripsi
Setiap mahasiswa diminta untuk membangun model klasifikasi gambar untuk dua jenis objek pilihan sendiri (misalnya: apel vs jeruk, motor vs sepeda, masker vs tanpa masker) menggunakan pretrained model (misalnya: VGG16, ResNet, MobileNet, dll).

Mahasiswa wajib mengumpulkan dataset sendiri minimal **200 foto**, terdiri dari **100 foto objek pertama** dan **100 foto objek kedua**, dengan **variasi sudut pengambilan** untuk memperkaya data.

### Output Individu
- Jupyter Notebook (`.ipynb`)
- Laporan PDF (maksimal 10 halaman)
- README.md
- Tautan GitHub individu dan taruh link Github tugas individu di SIAKAD

### Rubrik Penilaian Individu

| No | Aspek                          | Kriteria Penilaian                                                                                 | A (≥ 80.01)                                                                                                  | B (65.01–80.00)                                                                                     | C (50.01–65.00)                                                                 | D (≤ 50.00)                                                              | Bobot |
|----|--------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------|--------|
| 1  | Dokumentasi & Laporan          | Struktur laporan, penulisan kode, format file, dan dokumentasi                                     | Laporan rapi dan sistematis, kode jelas dan terdokumentasi dengan baik, ada penjelasan tiap tahap           | Laporan cukup sistematis, kode dapat dibaca, dokumentasi masih bisa ditingkatkan                   | Struktur kurang rapi, komentar kode dan dokumentasi minim                      | Laporan tidak sistematis, tidak ada komentar, sulit dipahami              | 25     |
| 2  | Pengumpulan & Kualitas Dataset | Jumlah foto, variasi sudut, relevansi dua objek                                                    | 200 foto lengkap, dua objek jelas, variasi sudut nyata, data bersih                                          | 200 foto lengkap tapi variasi sudut terbatas                                                       | Kurang dari 200 atau variasi sudut sangat minim                                | Dataset tidak sesuai instruksi                                           | 20     |
| 3  | Implementasi Transfer Learning | Pemilihan pretrained model, cara fine-tuning, freezing layer                                       | Pemilihan model relevan, teknik transfer learning benar, implementasi rapi dan tepat                        | Implementasi cukup tepat, teknik dasar digunakan meski eksplorasi terbatas                          | Implementasi kurang tepat atau pemilihan model tidak sesuai                   | Banyak kesalahan teknis, tidak memahami prinsip transfer learning         | 20     |
| 4  | Visualisasi & Evaluasi Hasil   | Visualisasi akurasi/loss, confusion matrix, evaluasi model                                         | Visualisasi lengkap dan jelas (training history, confusion matrix), evaluasi akurat dan tajam               | Visualisasi dan evaluasi ada, meskipun kurang mendalam                                            | Visualisasi terbatas, evaluasi tidak lengkap                                  | Tidak ada visualisasi atau evaluasi                                     | 20     |
| 5  | Analisis Refleksi Pribadi      | Penjelasan pengalaman individu, tantangan, insight yang diperoleh                                 | Refleksi mendalam, menyebutkan tantangan dan solusi, pembelajaran personal terasa kuat                     | Refleksi cukup jelas, menyebutkan proses dan hasil pembelajaran individu                           | Refleksi dangkal, hanya menceritakan proses secara umum                        | Tidak ada refleksi atau hanya mengulang isi laporan teknis                | 15     |
|    | **Total**                      |                                                                                                     |                                                                                                              |                                                                                                      |                                                                                  |                                                                           | **100** |

---

## B. Tugas Kelompok

### Deskripsi
Setiap kelompok beranggotakan 3–4 mahasiswa. Setelah menyelesaikan tugas individu, anggota kelompok diminta untuk melakukan:

- Komparasi hasil model antar anggota (misalnya akurasi, loss, confusion matrix)
- Diskusi pendekatan (augmentasi, pretrained model, fine-tuning)
- Menarik kesimpulan bersama tentang strategi paling efektif dari eksperimen

### Output Kelompok
- Laporan PDF gabungan hasil komparasi (maks. 5 halaman)
- Link GitHub berisi source code, file laporan pdf dan taruh link Github di SIAKAD

### Rubrik Penilaian Kelompok

| No | Aspek                      | Kriteria Penilaian                                                                 | A (≥ 80.01)                                                                       | B (65.01–80.00)                                                         | C (50.01–65.00)                                                | D (≤ 50.00)                                                       | Bobot |
|----|----------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------|--------|
| 1  | Komparasi Model            | Perbandingan hasil model dari tiap anggota                                         | Komparasi lengkap dan objektif, mencakup metrik dan analisis kelebihan/kekurangan | Komparasi dilakukan tapi belum mendalam                                | Komparasi kurang jelas, hanya menampilkan hasil akhir saja      | Tidak ada komparasi hasil antar anggota                          | 40     |
| 2  | Diskusi Strategi           | Pembahasan pendekatan: data, model, augmentasi, fine-tuning                        | Diskusi mendalam dan analisis strategi yang dipakai tiap anggota                  | Diskusi ada tapi belum menyentuh banyak aspek                         | Diskusi umum dan dangkal, kurang eksplorasi                      | Tidak ada diskusi strategi                                       | 30     |
| 3  | Kesimpulan Bersama         | Ringkasan tentang model dan pendekatan terbaik menurut kelompok                   | Kesimpulan logis, mencerminkan hasil diskusi dan eksperimen                       | Kesimpulan ada namun kurang kuat argumennya                          | Kesimpulan kurang mewakili analisis                             | Tidak ada kesimpulan atau hanya berupa opini tanpa dasar         | 20     |
| 4  | Koordinasi & Dokumentasi   | Bukti kolaborasi dan pembagian tugas (misal: dokumentasi nama dan peran anggota)  | Dokumentasi kontribusi tiap anggota jelas dan merata                              | Dokumentasi kontribusi ada namun tidak merata                         | Satu-dua anggota dominan, dokumentasi minim                     | Tidak jelas siapa mengerjakan apa                                | 10     |
|    | **Total**                  |                                                                                    |                                                                                   |                                                                        |                                                                |                                                                  | **100** |

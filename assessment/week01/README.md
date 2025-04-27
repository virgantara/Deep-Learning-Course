# Asesmen Pertemuan ke-1

Tema Pengenalan Deep Learning dan Neural Network
## Sub-CPMK/Sub-CLOs
1. SubCPMK01-01 Menjelaskan konsep dasar neural network dan deep learning 
2. SubCPMK01-02 Memahami arsitektur perceptron dan multilayer perceptron (MLP)

## Asesmen
Diskusi Kelompok untuk Pertemuan 1: Pengenalan Deep Learning dan Neural Network
Tema Diskusi:

"Mengapa Kita Membutuhkan Deep Learning?"
Tujuan Diskusi:

1. Mahasiswa memahami perbedaan antara Machine Learning (ML) tradisional dan Deep Learning (DL).
2. Mahasiswa mampu menjelaskan kelebihan dan tantangan Deep Learning dibandingkan ML konvensional.
3. Mahasiswa mampu mengidentifikasi aplikasi Deep Learning di berbagai bidang.

## Teknis Diskusi

### Pembagian Kelompok
1. Mahasiswa dibagi menjadi 5-6 kelompok (tergantung jumlah kelas).
2. Setiap kelompok terdiri dari 3-5 mahasiswa.
3. Setiap kelompok diberikan studi kasus berbeda terkait penerapan Deep Learning.

### Tugas Kelompok
1. Setiap kelompok diberikan studi kasus nyata terkait Deep Learning.
2. Mereka harus menganalisis apakah ML tradisional cukup atau apakah DL diperlukan.
3. Mereka mendiskusikan kelebihan dan kekurangan pendekatan DL dalam studi kasus mereka.
4. Hasil diskusi dipresentasikan dalam bentuk mind map atau tabel perbandingan.

### Durasi
1. 15 menit: Mahasiswa membaca studi kasus dan berdiskusi dalam kelompok.
2. 10 menit: Setiap kelompok membuat ringkasan dalam mind map/tabel.
3. 15 menit: Presentasi singkat dari masing-masing kelompok (maks. 3 menit/kelompok).
4. 10 menit: Diskusi kelas dan refleksi dari dosen.

## Rubrik Penilaian
#### 1. Pemahaman Konsep (30 Poin)
- 30: Memahami dengan sangat baik perbedaan ML vs DL dan dapat menjelaskan dengan jelas.
- 25: Memahami perbedaan ML vs DL dengan baik, tetapi ada sedikit kekurangan dalam penjelasan.
- 20: Memahami konsep secara umum tetapi kurang mendalam dalam menjelaskan.
- 15: Pemahaman kurang dan sulit menjelaskan perbedaan ML vs DL.
- 10: Tidak menunjukkan pemahaman konsep dengan baik.

#### 2. Analisis Studi Kasus (25 Poin)
- 25: Analisis sangat mendalam, mengaitkan teori dengan studi kasus secara akurat.
- 20: Analisis cukup baik, tetapi masih ada aspek yang kurang diperhatikan.
- 15: Analisis masih dangkal dan kurang menghubungkan teori dengan studi kasus.
- 10: Analisis lemah dan tidak menunjukkan pemahaman terhadap studi kasus.
- 5: Tidak ada analisis yang jelas terhadap studi kasus.

#### 3. Kualitas Presentasi Kelompok (20 Poin)

- 20: Penyampaian sangat jelas, sistematis, dan menarik.
- 17: Penyampaian jelas, tetapi masih ada sedikit kurang dalam struktur atau penjelasan.
- 14: Penyampaian kurang sistematis atau tidak cukup menarik.
- 10: Penyampaian tidak jelas atau sulit dipahami.
- 5: Tidak berpartisipasi dalam presentasi.

#### 4. Partisipasi dan Kerja Sama Tim (15 Poin)

- 15: Semua anggota berkontribusi aktif dan diskusi berjalan dengan baik.
- 12: Sebagian besar anggota aktif, tetapi ada yang kurang berkontribusi.
- 10: Partisipasi tidak merata dan sebagian anggota pasif.
- 7: Hanya satu atau dua orang yang aktif, yang lain tidak berpartisipasi.
- 5: Tidak ada kerja sama tim yang jelas.

#### 5. Kreativitas Penyajian (10 Poin)
- 10: Menggunakan mind map/tabel yang sangat menarik dan jelas.
- 8: Visualisasi cukup baik, tetapi bisa lebih menarik.
- 6: Mind map/tabel ada tetapi kurang jelas atau rapi.
- 4: Penyajian kurang menarik dan sulit dipahami.
- 2: Tidak menggunakan mind map/tabel dalam presentasi.


## Tema Diskusi:
### Studi Kasus 1: Deteksi Penyakit dari Citra Medis
Sebuah rumah sakit ingin mengembangkan sistem otomatis untuk mendeteksi kanker paru-paru dari hasil CT-Scan.
Tim sebelumnya mencoba menggunakan machine learning tradisional (SVM, Random Forest) dengan fitur buatan tangan (misal tekstur, ukuran nodul), tapi akurasinya belum memuaskan.

Pertanyaan Diskusi:
 1. Menurut kalian, apa tantangan mengajarkan komputer untuk mendeteksi kanker dari gambar?
 1. Kira-kira lebih mudah pakai aturan buatan manusia atau membiarkan komputer belajar sendiri dari data?
 1. Bagaimana kalian membayangkan Deep Learning membantu di kasus ini?

### Studi Kasus 2: Pengenalan Wajah di Smartphone
Sebuah perusahaan teknologi ingin mengembangkan fitur buka-kunci smartphone berbasis pengenalan wajah.
Model awal menggunakan pengolahan citra tradisional: ekstraksi fitur wajah (seperti jarak antara mata) lalu diklasifikasikan dengan KNN. Namun, hasilnya tidak akurat untuk kondisi pencahayaan buruk.
Pertanyaan Diskusi:
Menurut kalian, apa kesulitan mengenali wajah manusia dengan komputer? 1. Bagaimana DL (seperti CNN) memperbaiki akurasi dalam kondisi nyata?
 1. Apa risiko atau masalah lain yang muncul saat menggunakan DL untuk biometrik?
 1. Kalau manusia bisa mengenali wajah dengan mudah, kenapa komputer perlu belajar dari banyak data?
 1. Mengapa Deep Learning mungkin diperlukan di sini?

### Studi Kasus 3: Mobil Otonom (Self-Driving Car)
Perusahaan mobil otonom memerlukan sistem yang mampu mengenali rambu lalu lintas, pejalan kaki, dan kendaraan lain dalam berbagai kondisi cuaca. 
Pertanyaan Diskusi:
 1. Apa saja hal yang harus dikenali mobil otonom di jalanan?
 1. Menurut kalian, apakah cukup memberi aturan-aturan sederhana supaya mobil berjalan?
 1. Bagaimana Deep Learning bisa membuat mobil belajar dari banyak situasi nyata?

### Studi Kasus 4: Penerjemahan Bahasa Otomatis (Machine Translation)
Sebuah perusahaan ingin membuat aplikasi penerjemah bahasa real-time dari Bahasa Indonesia ke Bahasa Jepang.
Sistem sebelumnya menggunakan pendekatan berbasis aturan (rule-based) dan machine learning klasik, tapi hasil terjemahannya kaku dan tidak natural.
Pertanyaan Diskusi:
 1. Menurut kalian, apa kesulitan menerjemahkan bahasa manusia?
 1. Kenapa tidak cukup kalau hanya pakai daftar kata dan aturan?
 1. Apa manfaat membiarkan komputer belajar dari banyak contoh kalimat?

### Studi Kasus 5: Sistem Rekomendasi Film
Sebuah platform streaming ingin mengembangkan sistem rekomendasi film untuk pengguna baru.
Pertanyaan Diskusi:
 1. Menurut kalian, bagaimana cara komputer tahu film apa yang disukai seseorang?
 1. Apakah cukup hanya melihat film yang sudah pernah ditonton?
 1. Bagaimana Deep Learning bisa membantu mengenali pola selera seseorang?

### Studi Kasus 6: Chatbot Layanan Pelanggan
Sebuah perusahaan ingin membuat chatbot otomatis yang bisa menjawab pertanyaan pelanggan melalui chat tanpa perlu campur tangan manusia.
Pertanyaan Diskusi:
 1. Menurut kalian, apa kesulitan membuat chatbot yang bisa mengerti banyak pertanyaan berbeda dari pelanggan?
 1. Apakah cukup kalau chatbot hanya diberi daftar pertanyaan dan jawaban?
 1. Bagaimana Deep Learning bisa membantu chatbot menjadi lebih pintar dalam memahami bahasa manusia?
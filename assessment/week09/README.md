# Tugas [T13] Ringkasan Diskusi Kelompok: Konsep dan Prinsip Contrastive Learning

**Nama Kelompok:** [Nama Kelompok]  
**Anggota:**  
1. Nama Lengkap 1 (NIM)  
2. Nama Lengkap 2 (NIM)  
3. ...  
**Deadline pengumpulan ringkasan:** 02 Agustus 2025 pukul 23:59 WIB

## 1. Pengertian Contrastive Learning
Tuliskan pengertian contrastive learning secara ringkas namun komprehensif.

## 2. Tujuan dan Prinsip Dasar
Jelaskan tujuan utama dan prinsip dasar contrastive learning (misalnya: positive/negative pairs, InfoNCE loss, dll.)

## 3. Jenis dan Variasi
Jelaskan perbedaan intra-modal dan cross-modal contrastive learning, serta contoh metodenya (SimCLR,  BYOL, dll.)

## 4. Insight Kelompok
Tuliskan pemahaman yang dianggap penting oleh kelompok dari hasil diskusi.

## 5. Refleksi Diskusi
Tuliskan refleksi kelompok: hal yang dipelajari, tantangan diskusi, dan kontribusi masing-masing anggota.

# Pengumpulan Tugas
1. Tugas dikumpulkan di LMS El-Qalam
2. Format file pdf
3. Meskipun tugas kelompok, setiap individu wajib submit tugasnya
4. Penamaan file nim_nama.pdf 
5. Ubah nama file spasi dengan underscore `(_)`


### Rubrik Penilaian Diskusi Kelompok

| **Criterion**              | **Level 1 Desc**                                         | **Score** | **Level 2 Desc**                                             | **Score** | **Level 3 Desc**                                                 | **Score** | **Level 4 Desc**                                                                 | **Score** |
|---------------------------|----------------------------------------------------------|-----------|---------------------------------------------------------------|-----------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------|-----------|
| **Kolaborasi Tim**        | Tidak ada kerja sama, dikerjakan individu                | 0         | Beberapa anggota dominan, sebagian pasif                      | 33        | Seluruh anggota berkontribusi, namun belum seimbang             | 66        | Setiap anggota berkontribusi aktif dan setara dalam kerja tim                   | 100       |
| **Pembagian Tugas**       | Tidak ada pembagian tugas                                | 0         | Ada pembagian namun tidak jelas atau tidak merata             | 33        | Pembagian tugas jelas namun kurang dokumentasi                  | 66        | Pembagian tugas jelas, adil, dan terdokumentasi dengan baik                    | 100       |
| **Kualitas Produk Akhir** | Produk tidak sesuai tujuan dan banyak kesalahan          | 0         | Produk kurang lengkap atau banyak revisi                      | 33        | Produk sesuai tujuan, ada kekurangan kecil                      | 66        | Produk sangat baik, inovatif, dan sesuai atau melebihi ekspektasi              | 100       |
| **Laporan & Komunikasi**  | Tidak ada laporan atau tidak dapat dipahami              | 0         | Laporan kurang jelas atau disusun oleh satu orang saja        | 33        | Laporan cukup jelas disusun oleh beberapa anggota               | 66        | Laporan menarik, sistematis, dan disusun oleh seluruh anggota                 | 100       |
| **Refleksi Kelompok**     | Tidak ada refleksi atau hanya satu orang                 | 0         | Refleksi hanya menyebut hasil, tidak mendalam                 | 33        | Refleksi menyeluruh tetapi belum analitis                      | 66        | Refleksi kritis, jujur, dan mengandung pembelajaran untuk proyek berikutnya   | 100       |


---

# Tugas individu [T14] Eksperimen Self-Supervised Learning dengan SimCLR
Mahasiswa mengimplementasikan metode SimCLR dengan mempelajari, menjalankan, dan memodifikasi kode SimCLR yang telah disediakan dosen, serta memahami komponen utama seperti augmentation, backbone, projection head, dan contrastive loss.

# Instruksi

Dosen akan memberikan kode dasar SimCLR (dalam format `.py` atau notebook) yang telah berfungsi dan mencakup komponen utama: augmentasi, backbone, projection head, dan InfoNCE loss.


## Dataset
Gunakan dataset ini untuk melatih model SimCLR pada dataset Tiny ImageNet di [sini](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
**Nama Tugas:** Modifikasi dan Analisis SimCLR pada Tiny ImageNet  
**Platform:** Kaggle Notebook  
**Jenis Tugas:** Implementasi Mandiri dan Analisis  
**Deadline:** 01 Agustus 2025 pukul 23:59 WIB

## Tujuan Pembelajaran

Mahasiswa mampu memahami dan memodifikasi arsitektur SimCLR berbasis kode nyata, serta mengevaluasi dampaknya terhadap performa dan representasi.

## Materi Awal

Notebook yang digunakan:  
[**SimCLR Tiny ImageNet**](https://www.kaggle.com/code/oddyvirgantara/simclr-tiny-imagenet)
Silakan salin (duplicate/fork) notebook dari dosen ke akun Kaggle Anda untuk dimodifikasi.

## Instruksi Tugas

1. **Salin notebook SimCLR** dari dosen ke akun Kaggle pribadi Anda.
2. **Pelajari dan pahami** setiap bagian kode. Tambahkan komentar penjelas di bagian:
   - Augmentasi
   - Backbone encoder
   - Projection head
   - Loss function
   - Training loop
3. **Lakukan minimal 2 modifikasi** dari daftar berikut:
   - Mengganti jenis augmentasi (misalnya `Solarize`, `Equalize`, `RandomAffine`)
   - Mengganti backbone CNN (misalnya `ResNet18` ke `ResNet34` atau `EfficientNet`)
   - Mengubah struktur atau dimensi projection head
   - Mengubah batch size atau temperature
   - Menambahkan visualisasi representasi menggunakan t-SNE atau PCA
4. **Jalankan training dan catat hasilnya**:
   - Loss curve
   - t-SNE visualisasi (jika ada)
   - Catatan hasil eksperimen
5. **Susun laporan** (maksimal 6 halaman) menggunakan format IEEE Conference Word Template, mencakup:
   - Pendahuluan
   - Penjelasan kode dasar
   - Modifikasi yang dilakukan dan alasan
   - Hasil eksperimen dan analisis
   - Refleksi pribadi

## Pengumpulan
- Unduk template laporan di [sini](https://www.ieee.org/conferences/publishing/templates)
- Pilih yang versi word A4
- File notebook `.ipynb` dengan nama `SimCLR_Modified_Nama.ipynb`
- File laporan `.pdf` sesuai format IEEE Conference
- Pastikan semua modifikasi dan komentar telah disimpan sebelum dikumpulkan.

---

## Rubrik Penilaian

| **Criterion**              | **Level 1 Desc**                                   | **Score** | **Level 2 Desc**                                             | **Score** | **Level 3 Desc**                                                 | **Score** | **Level 4 Desc**                                                                 | **Score** |
|---------------------------|----------------------------------------------------|-----------|---------------------------------------------------------------|-----------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------|-----------|
| **Pemahaman Kode**        | Tidak membaca/memahami kode dasar                  | 0         | Memahami sebagian, komentar tidak menyeluruh                 | 33        | Komentar cukup baik, sebagian belum menjelaskan logika utuh     | 66        | Komentar menyeluruh dan menjelaskan semua blok penting             | 100       |
| **Eksperimen Modifikasi** | Tidak ada modifikasi                               | 0         | Hanya 1 modifikasi tanpa analisis                            | 33        | 2 modifikasi dilakukan, hasil dijelaskan                        | 66        | ≥2 modifikasi signifikan, dikaji dan dianalisis hasilnya           | 100       |
| **Analisis Hasil**        | Tidak ada hasil                                    | 0         | Hanya menampilkan loss/hasil mentah                          | 33        | Terdapat grafik dan penjelasan hasil                            | 66        | Analisis mendalam + visualisasi representasi (t-SNE/PCA)          | 100       |
| **Refleksi dan Insight**  | Tidak ada refleksi                                 | 0         | Refleksi umum dan deskriptif saja                            | 33        | Refleksi jujur dan menyeluruh namun kurang mendalam             | 66        | Refleksi kritis, membahas tantangan dan strategi perbaikan ke depan | 100       |
| **Laporan IEEE**          | Format tidak sesuai / asal-asalan                  | 0         | Format IEEE tidak konsisten atau kurang lengkap              | 33        | Format cukup rapi, sesuai, namun visualisasi biasa              | 66        | Format IEEE rapi, visual menarik, penulisan ilmiah terstruktur    | 100       |



# Pengumpulan Tugas
1. Tugas dikumpulkan di LMS El-Qalam
2. Format file pdf
3. Meskipun tugas kelompok, setiap individu wajib submit tugasnya
4. Penamaan file nim_nama.pdf 
5. Ubah nama file spasi dengan underscore `(_)`
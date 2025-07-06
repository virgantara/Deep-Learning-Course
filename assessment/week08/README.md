# Daftar Opsi Mini Project Tugas Kelompok

Ini adalah daftar opsi topik tugas mini project yang dapat dipilih oleh masing-masing kelompok.

## 1. Klasifikasi Gambar Menggunakan CNN
Bangun model CNN dari awal untuk mengklasifikasikan dataset gambar sederhana seperti MNIST, FashionMNIST, atau CIFAR-10. Fokus pada arsitektur CNN, pelatihan model, dan evaluasi hasil.

## 2. Transfer Learning untuk Klasifikasi Gambar Khusus
Gunakan model pretrained (misalnya ResNet atau MobileNet) dan lakukan fine-tuning untuk mengklasifikasikan dataset khusus seperti makanan lokal, tanaman, atau logo. Dataset dapat dibuat sendiri.

## 3. Analisis Sentimen Teks dengan RNN/LSTM
Kembangkan model RNN atau LSTM untuk melakukan klasifikasi sentimen pada teks, misalnya review film (IMDb), review produk, atau ulasan restoran.

## 4. Image Denoising dengan Autoencoder
Bangun Autoencoder sederhana untuk menghilangkan noise dari gambar. Gunakan dataset MNIST atau FashionMNIST yang sudah ditambahkan noise sebagai input dan bandingkan hasil output-nya.

## 5. Image Generation dengan Variational Autoencoder (VAE)
Implementasikan VAE untuk menghasilkan gambar baru yang mirip dengan dataset aslinya. Dataset yang disarankan: MNIST, FashionMNIST, atau CelebA jika resource memadai.

## 6. Generative Adversarial Network (GAN) untuk Gambar Sederhana
Bangun GAN sederhana untuk menghasilkan gambar seperti tulisan tangan dari MNIST atau pakaian dari FashionMNIST. Tampilkan perkembangan hasil setiap epoch.

## 7. Time Series Forecasting dengan RNN
Gunakan RNN atau LSTM untuk memprediksi nilai masa depan dari data deret waktu, seperti suhu harian, data penjualan, atau sinyal sensor.

## 8. Deteksi Wajah Bermasker dengan Transfer Learning
Bangun sistem klasifikasi untuk membedakan wajah yang memakai masker dan yang tidak, menggunakan pretrained CNN dan dataset open-source seperti Face Mask Detection.

## 9. Anomaly Detection Menggunakan Autoencoder
Gunakan Autoencoder untuk mendeteksi anomali (outlier) dalam data numerik, seperti data transaksi keuangan, data IoT, atau log sistem.

## 10. Style Transfer dengan Pretrained CNN
Implementasikan style transfer untuk mentransfer gaya dari lukisan ke foto menggunakan pretrained VGG. Proyek ini bisa dikembangkan lebih jauh untuk aplikasi seni atau edukasi.

---

> Catatan: Setelah memilih topik, setiap kelompok wajib membuat repositori GitHub dan mengembangkan proyeknya dalam bentuk notebook yang lengkap, berjalan sempurna, dan terdokumentasi dengan baik. Riwayat commit akan digunakan sebagai bukti kontribusi individu.

# Penilaian Proyek Mini
Evaluasi Proyek mini dinilai dalam dua bentuk, *laporan* dan *presentasi* 

## Evaluasi Laporan
### Deskripsi
Setiap kelompok wajib menyerahkan laporan akhir proyek mini dalam format PDF (boleh ditulis di Jupyter Notebook dan diekspor, atau disusun terpisah). Laporan digunakan untuk mengevaluasi pemahaman, proses, dan hasil kerja kelompok secara menyeluruh.

Wajib menggunakan template IEEE Conference. Klik di [sini](https://www.ieee.org/conferences/publishing/templates) untuk mengunduh. Cari yang formatnya A4 ekstensi doc.

### Format
Format Laporan:
1. Halaman Judul: Judul proyek, nama dan NIM seluruh anggota kelompok.
2. Pendahuluan: Latar belakang, perumusan masalah, dan tujuan proyek.
3. Metodologi: Penjelasan arsitektur/model yang digunakan, dataset, preprocessing, dan pendekatan pelatihan.
4. Implementasi: Penjabaran proses training, library yang digunakan, eksperimen, dan tantangan teknis.
5. Hasil dan Evaluasi: Visualisasi hasil, analisis performa model, interpretasi, serta perbandingan jika ada.
6. Kesimpulan dan Saran: Ringkasan temuan, kelebihan/kekurangan model, dan saran pengembangan.
7. Referensi: Daftar pustaka jika ada.
8. Lampiran: Link GitHub, screenshot hasil training, dsb.

### Keterangan
- Laporan ditulis dengan rapi, sistematis, dan menggunakan bahasa ilmiah yang jelas.
- Diserahkan dalam bentuk PDF sebelum batas waktu yang ditentukan.
- Laporan akan dinilai berdasarkan rubrik berikut:
### Rubrik

| **Criterion**              | **Level 1 Desc**                                         | **Score** | **Level 2 Desc**                                             | **Score** | **Level 3 Desc**                                                 | **Score** | **Level 4 Desc**                                                                 | **Score** |
|---------------------------|----------------------------------------------------------|-----------|---------------------------------------------------------------|-----------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------|-----------|
| **Kolaborasi Tim**        | Tidak ada kerja sama, dikerjakan individu                | 0         | Beberapa anggota dominan, sebagian pasif                      | 33        | Seluruh anggota berkontribusi, namun belum seimbang             | 66        | Setiap anggota berkontribusi aktif dan setara dalam kerja tim                   | 100       |
| **Pembagian Tugas**       | Tidak ada pembagian tugas                                | 0         | Ada pembagian namun tidak jelas atau tidak merata             | 33        | Pembagian tugas jelas namun kurang dokumentasi                  | 66        | Pembagian tugas jelas, adil, dan terdokumentasi dengan baik                    | 100       |
| **Kualitas Produk Akhir** | Produk tidak sesuai tujuan dan banyak kesalahan          | 0         | Produk kurang lengkap atau banyak revisi                      | 33        | Produk sesuai tujuan, ada kekurangan kecil                      | 66        | Produk sangat baik, inovatif, dan sesuai atau melebihi ekspektasi              | 100       |
| **Laporan & Komunikasi**  | Tidak ada laporan atau tidak dapat dipahami              | 0         | Laporan kurang jelas atau disusun oleh satu orang saja        | 33        | Laporan cukup jelas disusun oleh beberapa anggota               | 66        | Laporan menarik, sistematis, dan disusun oleh seluruh anggota                 | 100       |
| **Refleksi Kelompok**     | Tidak ada refleksi atau hanya satu orang                 | 0         | Refleksi hanya menyebut hasil, tidak mendalam                 | 33        | Refleksi menyeluruh tetapi belum analitis                      | 66        | Refleksi kritis, jujur, dan mengandung pembelajaran untuk proyek berikutnya   | 100       |

## Evaluasi Presentasi
### Deskripsi
Setiap kelompok wajib melakukan presentasi untuk memaparkan hasil proyek mini yang telah mereka kerjakan. Presentasi merupakan bentuk evaluasi lisan dan visual yang mencerminkan:
- Pemahaman terhadap topik dan metode yang digunakan.
- Kolaborasi dan pembagian peran dalam kelompok.
- Kemampuan menyampaikan ide, hasil, dan analisis secara runtut dan jelas.

### Ketentuan Presentasi
Ketentuan Presentasi

- Durasi presentasi maksimal 10 menit, dilanjutkan dengan sesi tanya-jawab 5 menit.
- Seluruh anggota kelompok wajib terlibat aktif, minimal dua orang menyampaikan materi.
- Media presentasi dapat berupa PowerPoint/Google Slides, PDF, atau presentasi langsung dari Jupyter Notebook.
- Materi yang wajib disampaikan:
	- Latar belakang dan tujuan proyek
	- Metodologi (arsitektur model, dataset, preprocessing)
	- Hasil eksperimen dan visualisasi
	- Analisis dan kesimpulan
	- Tautan GitHub dan pembagian peran

- Bahasa yang digunakan adalah bahasa Indonesia yang baik dan benar (atau bahasa Inggris jika disyaratkan oleh dosen).

### Rubrik Penilaian Presentasi Kelompok
| **Criterion**              | **Level 1 Desc**                                         | **Score** | **Level 2 Desc**                                             | **Score** | **Level 3 Desc**                                                 | **Score** | **Level 4 Desc**                                                                 | **Score** |
|---------------------------|----------------------------------------------------------|-----------|---------------------------------------------------------------|-----------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------|-----------|
| **Kolaborasi Tim**        | Tidak ada kerja sama, dikerjakan individu                | 0         | Beberapa anggota dominan, sebagian pasif                      | 33        | Seluruh anggota berkontribusi, namun belum seimbang             | 66        | Setiap anggota berkontribusi aktif dan setara dalam kerja tim                   | 100       |
| **Pembagian Tugas**       | Tidak ada pembagian tugas                                | 0         | Ada pembagian namun tidak jelas atau tidak merata             | 33        | Pembagian tugas jelas namun kurang dokumentasi                  | 66        | Pembagian tugas jelas, adil, dan terdokumentasi dengan baik                    | 100       |
| **Kualitas Produk Akhir** | Produk tidak sesuai tujuan dan banyak kesalahan          | 0         | Produk kurang lengkap atau banyak revisi                      | 33        | Produk sesuai tujuan, ada kekurangan kecil                      | 66        | Produk sangat baik, inovatif, dan sesuai atau melebihi ekspektasi              | 100       |
| **Presentasi & Komunikasi**  | Tidak ada presentasi atau tidak dapat dipahami              | 0         | Presentasi kurang jelas atau disusun oleh satu orang saja        | 33        | Presentasi cukup jelas disusun oleh beberapa anggota               | 66        | Presentasi menarik, sistematis, dan disusun oleh seluruh anggota                 | 100       |
| **Refleksi Kelompok**     | Tidak ada refleksi atau hanya satu orang                 | 0         | Refleksi hanya menyebut hasil, tidak mendalam                 | 33        | Refleksi menyeluruh tetapi belum analitis                      | 66        | Refleksi kritis, jujur, dan mengandung pembelajaran untuk proyek berikutnya   | 100       |

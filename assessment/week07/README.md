# Tugas Kelompok: Implementasi Generative Adversarial Network (GAN)

## Tujuan Tugas
Tugas ini bertujuan untuk:
- Memahami konsep dasar Generative Adversarial Network (GAN).
- Mengimplementasikan GAN untuk menghasilkan gambar dari data sederhana.
- Melatih kerja sama tim dalam menyusun proyek berbasis deep learning dan mendokumentasikan hasilnya.

## Ketentuan Kelompok
- Kelompok terdiri dari 3–5 orang mahasiswa.
- Setiap anggota wajib berkontribusi, dibuktikan melalui riwayat commit GitHub.
- Penilaian memperhatikan kualitas hasil, dokumentasi, serta distribusi kontribusi.

## Deskripsi Tugas
Setiap kelompok diminta untuk:
1. Mempelajari konsep GAN secara singkat dan menuliskannya di awal notebook.
2. Mengimplementasikan GAN sederhana menggunakan **PyTorch atau TensorFlow**.
3. Melatih GAN pada dataset sederhana, seperti:
   - MNIST
   - FashionMNIST
   - CIFAR-10 (opsional untuk tantangan)
4. Menyimpan dan menampilkan hasil gambar yang dihasilkan selama proses training.

## Spesifikasi Teknis
- Arsitektur GAN minimal terdiri dari:
  - **Generator**: Menerima noise (vektor acak) dan menghasilkan gambar.
  - **Discriminator**: Membedakan antara gambar nyata dan hasil generator.
- Notebook harus berisi:
  - Penjelasan setiap bagian (dalam markdown cell).
  - Visualisasi hasil loss selama training.
  - Contoh gambar hasil generator tiap beberapa epoch.
- Disarankan menyimpan model `.pth` (jika PyTorch) atau `.h5` (jika TensorFlow).

## Output yang Dikumpulkan
- Repositori GitHub publik yang berisi:
  - `gan_<nama_kelompok>.ipynb`: Notebook Jupyter yang sudah dijalankan (lengkap dengan output).
  - Folder `generated_images/`: Menyimpan beberapa contoh hasil dari generator.
  - File `README.md`: Deskripsi singkat tugas, anggota kelompok, dan cara menjalankan kode.
  - Riwayat commit mencerminkan kontribusi setiap anggota.

## Struktur Direktori yang Disarankan
```bash
/nama-kelompok-gan/
├── gan_kelompokX.ipynb
├── generated_images/
│ ├── epoch_10.png
│ ├── epoch_20.png
│ └── ...
├── models/ untuk menyimpan file .pth/.h5
├── README.md
```


## Catatan Penting
- Semua hasil gambar harus dapat ditampilkan dalam notebook.
- Gunakan fungsi visualisasi loss dan hasil generator secara berkala.
- Jelaskan perbedaan hasil antara epoch awal dan akhir secara ringkas di akhir notebook.

## Tenggat Waktu
Tugas harus dikumpulkan paling lambat: **[Tanggal XX/XX/2025, Pukul 23:59 WIB]**

# Rubrik
# Rubrik Penilaian Tugas Kelompok

| **Criterion**              | **Level 1 Desc**                                         | **Score** | **Level 2 Desc**                                             | **Score** | **Level 3 Desc**                                                 | **Score** | **Level 4 Desc**                                                                 | **Score** |
|---------------------------|----------------------------------------------------------|-----------|---------------------------------------------------------------|-----------|------------------------------------------------------------------|-----------|----------------------------------------------------------------------------------|-----------|
| **Kolaborasi Tim**        | Tidak ada kerja sama, dikerjakan individu                | 0         | Beberapa anggota dominan, sebagian pasif                      | 33        | Seluruh anggota berkontribusi, namun belum seimbang             | 66        | Setiap anggota berkontribusi aktif dan setara dalam kerja tim                   | 100       |
| **Pembagian Tugas**       | Tidak ada pembagian tugas                                | 0         | Ada pembagian namun tidak jelas atau tidak merata             | 33        | Pembagian tugas jelas namun kurang dokumentasi                  | 66        | Pembagian tugas jelas, adil, dan terdokumentasi dengan baik                    | 100       |
| **Kualitas Produk Akhir** | Produk tidak sesuai tujuan dan banyak kesalahan          | 0         | Produk kurang lengkap atau banyak revisi                      | 33        | Produk sesuai tujuan, ada kekurangan kecil                      | 66        | Produk sangat baik, inovatif, dan sesuai atau melebihi ekspektasi              | 100       |
| **Laporan & Komunikasi**  | Tidak ada laporan atau tidak dapat dipahami              | 0         | Laporan kurang jelas atau disusun oleh satu orang saja        | 33        | Laporan cukup jelas disusun oleh beberapa anggota               | 66        | Laporan menarik, sistematis, dan disusun oleh seluruh anggota                 | 100       |
| **Refleksi Kelompok**     | Tidak ada refleksi atau hanya satu orang                 | 0         | Refleksi hanya menyebut hasil, tidak mendalam                 | 33        | Refleksi menyeluruh tetapi belum analitis                      | 66        | Refleksi kritis, jujur, dan mengandung pembelajaran untuk proyek berikutnya   | 100       |

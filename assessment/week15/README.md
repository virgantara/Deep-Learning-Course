# Tugas Individu: Machine Translation dengan PyTorch  
**Mata Kuliah:** Pembelajaran Mesin 2  
**Topik:** Penerapan Deep Learning dalam NLP dan Skenario Nyata (Machine Translation)

---

## Tujuan Pembelajaran
- Membangun sistem penerjemah otomatis berbasis deep learning.  
- Membandingkan baseline RNN+Attention dengan Transformer (wajib).  
- Mengevaluasi performa dengan metrik standar (mis. SacreBLEU, chrF) serta menulis laporan ilmiah sesuai format IEEE.  

---

## Setup
- Siapkan environment Python dengan dependensi sesuai kebutuhan.  
- Gunakan dataset penerjemahan bilingual (misalnya EN–ID atau pasangan bahasa lain yang relevan).  
- Lakukan preprocessing: pembersihan data, tokenisasi subword (BPE/SentencePiece), serta pembagian data train/valid/test.  
- Dataset diambil dari https://www.manythings.org/anki/

---

## Tugas yang Harus Dilakukan

### 1) Persiapan Data
- Unduh dan bersihkan dataset.  
- Lakukan tokenisasi subword dan simpan model vocab.  
- Buat pembagian data train/valid/test dengan rasio yang jelas.  

### 2) Implementasi Baseline (RNN + Attention)
- Implementasikan arsitektur encoder–decoder dengan attention.  
- Latih model hingga konvergen, catat metrik pelatihan dan validasi.  

### 3) Implementasi Transformer (Wajib)
- Implementasikan arsitektur encoder–decoder Transformer.  
- Gunakan teknik tambahan seperti label smoothing atau warmup learning rate (opsional).  
- Bandingkan hasilnya dengan baseline.  

### 4) Evaluasi & Analisis
- Gunakan metrik SacreBLEU (wajib) dan chrF (opsional).  
- Sajikan contoh hasil terjemahan dan analisis kesalahan.  
- Lakukan ablation study minimal satu variabel (misalnya ukuran vocab, dropout, beam size).  

### 5) Laporan
- Tulis laporan dengan format IEEE Conference (Word/LaTeX).  
- Struktur laporan minimal mencakup: Abstract, Introduction, Related Work, Method, Experiments, Results & Discussion, Conclusion, References.  
- Sertakan tabel metrik, kurva pelatihan, dan analisis hasil.  

---

## Aturan
- Wajib menggunakan PyTorch.  
- Dilarang menggunakan model siap pakai tanpa penjelasan.  
- Semua sumber data atau kode pihak ketiga harus dicantumkan secara jelas.  
---

## Pengumpulan
- Kumpulkan repository berisi kode dan dokumentasi berupa tautan github.  
- Lampirkan laporan dalam format PDF IEEE Conference.  
- Sertakan hasil eksperimen (checkpoints, logs, grafik, evaluasi).  
- Penamaan file laporan nama_mahasiswa_nim.pdf
- Penamaan file tanpa ada spasi.
- Upload di LMS El-Qalam
---

## Rubrik Penilaian (Total 100%)
| Aspek Penilaian | Bobot | Tingkatan (4) | Tingkatan (3) | Tingkatan (2) | Tingkatan (1) |
|-----------------|-------|---------------|---------------|---------------|---------------|
| **Pemahaman & Formulasi Masalah** | 10% | Sangat jelas dan kontekstual (10%) | Cukup jelas (8%) | Umum dan dangkal (5%) | Tidak tepat (2%) |
| **Data & Preprocessing** | 15% | Lengkap, valid, dan terdokumentasi (15%) | Sebagian lengkap (12%) | Minim dokumentasi (8%) | Tidak jelas (4%) |
| **Implementasi Baseline** | 15% | Lengkap dan stabil (15%) | Ada minor kekurangan (12%) | Tidak stabil (8%) | Tidak berjalan (4%) |
| **Implementasi Transformer** | 30% | Lengkap, stabil, hasil baik (30%) | Hasil cukup (24%) | Hasil marginal (16%) | Tidak berjalan (8%) |
| **Evaluasi & Analisis** | 25% | Lengkap, mendalam, dengan ablation (25%) | Cukup lengkap (19%) | Dangkal (12%) | Minim evaluasi (6%) |
| **Laporan IEEE** | 5% | Rapi dan sesuai format (5%) | Minor kesalahan (4%) | Banyak kesalahan (3%) | Tidak sesuai (1%) |

---

## Studi Kasus
- Pilih pasangan bahasa (disarankan EN–ID).  
- Jelaskan alasan pemilihan dataset.  
- Identifikasi tantangan khusus (morfologi, OOV, idiom, dll.).  
- Bandingkan hasil baseline dengan Transformer dan berikan analisis kesalahan.  

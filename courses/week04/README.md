# Chapter: Transfer Learning dan Optimasi Model

Bab ini membahas pendekatan lanjutan dalam pengembangan model deep learning yang lebih efisien, adaptif, dan unggul dalam menghadapi keterbatasan data serta tantangan overfitting. Fokus utama bab ini adalah pada Transfer Learning (TL)—sebuah strategi yang memungkinkan model memanfaatkan pengetahuan dari domain lain untuk mempercepat pelatihan dan meningkatkan performa pada tugas baru.

## Tujuan Pembelajaran

- Memahami konsep dasar Transfer Learning dan relevansinya dalam deep learning modern
- Membedakan domain sumber dan domain target serta jenis adaptasi domain (SDA, SSDA, UDA)
- Menerapkan strategi fine-tuning dan ekstraksi fitur dalam arsitektur jaringan
- Memahami formulasi matematis dalam proses transfer dan evaluasi fitur
- Mengimplementasikan pre-training dan fine-tuning menggunakan PyTorch

## Instalasi library
```cmd
pip install -r requirements.txt
```

## 1. Konsep Dasar Transfer Learning
- Keterbatasan deep learning konvensional
- Pentingnya pengetahuan awal dalam TL

## 2. Domain dalam Transfer Learning
- Definisi domain dalam pembelajaran mesin
- Jenis domain: sumber vs target
- Adaptasi domain: supervised, semi-supervised, unsupervised

## 3. Strategi Transfer Learning
- Ekstraksi fitur umum dari lapisan dangkal
- Fine-tuning pada lapisan atas
- Pertimbangan dalam memilih lapisan yang dibekukan dan disesuaikan

## 4. Formulasi Matematis TL
- Representasi transfer dan fungsi transformasi
- Fungsi loss kanal dengan pembobotan pentingnya fitur
- Strategi single, one-to-one, all-to-all transfer

## 5. Model Pre-Trained sebagai Ekstraktor Fitur
- Perbandingan fitur manual (misal: SIFT) vs fitur dari deep learning
- Kombinasi teknik klasik dan modern (misal: EasyTL)

## 6. Implementasi di PyTorch
### Dataset
1. Silakan unduh dataset dari [sini](https://drive.google.com/file/d/1C9VNnnQb9petAcc0AoiTV_F_Pc8oqXfI/view?usp=drive_link)
1. Taruh di dalam folder data yang ada di direktori week04
1. Contoh kode fine-tuning dan ekstraksi fitur di: https://github.com/virgantara/Deep-Learning-Course/tree/master/courses/week04



## Kesimpulan

Transfer Learning memberikan solusi praktis untuk membangun model yang lebih efisien, tahan terhadap overfitting, dan cepat beradaptasi pada domain baru. Melalui bab ini, pembaca dibekali landasan teori, rumus matematis, serta contoh implementasi yang dapat diterapkan langsung dalam berbagai skenario nyata.

Selanjutnya: Kita akan membahas penerapan teknik TL pada berbagai studi kasus dan proyek praktis.

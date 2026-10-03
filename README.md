# P2 Transfer Learning menggunakan model Resnet 18 untuk klasifikasi motor dan mobil untuk Security Patrol Robot

Nur Hafidzatun Nabila | 4222401012 | RET503 Computer Vision and Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/p2-transfer-learning/blob/main/P2_transfer_learning_colab.ipynb)

Security Patrol Robot harus bisa membedakan motor dan mobil saat berpatroli. Model yang dipakai ResNet-18 (pretrained ImageNet).

feature: cuma lapisan terakhir (fc) yang dilatih
partial: layer4 + fc yang dilatih
scratch: dilatih dari nol, tanpa bobot ImageNet

**Dataset:** 114 foto dari internet (mobil 54, motor 60), ada di `dataset_raw/`. Dibagi 80/10/10: train 92, val 11, test 11.

## Hasil

| Mode | Val terbaik | Test | Waktu latih |
|---|---|---|---|
| feature | 100% | 100% (11/11) | 31 s |
| partial | 100% | 100% (11/11) | 48 s |
| scratch | 90,9% | 90,9% (10/11) | 83 s |

Di data test, feature dan partial bener semua, scratch salah 1 (mobil kebaca motor).
Hasil di atas dijalanin di laptop (CPU). Di Colab pola hasilnya sama, tapi angkanya bisa beda dikit (terutama scratch) karena latihnya ada unsur acak.
Latensi ResNet-18: sekitar 30 ms per foto (sekitar 33 foto/detik) di CPU laptop. 

## Kesimpulan

- Feature dan partial sama-sama 100%, scratch lebih rendah dan naik turun. Jadi pakai bobot ImageNet jelas lebih bagus kalau datanya sedikit.
- Kami pilih feature. Hasilnya sama kayak partial, tapi yang dilatih cuma 1.026 parameter dan lebih cepat (31 s vs 48 s).
- val dan test cuma 11 foto, jadi salah satu foto aja sudah mengubah akurasi sekitar 9%. Selisih tipis jangan dianggap pasti. 

## Cara Menjalankan

Klik **Open in Colab**, ganti `GITHUB_USER` di sel 2 dengan username GitHub, run semua sel dari atas ke bawah.

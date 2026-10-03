# P2 Transfer Learning: ResNet-18, klasifikasi mobil vs motor (RET503)

Kelas: `mobil` dan `motor`. Urutan indeks di PyTorch mengikuti abjad: mobil = 0, motor = 1.

## Isi folder

```
P2_transfer_learning/
├── dataset_patrol.zip      <- taruh zip fotomu di sini (jangan diekstrak)
├── prepare_data.py
├── train.py
├── latency.py
├── plot_results.py
└── README.md
```

## Persiapan (sekali saja)

Python 3.12 atau 3.13 disarankan. Pasang pustaka:

```
python -m pip install torch torchvision pillow pandas matplotlib
```

## Langkah (ketik di terminal, jangan klik tombol Run)

```
python prepare_data.py --zip dataset_patrol.zip   # rapikan foto, buat metadata.csv, bagi 80/10/10
python train.py --mode feature                    # fc saja, lr 1e-3
python train.py --mode partial                    # layer4 + fc, lr 1e-4 / 1e-3
python train.py --mode scratch                    # bobot acak, semua dilatih
python latency.py                                 # ResNet-18 vs MobileNetV3-Small
python plot_results.py                            # tabel_hasil.md + akurasi_per_epoch.png
```

Pertama kali menjalankan `feature` atau `partial`, torchvision mengunduh bobot ImageNet (sekitar 45 MB).
Jika tidak ada internet, unduh `resnet18-f37072fd.pth` lalu tambahkan `--weights-path resnet18-f37072fd.pth`.

Jika tiap anggota kelompok menjalankan satu mode, kumpulkan `results/<mode>_log.csv` dan
`results/<mode>_summary.json` dari semua anggota ke satu folder `results/`, lalu jalankan `plot_results.py`.
Pakai `data/` yang sama (hasil satu kali `prepare_data.py`, seed 42) supaya hasilnya sebanding.

## Pembagian data: 80 / 10 / 10

| Bagian | Fungsi |
|---|---|
| train (80%) | model belajar dari sini (bobot diubah) |
| val (10%) | memilih epoch terbaik tiap epoch (bobot tidak diubah) |
| test (10%) | nilai akhir; dipakai sekali setelah training selesai, dengan bobot epoch terbaik |

Pembagian dilakukan per kelas. Foto yang hampir sama (hash perseptual) tidak dipisah ke bagian yang berbeda.

## Keluaran

| File | Isi |
|---|---|
| `dataset_raw/metadata.csv` | nama_file, kelas, tanggal, kondisi_cahaya, sumber, file_asli, lebar, tinggi, split |
| `results/<mode>_log.csv` | loss dan akurasi per epoch |
| `results/<mode>_summary.json` | akurasi val terbaik, akurasi test, confusion matrix, waktu latih, epoch pertama val >= 90% |
| `results/latency.csv` | latensi dan FPS tiap model |
| `results/tabel_hasil.md`, `results/akurasi_per_epoch.png` | untuk README laporan |

## Catatan data

- Foto berasal dari internet (dataset_patrol), bukan dari kamera robot. Sumber dicatat di kolom `sumber`.
- Dataset kecil (mobil 54, motor 60), jadi val dan test masing-masing hanya sekitar 11 foto. Satu foto sama dengan sekitar 9% akurasi. Selisih beberapa persen antar mode bisa hanya kebetulan.
- Ukuran foto sangat timpang (mobil kecil, motor besar). Foto di `data/` diperkecil (sisi terpanjang maks. 800 px) dan diubah ke 224x224 saat training, tetapi tetap ada kemungkinan model memakai petunjuk yang tidak berhubungan dengan jenis kendaraan (kualitas gambar, latar, gaya foto). Bahas ini di analisis.
- Akurasi tinggi pada foto internet belum menjamin model bekerja di kamera robot.

"""
plot_results.py - gabungkan hasil feature / partial / scratch menjadi:
  results/akurasi_per_epoch.png   (grafik untuk README)
  results/tabel_hasil.md          (tabel untuk README)

Mode yang belum dijalankan dilewati, jadi bisa dipakai juga kalau tiap anggota
kelompok menjalankan satu mode: kumpulkan semua file <mode>_log.csv dan
<mode>_summary.json ke satu folder results/, lalu jalankan skrip ini.

Pemakaian:
  python plot_results.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE = ["feature", "partial", "scratch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()
    d = Path(args.dir)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ringkasan = []
    for mode in MODE:
        log, ringkas = d / f"{mode}_log.csv", d / f"{mode}_summary.json"
        if not (log.exists() and ringkas.exists()):
            print(f"(lewati {mode}: file hasil belum ada)")
            continue
        with open(log) as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        ax[0].plot(ep, [float(r["val_acc"]) for r in rows], marker="o", label=mode)
        ax[1].plot(ep, [float(r["train_acc"]) for r in rows], marker="o", label=mode)
        ringkasan.append(json.load(open(ringkas)))

    ax[0].set_title("Akurasi validasi per epoch")
    ax[1].set_title("Akurasi train per epoch")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylabel("Akurasi")
        a.set_ylim(0, 1.02)
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(d / "akurasi_per_epoch.png", dpi=150)

    with open(d / "tabel_hasil.md", "w") as f:
        f.write("| Mode | Akurasi val terbaik | Epoch terbaik | Epoch pertama val >= 90% "
                "| Akurasi test | Waktu latih (s) | Parameter dilatih |\n|---|---|---|---|---|---|---|\n")
        for s in ringkasan:
            n_test = s.get("jumlah_test")
            t = s.get("akurasi_test")
            teks_test = "-" if t is None else f"{t*100:.1f}% ({round(t*n_test)}/{n_test})"
            f.write(f"| {s['mode']} | {s['akurasi_val_terbaik']*100:.1f}% | {s['epoch_terbaik']} | "
                    f"{s['epoch_pertama_val_90'] or 'tidak tercapai'} | {teks_test} | "
                    f"{s['waktu_pelatihan_s']} | {s['parameter_dilatih']:,} |\n")
    print((d / "tabel_hasil.md").read_text())
    print(f"Grafik: {d / 'akurasi_per_epoch.png'}")


if __name__ == "__main__":
    main()

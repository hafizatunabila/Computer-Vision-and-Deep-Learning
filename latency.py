"""
latency.py - ukur latensi inferensi ResNet-18 (langkah 4 praktikum).

Latensi tidak bergantung pada nilai bobot, jadi bobot acak sudah cukup (tanpa unduh).
Jalankan di perangkat yang mewakili robot (Raspberry Pi / Jetson) kalau bisa;
angka di laptop hanya perbandingan relatif.

Yang diukur: satu citra 1x3x224x224, hanya inferensi model (belum termasuk
akuisisi kamera, undistort, preprocessing, dan postprocessing).

Pemakaian:
  python latency.py
  python latency.py --device cuda --runs 200
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from torchvision import models


def ukur(m, device, warmup, runs):
    m.eval().to(device)
    x = torch.randn(1, 3, 224, 224, device=device)
    waktu = []
    with torch.inference_mode():
        for _ in range(warmup):                 # pemanasan: iterasi awal selalu lebih lambat
            m(x)
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            waktu.append((time.perf_counter() - t) * 1000)
    waktu.sort()
    return dict(rata2_ms=statistics.mean(waktu), median_ms=statistics.median(waktu),
                p95_ms=waktu[int(0.95 * len(waktu)) - 1], min_ms=waktu[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=None, help="jumlah thread CPU (opsional)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    kandidat = {
        "ResNet-18": models.resnet18(weights=None),
    }

    hasil = []
    print(f"Device: {device} | thread CPU: {torch.get_num_threads()} | "
          f"warmup {args.warmup}, ukur {args.runs}x\n")
    print(f"{'Model':20s} {'Param (juta)':>12s} {'Rata2 (ms)':>11s} {'Median':>8s} {'P95':>8s} {'FPS':>7s}")
    for nama, m in kandidat.items():
        r = ukur(m, device, args.warmup, args.runs)
        r["model"] = nama
        r["parameter_juta"] = sum(p.numel() for p in m.parameters()) / 1e6
        r["fps"] = 1000 / r["rata2_ms"]
        hasil.append(r)
        print(f"{nama:20s} {r['parameter_juta']:12.2f} {r['rata2_ms']:11.2f} "
              f"{r['median_ms']:8.2f} {r['p95_ms']:8.2f} {r['fps']:7.1f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kolom = ["model", "parameter_juta", "rata2_ms", "median_ms", "p95_ms", "min_ms", "fps"]
    with open(out / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        w.writerows(hasil)
    print(f"\nTersimpan: {out / 'latency.csv'}")


if __name__ == "__main__":
    main()
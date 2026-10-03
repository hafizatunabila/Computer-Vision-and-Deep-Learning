"""
train.py - ResNet-18 untuk klasifikasi mobil vs motor, tiga mode (slide 21):

  feature : bobot ImageNet, hanya fc dilatih,            lr 1e-3
  partial : bobot ImageNet, layer4 + fc dilatih,         lr 1e-4 (layer4) / 1e-3 (fc)
  scratch : bobot acak, semua lapisan dilatih,           lr 1e-3

Augmentasi hanya untuk train: RandomResizedCrop, HorizontalFlip, ColorJitter.
10 epoch, Adam, CosineAnnealingLR.

Peran tiga bagian data:
  train : dipakai untuk BELAJAR (bobot diubah di sini)
  val   : dipakai untuk MEMILIH epoch terbaik tiap epoch (bobot tidak diubah)
  test  : dipakai SEKALI di akhir, dengan bobot dari epoch terbaik, untuk nilai akhir yang jujur

Yang dicatat per mode (folder results/):
  <mode>_log.csv      : loss & akurasi per epoch (untuk grafik)
  <mode>_summary.json : akurasi val terbaik, akurasi TEST, waktu, epoch pertama val >= 90%, dll.
  <mode>_best.pt      : bobot epoch terbaik

Pemakaian:
  python train.py --mode feature
  python train.py --mode partial
  python train.py --mode scratch
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]   # statistik ImageNet (slide 13)
STD = [0.229, 0.224, 0.225]

# bagian ResNet yang dibekukan pada tiap mode
BEKU = {
    "feature": {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4"},
    "partial": {"conv1", "bn1", "layer1", "layer2", "layer3"},
    "scratch": set(),
}


def pilih_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def buat_loader(data_dir, batch, workers):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # val dan test: preprocessing yang sama persis dengan saat dipakai di robot (tanpa augmentasi)
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    d = Path(data_dir)
    ds_train = datasets.ImageFolder(d / "train", tf_train)
    ds_val = datasets.ImageFolder(d / "val", tf_eval)
    ds_test = datasets.ImageFolder(d / "test", tf_eval)
    assert ds_train.classes == ds_val.classes == ds_test.classes, "kelas di train/val/test tidak sama"
    kw = dict(num_workers=workers, persistent_workers=workers > 0)
    return (DataLoader(ds_train, batch, shuffle=True, **kw),
            DataLoader(ds_val, batch, shuffle=False, **kw),
            DataLoader(ds_test, batch, shuffle=False, **kw),
            ds_train.classes)


def buat_model(mode, n_kelas, pretrained=True, weights_path=None):
    """Kembalikan (model, optimizer)."""
    if mode == "scratch":
        m = models.resnet18(weights=None)
    else:
        if weights_path:                         # bobot ImageNet dari file lokal
            m = models.resnet18(weights=None)
            m.load_state_dict(torch.load(weights_path, map_location="cpu"))
        elif pretrained:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:                                    # hanya untuk uji skrip tanpa internet
            m = models.resnet18(weights=None)

    m.fc = nn.Linear(m.fc.in_features, n_kelas)  # head baru: otomatis bisa dilatih

    for nama, anak in m.named_children():         # bekukan bagian yang tidak dilatih
        if nama in BEKU[mode]:
            for p in anak.parameters():
                p.requires_grad = False

    if mode == "feature":
        opt = torch.optim.Adam(m.fc.parameters(), lr=1e-3)
    elif mode == "partial":
        opt = torch.optim.Adam([
            {"params": m.layer4.parameters(), "lr": 1e-4},
            {"params": m.fc.parameters(), "lr": 1e-3},
        ])
    else:
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    return m, opt


def mode_train(m, mode):
    """m.train() tetapi lapisan beku tetap eval() agar statistik BatchNorm ImageNet tidak berubah."""
    m.train()
    for nama, anak in m.named_children():
        if nama in BEKU[mode]:
            anak.eval()


@torch.no_grad()
def evaluasi(m, loader, loss_fn, device, n_kelas):
    m.eval()
    total_loss, benar, n = 0.0, 0, 0
    conf = [[0] * n_kelas for _ in range(n_kelas)]   # baris = kelas asli, kolom = tebakan
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = m(x)
        pred = out.argmax(1)
        total_loss += loss_fn(out, y).item() * x.size(0)
        benar += (pred == y).sum().item()
        n += x.size(0)
        for a, b in zip(y.tolist(), pred.tolist()):
            conf[a][b] += 1
    return total_loss / n, benar / n, conf


def run(args, pretrained=True):
    torch.manual_seed(args.seed)                  # seed sama -> perbandingan antar mode adil
    device = pilih_device()
    tr, va, te, kelas = buat_loader(args.data, args.batch, args.workers)
    print(f"Mode: {args.mode} | device: {device} | kelas: {kelas} "
          f"| train: {len(tr.dataset)} | val: {len(va.dataset)} | test: {len(te.dataset)}")

    m, opt = buat_model(args.mode, len(kelas), pretrained, args.weights_path)
    m.to(device)
    n_latih = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"Parameter dilatih: {n_latih:,} dari {n_total:,}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    terbaik, loss_terbaik, ep_terbaik, ep_90 = -1.0, float("inf"), 0, None
    t_mulai = time.time()

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        mode_train(m, args.mode)
        tot_loss, benar, n = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            o = m(x)
            loss = loss_fn(o, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
            benar += (o.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()
        v_loss, v_acc, _ = evaluasi(m, va, loss_fn, device, len(kelas))
        baris = dict(epoch=ep, train_loss=tot_loss / n, train_acc=benar / n,
                     val_loss=v_loss, val_acc=v_acc, waktu_epoch_s=time.time() - t0)
        log.append(baris)
        print(f"  epoch {ep:2d}/{args.epochs} | train {baris['train_loss']:.3f}/{baris['train_acc']:.3f}"
              f" | val {v_loss:.3f}/{v_acc:.3f} | {baris['waktu_epoch_s']:.1f}s")

        # epoch terbaik = akurasi val tertinggi; kalau seri, pilih val loss yang lebih kecil
        if v_acc > terbaik or (v_acc == terbaik and v_loss < loss_terbaik):
            terbaik, loss_terbaik, ep_terbaik = v_acc, v_loss, ep
            torch.save(m.state_dict(), out / f"{args.mode}_best.pt")
        if ep_90 is None and v_acc >= 0.90:
            ep_90 = ep

    total_s = time.time() - t_mulai

    # ujian akhir: test set dipakai SEKALI, dengan bobot epoch terbaik
    m.load_state_dict(torch.load(out / f"{args.mode}_best.pt", map_location=device))
    t_loss, t_acc, conf = evaluasi(m, te, loss_fn, device, len(kelas))

    with open(out / f"{args.mode}_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)
    ringkas = dict(mode=args.mode, kelas=kelas, epochs=args.epochs,
                   akurasi_val_terbaik=terbaik, epoch_terbaik=ep_terbaik,
                   epoch_pertama_val_90=ep_90,
                   akurasi_test=t_acc, loss_test=t_loss,
                   confusion_test=conf,
                   jumlah_train=len(tr.dataset), jumlah_val=len(va.dataset), jumlah_test=len(te.dataset),
                   waktu_pelatihan_s=round(total_s, 1),
                   parameter_dilatih=n_latih, parameter_total=n_total,
                   device=str(device), seed=args.seed, batch=args.batch)
    with open(out / f"{args.mode}_summary.json", "w") as f:
        json.dump(ringkas, f, indent=2)

    print(f"\nAkurasi val terbaik: {terbaik:.3f} (epoch {ep_terbaik}) | "
          f"epoch pertama val >= 90%: {ep_90} | waktu: {total_s:.0f}s")
    print(f"AKURASI TEST (ujian akhir): {t_acc:.3f}  ({round(t_acc * len(te.dataset))} dari {len(te.dataset)} foto benar)")
    print(f"Confusion matrix test (baris = kelas asli, kolom = tebakan; urutan {kelas}):")
    for k, baris_c in zip(kelas, conf):
        print(f"  {k:8s} {baris_c}")
    return ringkas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["feature", "partial", "scratch"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0, help="0 paling aman di Windows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights-path", default=None,
                    help="file resnet18-f37072fd.pth jika tidak bisa mengunduh bobot ImageNet")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

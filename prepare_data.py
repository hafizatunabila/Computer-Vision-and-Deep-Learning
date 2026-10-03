"""
prepare_data.py - merapikan dataset dan membaginya jadi train / val / test (80 / 10 / 10).

Yang dikerjakan:
  1. Membaca zip (atau folder) yang berisi satu folder per kelas. Folder yang
     bersarang (mis. car/car/car/) dan folder bawaan zip dari Mac (__MACOSX) ditangani otomatis.
  2. Setiap foto disimpan ulang sebagai JPG RGB dengan nama:
        <kelas>_<tanggal>_<sumber>_<nomor>.jpg        (pola slide 19)
     ke dataset_raw/<kelas>/  (ukuran asli dipertahankan).
  3. Menulis dataset_raw/metadata.csv.
  4. Membagi data ke data/train, data/val, data/test.
     - Pembagian dilakukan PER KELAS, supaya tiap kelas ada di ketiga bagian.
     - Foto yang hampir sama (mis. mobil yang sama dari sudut berbeda) dikelompokkan
       dengan perceptual hash, lalu dibagi per kelompok. Tujuannya mencegah data leakage:
       foto yang mirip tidak boleh ada di train sekaligus di val/test.
     - Salinan di data/ diperkecil (sisi terpanjang maks. 800 px) supaya training tidak
       lambat membaca foto raksasa. Ukuran asli tetap ada di dataset_raw/.

Pemakaian:
  python prepare_data.py --zip dataset_patrol.zip
  python prepare_data.py --src dataset_patrol            # kalau sudah diekstrak
"""
import argparse
import csv
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

# nama folder asal (huruf kecil) -> nama kelas. Kalau tidak ada di sini, nama folder dipakai apa adanya.
TERJEMAH = {"car": "mobil", "motorcycle": "motor", "fork": "garpu", "spoon": "sendok"}
ABAIKAN_EKSTENSI = {".txt", ".csv", ".json", ".md", ".db", ".ini"}


# ---------- hash untuk mendeteksi foto yang hampir sama ----------
def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def kelompokkan(hashes, ambang):
    """Gabungkan indeks yang hash-nya berjarak <= ambang (union-find)."""
    induk = list(range(len(hashes)))

    def cari(x):
        while induk[x] != x:
            induk[x] = induk[induk[x]]
            x = induk[x]
        return x

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= ambang:
                induk[cari(i)] = cari(j)
    grup = defaultdict(list)
    for i in range(len(hashes)):
        grup[cari(i)].append(i)
    return list(grup.values())


# ---------- membaca sumber ----------
def buka_rgb(path):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        latar = Image.new("RGB", im.size, (255, 255, 255))
        latar.paste(im, mask=im.split()[-1])
        return latar
    return im.convert("RGB")


def folder_valid(p):
    return p.is_dir() and not p.name.startswith((".", "__"))


def cari_akar(src):
    """Turun selama hanya ada satu subfolder dan tidak ada file; berhenti di folder yang berisi kelas-kelas."""
    cur = Path(src)
    while True:
        subdirs = [d for d in cur.iterdir() if folder_valid(d)]
        files = [f for f in cur.iterdir() if f.is_file() and not f.name.startswith(".")]
        if len(subdirs) == 1 and not files:
            cur = subdirs[0]
        else:
            return cur, sorted(subdirs, key=lambda d: d.name.lower())


def kumpulkan_gambar(folder):
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and "__MACOSX" not in p.parts \
                and p.suffix.lower() not in ABAIKAN_EKSTENSI:
            out.append(p)
    return out


def simpan(im, path, max_sisi=None):
    if max_sisi and max(im.size) > max_sisi:
        im = im.copy()
        im.thumbnail((max_sisi, max_sisi), Image.LANCZOS)
    im.save(path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path file zip dataset")
    ap.add_argument("--src", help="folder yang berisi satu subfolder per kelas")
    ap.add_argument("--out", default=".", help="folder keluaran (default: folder ini)")
    ap.add_argument("--val", type=float, default=0.10, help="proporsi validasi (default 0.10)")
    ap.add_argument("--test", type=float, default=0.10, help="proporsi test (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ambang", type=int, default=6,
                    help="jarak hash maksimum agar dua foto dianggap hampir sama")
    ap.add_argument("--max-sisi", type=int, default=800,
                    help="sisi terpanjang foto di folder data/ (0 = jangan perkecil)")
    ap.add_argument("--sumber", default="internet_dataset_patrol",
                    help="catatan provenance untuk metadata.csv")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = None
    if args.zip:
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(tmp)
        sumber_dir = Path(tmp)
    elif args.src:
        sumber_dir = Path(args.src)
    else:
        ap.error("isi --zip atau --src")

    akar, kelas_dirs = cari_akar(sumber_dir)
    if len(kelas_dirs) < 2:
        raise SystemExit("Butuh minimal 2 folder kelas di dalam zip.")
    print(f"Sumber data: {akar}")
    print("Folder kelas ditemukan:", ", ".join(d.name for d in kelas_dirs))

    tanggal = date.today().strftime("%Y%m%d")
    for d in (out / "dataset_raw", out / "data"):
        if d.exists():
            shutil.rmtree(d)

    # 1) baca semua foto, simpan ke dataset_raw
    per_kelas = {}
    for kd in kelas_dirs:
        kelas = TERJEMAH.get(kd.name.lower(), kd.name.lower().replace(" ", "_"))
        (out / "dataset_raw" / kelas).mkdir(parents=True, exist_ok=True)
        item = []
        for p in kumpulkan_gambar(kd):
            try:
                im = buka_rgb(p)
            except Exception as e:
                print(f"  dilewati (bukan gambar valid): {p.name} ({type(e).__name__})")
                continue
            nama = f"{kelas}_{tanggal}_internet_{len(item) + 1:03d}.jpg"
            simpan(im, out / "dataset_raw" / kelas / nama)
            item.append({"nama_file": nama, "kelas": kelas, "tanggal": tanggal,
                         "kondisi_cahaya": "foto_internet", "sumber": args.sumber,
                         "file_asli": p.name, "lebar": im.size[0], "tinggi": im.size[1],
                         "_hash": dhash(im)})
        per_kelas[kelas] = item
        print(f"{kelas}: {len(item)} gambar")
        if len(item) < 50:
            print(f"  PERHATIAN: kelas {kelas} kurang dari 50 gambar (syarat tugas: >= 50)")

    # 2) bagi train / val / test per kelas, per kelompok foto mirip
    rng = random.Random(args.seed)
    baris = []
    for kelas, item in per_kelas.items():
        grup = kelompokkan([x["_hash"] for x in item], args.ambang)
        besar = [g for g in grup if len(g) > 1]
        print(f"{kelas}: {len(grup)} kelompok ({len(besar)} kelompok berisi >1 foto mirip, "
              f"{sum(len(g) for g in besar)} foto)")
        rng.shuffle(grup)
        t_test = max(1, round(len(item) * args.test))
        t_val = max(1, round(len(item) * args.val))
        n_test = n_val = 0
        for g in grup:
            if n_test + len(g) <= t_test:
                split, n_test = "test", n_test + len(g)
            elif n_val + len(g) <= t_val:
                split, n_val = "val", n_val + len(g)
            else:
                split = "train"
            for i in g:
                item[i]["split"] = split
        for x in item:
            tujuan = out / "data" / x["split"] / kelas
            tujuan.mkdir(parents=True, exist_ok=True)
            im = Image.open(out / "dataset_raw" / kelas / x["nama_file"])
            simpan(im, tujuan / x["nama_file"], args.max_sisi or None)
            baris.append(x)

    kolom = ["nama_file", "kelas", "tanggal", "kondisi_cahaya", "sumber",
             "file_asli", "lebar", "tinggi", "split"]
    with open(out / "dataset_raw" / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        w.writerows(baris)

    print("\nRingkasan split:")
    print(f"  {'':6s}" + "".join(f"{k:>8s}" for k in per_kelas) + f"{'total':>8s}")
    for split in ("train", "val", "test"):
        jml = [sum(1 for x in baris if x["split"] == split and x["kelas"] == k) for k in per_kelas]
        print(f"  {split:6s}" + "".join(f"{j:8d}" for j in jml) + f"{sum(jml):8d}")
    print(f"\nSelesai. Metadata: {out / 'dataset_raw' / 'metadata.csv'}")
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

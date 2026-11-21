"""Plumbing test of identity_eval.main(): fake LFW folder + synthetic embeddings (monkey-patched acquire).
Validates parsing, caching, verification, identification, degradations, JSON/MD/TeX output -- NOT model accuracy."""
import os, sys, shutil, json, numpy as np, cv2
import identity_eval as ie
root = "fake_lfw"; shutil.rmtree(root, ignore_errors=True)
rng = np.random.default_rng(0)
D = 512
nper = 140
names = [f"Person_{i:03d}" for i in range(nper)]
centers = rng.normal(size=(nper, D)); centers /= np.linalg.norm(centers, axis=1, keepdims=True)
counts = {}
for k, nm in enumerate(names):
    os.makedirs(f"{root}/lfw_funneled/{nm}", exist_ok=True)
    counts[nm] = int(rng.integers(2, 5))
    for j in range(1, counts[nm] + 1):
        img = np.full((250, 250, 3), 0, np.uint8); img[:, :, 0] = k; img[:, :, 1] = rng.integers(0, 255)   # id coded in blue channel
        cv2.imwrite(f"{root}/lfw_funneled/{nm}/{nm}_{j:04d}.png", img)
# pairs.txt uses .jpg names -> write jpgs instead (lossless-ish check of id decoding not needed: id coded by quality-robust value)
for k, nm in enumerate(names):
    for j in range(1, counts[nm] + 1):
        img = np.zeros((250, 250, 3), np.uint8); img[:, :, 0] = k; img[:, :, 1] = int(rng.integers(0, 255))
        cv2.imwrite(f"{root}/lfw_funneled/{nm}/{nm}_{j:04d}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
lines = ["10\t30"]
for f in range(10):
    for _ in range(30):
        nm = names[rng.integers(nper)]; i, j = rng.choice(counts[nm], 2, replace=False) + 1
        lines.append(f"{nm}\t{i}\t{j}")
    for _ in range(30):
        a, b = rng.choice(nper, 2, replace=False)
        lines.append(f"{names[a]}\t{rng.integers(counts[names[a]]) + 1}\t{names[b]}\t{rng.integers(counts[names[b]]) + 1}")
open(f"{root}/pairs.txt", "w").write("\n".join(lines) + "\n")
# the real lfw-funneled.tgz also contains these text files next to the person folders (regression test for a crash)
for stray in ("pairs.txt", "pairsDevTrain.txt", "people.txt"):
    open(f"{root}/lfw_funneled/{stray}", "w").write("stray\n")

def fake_acquire(self, img, timing=None):
    k = int(np.median(img[:, :, 0]))
    if rng.random() < 0.03:
        return None, "no_face"
    k = min(k, nper - 1)
    x = centers[k] + 0.18 * rng.normal(size=D) / np.sqrt(D) * 4
    return x / np.linalg.norm(x), "ok"
ie.FacePipeline.acquire = fake_acquire
sys.argv = ["identity_eval.py", "--lfw-dir", root, "--model", "mbf", "--out", "fake_out/table2", "--skip-bench", "--ident-sizes", "2", "10", "40",
            "--degradation-pairs", "60"]
ie.main()
r = json.load(open("fake_out/table2.json"))
print(json.dumps(r["verification"]["operating_points"]["validation_fmr_target_0.01"], indent=1)[:600])
print(r["identification"]["10"]["FPIR"], r["identification"]["10"]["FNIR"])
print(open("fake_out/table2.md").read())

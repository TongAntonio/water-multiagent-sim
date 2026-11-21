#!/usr/bin/env python3
"""
identity_eval.py -- reproduces Table II (Identity Evaluation Reporting Template) of the paper.

Rows of Table II            what this script does
--------------------------  ----------------------------------------------------------------------
Verification errors         LFW 10-fold pairs (1:1).  Threshold tau_ver is chosen on the 9 *validation* folds
                            only and applied to the held-out fold.  Reports FMR / FNMR with Clopper-Pearson 95% CI,
                            trial counts, accuracy (mean +- sd over folds) and FNMR at validation-fixed FMR targets.
Identification errors       Open-set 1:N (Eq. 9-10: tau_id and margin delta) for several gallery sizes.  tau_id/delta
                            are calibrated on identities that are DISJOINT from the evaluation identities.
                            Reports FPIR (unknown accepted), FNIR (genuine rejected or mis-identified) and
                            mis-identification, with bootstrap CI over random galleries.
Acquisition failures        Failure-to-acquire (no face / small face / blur / dark / bright) on clean images and under
                            controlled degradations (low light, blur, lower-face occlusion, low resolution, JPEG).
Latency                     detect / align / embed / match, median and P95, in a fresh process (cold start excluded).
Resource use                peak RSS (memory) of that fresh process, model sizes; energy per attempt via Intel RAPL
                            when available, or from an external power meter (--measured-energy-j / --measured-attempts).
Population coverage         Counts of images / identities / pairs.  LFW = adult public figures, no age/gender/skin-tone
                            annotation and no children; optional --groups-csv for per-group error rates on your own
                            consented volunteer data.

Only the face DETECTOR and the face EMBEDDING model are loaded.  No age / gender / health / entitlement model is
used (the framework only checks identity).

Typical use
    python identity_eval.py --download --model mbf --out table2_mbf        # downloads LFW + models, runs everything
    python identity_eval.py --lfw-dir /data/lfw --model r50 --threads 1    # existing LFW folder
    python identity_eval.py --selftest                                     # unit tests of the metric code (no data)

Models: insightface 'buffalo_s' (SCRFD-500M + MobileFaceNet w600k_mbf) and 'buffalo_l' (SCRFD-10G + ArcFace-R50
w600k_r50), research / non-commercial licence of the insightface project.
"""
import argparse, json, math, os, platform, subprocess, sys, time, zipfile, tarfile, urllib.request
try:
    import resource          # Linux / macOS only
except ImportError:          # Windows
    resource = None
import numpy as np

BASE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/"
MODELS = {"mbf": ("buffalo_s.zip", "det_500m.onnx", "w600k_mbf.onnx"),
          "r50": ("buffalo_l.zip", "det_10g.onnx", "w600k_r50.onnx")}
LFW_FUNNELED = "https://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
LFW_PAIRS = "https://vis-www.cs.umass.edu/lfw/pairs.txt"
# Mirrors used when the original UMass host does not resolve (figshare files and SHA-256 are those used by scikit-learn).
LFW_FUNNELED_SRC = [("https://ndownloader.figshare.com/files/5976015", "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"),
                    (LFW_FUNNELED, None)]
LFW_PAIRS_SRC = [("https://ndownloader.figshare.com/files/5976006", "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
                 (LFW_PAIRS, None)]
ARC_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


# ============================================================ statistics helpers
def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def rate(k, n):
    lo, hi = clopper_pearson(k, n)
    return dict(k=int(k), n=int(n), rate=(k / n if n else float("nan")), ci95=[lo, hi])


def best_accuracy_threshold(gen, imp):
    """threshold maximising accuracy on validation scores (accept if score >= tau)."""
    cand = np.unique(np.concatenate([gen, imp]))
    cand = np.concatenate([[cand[0] - 1e-6], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-6]])
    gs, is_ = np.sort(gen), np.sort(imp)
    tp = len(gs) - np.searchsorted(gs, cand, "left")           # genuine accepted
    tn = np.searchsorted(is_, cand, "left")                     # impostor rejected
    acc = (tp + tn) / (len(gen) + len(imp))
    return float(cand[np.argmax(acc)])


def threshold_for_fmr(imp, target):
    """smallest tau such that validation FMR <= target (accept if score >= tau)."""
    s = np.sort(imp)
    k = int(math.floor(target * len(s)))
    return float(s[len(s) - k - 1] + 1e-9) if k < len(s) else float(s[0])


def eer(gen, imp):
    cand = np.unique(np.concatenate([gen, imp]))
    gs, is_ = np.sort(gen), np.sort(imp)
    fnmr = np.searchsorted(gs, cand, "left") / len(gs)
    fmr = 1 - np.searchsorted(is_, cand, "left") / len(is_)
    i = np.argmin(np.abs(fnmr - fmr))
    return float((fnmr[i] + fmr[i]) / 2), float(cand[i])


# ============================================================ face pipeline
class FacePipeline:
    def __init__(self, model="mbf", mdir="models_face", threads=1, det_size=320,
                 min_face=40, min_lap=10.0, dark=25.0, bright=235.0, det_thr=0.5):
        import onnxruntime as ort
        zipname, det, rec = MODELS[model]
        self.paths = ensure_models(model, mdir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.det = ort.InferenceSession(self.paths[0], so, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(self.paths[1], so, providers=["CPUExecutionProvider"])
        self.det_in = self.det.get_inputs()[0].name
        self.det_out = [o.name for o in self.det.get_outputs()]
        self.rec_in = self.rec.get_inputs()[0].name
        self.fmc = 3 if len(self.det_out) == 9 else 2
        self.use_kps = len(self.det_out) in (9, 15)
        assert self.use_kps, "detector without landmarks cannot be aligned"
        self.det_size, self.min_face, self.min_lap = det_size, min_face, min_lap
        self.dark, self.bright, self.det_thr = dark, bright, det_thr
        self.cache = {}

    # ---- SCRFD decode (same convention as the public insightface implementation)
    def _forward(self, img, thr):
        import cv2
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.det_size, self.det_size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.det.run(self.det_out, {self.det_in: blob})
        S, B, K = [], [], []
        h_in = w_in = self.det_size
        for idx, stride in enumerate((8, 16, 32)):
            sc = outs[idx]
            bb = outs[idx + self.fmc] * stride
            kp = outs[idx + self.fmc * 2] * stride
            h, w = h_in // stride, w_in // stride
            key = (h, w, stride)
            if key not in self.cache:
                a = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
                a = (a * stride).reshape(-1, 2)
                a = np.stack([a] * 2, axis=1).reshape(-1, 2)
                self.cache[key] = a
            ac = self.cache[key]
            pos = np.where(sc[:, 0] >= thr)[0]
            x1 = ac[:, 0] - bb[:, 0]; y1 = ac[:, 1] - bb[:, 1]; x2 = ac[:, 0] + bb[:, 2]; y2 = ac[:, 1] + bb[:, 3]
            boxes = np.stack([x1, y1, x2, y2], -1)
            kps = np.zeros((len(ac), 10), np.float32)
            for i in range(0, 10, 2):
                kps[:, i] = ac[:, 0] + kp[:, i]
                kps[:, i + 1] = ac[:, 1] + kp[:, i + 1]
            S.append(sc[pos]); B.append(boxes[pos]); K.append(kps[pos].reshape(-1, 5, 2))
        return S, B, K

    @staticmethod
    def _nms(dets, thr=0.4):
        x1, y1, x2, y2, s = dets.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][ovr <= thr]
        return keep

    def detect(self, img):
        import cv2
        ds = self.det_size
        r = img.shape[0] / img.shape[1]
        if r > 1.0:
            nh, nw = ds, int(ds / r)
        else:
            nw, nh = ds, int(ds * r)
        scale = nh / img.shape[0]
        canvas = np.zeros((ds, ds, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        S, B, K = self._forward(canvas, self.det_thr)
        S, B, K = np.vstack(S), np.vstack(B) / scale, np.vstack(K) / scale
        if len(S) == 0:
            return None
        pre = np.hstack([B, S]).astype(np.float32)
        keep = self._nms(pre)
        pre, K = pre[keep], K[keep]
        # most prominent face = largest area x score
        area = (pre[:, 2] - pre[:, 0]) * (pre[:, 3] - pre[:, 1]) * pre[:, 4]
        j = int(np.argmax(area))
        return pre[j], K[j], len(pre)

    def align(self, img, kps):
        import cv2
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARC_DST, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    def embed(self, crop):
        import cv2
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        f = self.rec.run(None, {self.rec_in: blob})[0][0].astype(np.float64)
        n = np.linalg.norm(f)
        if not np.all(np.isfinite(f)) or n <= 1e-6:        # Eq. (7) guard
            return None
        return f / n

    def acquire(self, img, timing=None):
        """returns (embedding | None, reason).  reason in {'ok','no_face','small_face','blur','dark','bright','bad_embedding'}"""
        import cv2
        t0 = time.perf_counter()
        d = self.detect(img)
        t1 = time.perf_counter()
        if d is None:
            if timing is not None: timing.append((t1 - t0, 0, 0))
            return None, "no_face"
        box, kps, nf = d
        if min(box[2] - box[0], box[3] - box[1]) < self.min_face:
            return None, "small_face"
        crop = self.align(img, kps)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        t2 = time.perf_counter()
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_lap:
            return None, "blur"
        m = float(gray.mean())
        if m < self.dark:
            return None, "dark"
        if m > self.bright:
            return None, "bright"
        e = self.embed(crop)
        t3 = time.perf_counter()
        if timing is not None:
            timing.append((t1 - t0, t2 - t1, t3 - t2))
        return (e, "ok") if e is not None else (None, "bad_embedding")


def ensure_models(kind, mdir):
    zipname, det, rec = MODELS[kind]
    os.makedirs(mdir, exist_ok=True)
    out = [os.path.join(mdir, det), os.path.join(mdir, rec)]
    if all(os.path.exists(p) for p in out):
        return out
    zp = os.path.join(mdir, zipname)
    if not os.path.exists(zp):
        print(f"downloading {BASE_URL + zipname} ...", flush=True)
        urllib.request.urlretrieve(BASE_URL + zipname, zp)
    with zipfile.ZipFile(zp) as z:                       # extract ONLY detector + embedding model
        for name in (det, rec):
            with z.open(name) as src, open(os.path.join(mdir, name), "wb") as dst:
                dst.write(src.read())
    return out


# ============================================================ LFW handling
def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch_file(sources, dst, what):
    """Try each (url, sha256-or-None) in turn; verify the checksum when known; write atomically."""
    part = dst + ".part"
    for url, sha in sources:
        try:
            print(f"downloading {what} from {url} ...", flush=True)
            urllib.request.urlretrieve(url, part)
            got = _sha256(part)
            if sha and got != sha:
                raise IOError(f"checksum mismatch (got {got[:12]}..., expected {sha[:12]}...)")
            os.replace(part, dst)
            print(f"  ok, sha256 {got[:16]}...", flush=True)
            return
        except Exception as e:
            print(f"  failed: {e}", flush=True)
            if os.path.exists(part):
                os.remove(part)
    sys.exit(f"could not download {what}. Download it in a browser from one of:\n  " +
             "\n  ".join(u for u, _ in sources) + f"\nand save it as {dst}, then re-run the same command.")


def ensure_lfw(root, download):
    img_dir = None
    for cand in ("lfw_funneled", "lfw", "lfw-funneled"):
        if os.path.isdir(os.path.join(root, cand)):
            img_dir = os.path.join(root, cand)
            break
    pairs = os.path.join(root, "pairs.txt")
    if img_dir is None or not os.path.exists(pairs):
        if not download:
            sys.exit(f"LFW not found in {root} (need lfw_funneled/ and pairs.txt). Re-run with --download or download manually from "
                     f"{LFW_FUNNELED} and {LFW_PAIRS}")
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "lfw-funneled.tgz")
        if not os.path.exists(tgz):
            fetch_file(LFW_FUNNELED_SRC, tgz, "LFW funneled images (233 MB)")
        if not os.path.exists(pairs):
            fetch_file(LFW_PAIRS_SRC, pairs, "LFW pairs.txt")
        with tarfile.open(tgz) as t:
            try:
                t.extractall(root, filter="data")      # Python >= 3.12 / patched 3.9-3.11
            except TypeError:
                t.extractall(root)
        img_dir = os.path.join(root, "lfw_funneled")
    return img_dir, pairs


def parse_pairs(pairs_txt, img_dir):
    """official LFW 'pairs.txt': header 'folds per_fold', then for each fold per_fold matched lines
       'name i j' followed by per_fold mismatched lines 'name1 i name2 j'."""
    lines = [l.strip().split() for l in open(pairs_txt) if l.strip()]
    nf, per = int(lines[0][0]), int(lines[0][1])
    out, p = [], 1
    f = lambda nm, i: os.path.join(img_dir, nm, f"{nm}_{int(i):04d}.jpg")
    for fold in range(nf):
        for _ in range(per):
            nm, i, j = lines[p]; p += 1
            out.append((f(nm, i), f(nm, j), 1, fold, nm, nm))
        for _ in range(per):
            n1, i, n2, j = lines[p]; p += 1
            out.append((f(n1, i), f(n2, j), 0, fold, n1, n2))
    return out


def read_img(path):
    import cv2
    return cv2.imread(path, cv2.IMREAD_COLOR)


def embed_all(pipe, paths, cache_file=None, log_every=1000):
    cache = {}
    if cache_file and os.path.exists(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        cache = dict(z["cache"].item())
    todo = [p for p in paths if p not in cache]
    t0 = time.time()
    for n, p in enumerate(todo):
        img = read_img(p)
        if img is None:
            cache[p] = (None, "unreadable")
        else:
            cache[p] = pipe.acquire(img)
        if (n + 1) % log_every == 0:
            print(f"  embedded {n + 1}/{len(todo)}  ({(time.time() - t0) / (n + 1) * 1000:.0f} ms/img)", flush=True)
    if cache_file:
        np.savez_compressed(cache_file, cache=np.array(cache, dtype=object))
    return cache


# ============================================================ evaluation: verification (1:1)
def eval_verification(pairs, cache, targets=(1e-1, 1e-2, 1e-3)):
    ok = [(cache[a][0] is not None and cache[b][0] is not None) for a, b, *_ in pairs]
    scores = np.array([float(cache[a][0] @ cache[b][0]) if o else np.nan for (a, b, *_), o in zip(pairs, ok)])
    lab = np.array([p[2] for p in pairs]); fold = np.array([p[3] for p in pairs]); ok = np.array(ok)
    out = dict(n_pairs=len(pairs), n_genuine_pairs=int(lab.sum()), n_impostor_pairs=int((1 - lab).sum()))
    fta_pairs = (~ok)
    out["pair_acquisition_failure"] = rate(int(fta_pairs.sum()), len(pairs))
    folds = sorted(set(fold))
    accs, ops = [], {}
    cnt = {t: dict(fn=0, ng=0, fm=0, ni=0, taus=[]) for t in ("acc",) + tuple(targets)}
    for f in folds:
        tr = (fold != f) & ok; te = (fold == f) & ok
        g_tr, i_tr = scores[tr & (lab == 1)], scores[tr & (lab == 0)]
        g_te, i_te = scores[te & (lab == 1)], scores[te & (lab == 0)]
        tau = best_accuracy_threshold(g_tr, i_tr)
        accs.append(((g_te >= tau).sum() + (i_te < tau).sum()) / (len(g_te) + len(i_te)))
        c = cnt["acc"]; c["fn"] += int((g_te < tau).sum()); c["ng"] += len(g_te); c["fm"] += int((i_te >= tau).sum()); c["ni"] += len(i_te); c["taus"].append(tau)
        for t in targets:
            tau = threshold_for_fmr(i_tr, t)
            c = cnt[t]; c["fn"] += int((g_te < tau).sum()); c["ng"] += len(g_te); c["fm"] += int((i_te >= tau).sum()); c["ni"] += len(i_te); c["taus"].append(tau)
    out["accuracy_10fold"] = dict(mean=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)), per_fold=[float(a) for a in accs])
    e, te_ = eer(scores[ok & (lab == 1)], scores[ok & (lab == 0)])
    out["eer_all_pairs_descriptive"] = dict(eer=e, tau=te_, note="descriptive only: threshold not validation-selected")
    out["operating_points"] = {}
    for key, c in cnt.items():
        name = "best_accuracy_tau" if key == "acc" else f"validation_fmr_target_{key:g}"
        out["operating_points"][name] = dict(tau_per_fold=[round(x, 4) for x in c["taus"]],
                                             FNMR=rate(c["fn"], c["ng"]), FMR=rate(c["fm"], c["ni"]))
    return out, scores, lab, ok


# ============================================================ evaluation: open-set identification (1:N)
def identify(Q, G, tau, delta):
    """Eq. (9)-(10).  returns (accepted: bool array, top index)."""
    S = Q @ G.T
    if G.shape[0] < 2:
        return np.zeros(len(Q), bool), np.zeros(len(Q), int)          # |G| < 2: identification unavailable
    idx = np.argsort(-S, axis=1)[:, :2]
    s1 = np.take_along_axis(S, idx[:, :1], 1)[:, 0]
    s2 = np.take_along_axis(S, idx[:, 1:2], 1)[:, 0]
    acc = (s1 >= tau) & ((s1 - s2) >= delta)                           # ties => margin 0 < delta => ambiguous => reject
    return acc, idx[:, 0]


def eval_identification(people, sizes=(2, 10, 50, 100, 500), trials=20, fpir_target=1e-2, seed=0, deltas=(0.0, 0.05, 0.1),
                        probes_per_id=2):
    """people: dict name -> (n_i, d) array of embeddings (n_i >= 2).  Identity-disjoint calibration / evaluation split."""
    rng = np.random.default_rng(seed)
    names = sorted(people)
    rng.shuffle(names)
    ncal = int(0.3 * len(names))
    cal, ev = names[:ncal], names[ncal:]

    def draw(pool, G):
        g = rng.choice(len(pool), min(2 * G, len(pool)) , replace=False)
        enrol = [pool[i] for i in g[:G]]
        unknown = [pool[i] for i in g[G:2 * G]]
        Gm, Qg, Qu, lab = [], [], [], []
        for k, nm in enumerate(enrol):
            E = people[nm]
            j = rng.integers(len(E))
            Gm.append(E[j])
            others = [x for t, x in enumerate(E) if t != j][:probes_per_id]
            for x in others:
                Qg.append(x); lab.append(k)
        for nm in unknown:
            Qu.append(people[nm][rng.integers(len(people[nm]))])
        return np.array(Gm), np.array(Qg), np.array(lab), np.array(Qu)

    res = {}
    for G in sizes:
        if G * 2 > len(ev) or G * 2 > len(cal):
            res[G] = dict(skipped=f"not enough identities for gallery size {G} (cal={len(cal)}, eval={len(ev)})")
            continue
        # --- calibrate (tau_id, delta) on calibration identities: FPIR <= target, then minimise FNIR
        cal_draws = [draw(cal, G) for _ in range(trials)]
        best = None
        unk_scores = []
        for Gm, Qg, lab, Qu in cal_draws:
            unk_scores.append((Qu @ Gm.T).max(1))
        pool = np.concatenate(unk_scores)
        for d in deltas:
            tau = threshold_for_fmr(pool, fpir_target)
            fp = np.mean(np.concatenate([identify(Qu, Gm, tau, d)[0] for Gm, Qg, lab, Qu in cal_draws]))
            fn = np.mean(np.concatenate([~(identify(Qg, Gm, tau, d)[0] & (identify(Qg, Gm, tau, d)[1] == lab)) for Gm, Qg, lab, Qu in cal_draws]))
            if fp <= fpir_target * 1.5 and (best is None or fn < best[0]):
                best = (fn, tau, d)
        if best is None:
            best = (1.0, threshold_for_fmr(pool, fpir_target), deltas[-1])
        _, tau, delta = best
        # --- evaluate on evaluation identities
        fp_t, fn_t, mis_t, rej_t = [], [], [], []
        nfp = nu = nfn = ng = 0
        for _ in range(trials):
            Gm, Qg, lab, Qu = draw(ev, G)
            a_u, _ = identify(Qu, Gm, tau, delta)
            a_g, top = identify(Qg, Gm, tau, delta)
            correct = a_g & (top == lab)
            fp_t.append(a_u.mean()); fn_t.append(1 - correct.mean())
            mis_t.append((a_g & (top != lab)).mean()); rej_t.append((~a_g).mean())
            nfp += int(a_u.sum()); nu += len(Qu); nfn += int((~correct).sum()); ng += len(Qg)

        def ci(x):
            x = np.array(x)
            return dict(mean=float(x.mean()), ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))])
        res[G] = dict(tau_id=float(tau), delta=float(delta), trials=trials, unknown_probes=nu, genuine_probes=ng,
                      FPIR=ci(fp_t), FNIR=ci(fn_t), misidentification=ci(mis_t), reject_genuine=ci(rej_t),
                      FPIR_pooled=rate(nfp, nu), FNIR_pooled=rate(nfn, ng))
    res["_split"] = dict(calibration_identities=len(cal), evaluation_identities=len(ev), fpir_target=fpir_target,
                         single_entry_gallery="identification disabled (|G| < 2)")
    return res


# ============================================================ degradations
def degrade(img, kind):
    import cv2
    if kind == "clean":
        return img
    if kind == "lowlight":
        x = (img.astype(np.float32) / 255.0) ** 2.2 * 0.35
        return np.clip(x * 255, 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 3.0)
    if kind == "occlusion_lower":                                  # lower-face occlusion (mask-like), 250x250 funneled geometry
        o = img.copy(); h, w = o.shape[:2]
        o[int(0.55 * h):int(0.85 * h), int(0.25 * w):int(0.75 * w)] = 0
        return o
    if kind == "lowres":
        h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (int(w * 0.3), int(h * 0.3))), (w, h))
    if kind == "jpeg20":
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    raise ValueError(kind)


def eval_degradations(pipe, pairs, cache, tau, n_pairs=600, kinds=("clean", "lowlight", "blur", "occlusion_lower", "lowres", "jpeg20")):
    sel = [p for p in pairs if cache[p[0]][0] is not None][:: max(1, len(pairs) // n_pairs)][:n_pairs]
    out = {}
    for k in kinds:
        reasons, g, i = {}, [], []
        fta = 0
        for a, b, lab, *_ in sel:
            ea = cache[a][0]
            img = degrade(read_img(b), k)
            eb, why = pipe.acquire(img)
            reasons[why] = reasons.get(why, 0) + 1
            if eb is None:
                fta += 1
                continue
            (g if lab == 1 else i).append(float(ea @ eb))
        out[k] = dict(attempts=len(sel), FTA=rate(fta, len(sel)), reasons=reasons,
                      FNMR_at_tau=rate(int((np.array(g) < tau).sum()), len(g)), FMR_at_tau=rate(int((np.array(i) >= tau).sum()), len(i)),
                      tau=tau)
    return out


# ============================================================ bench (latency / memory / energy) -- run in a FRESH process
def read_rapl():
    for p in ("/sys/class/powercap/intel-rapl:0/energy_uj", "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"):
        try:
            return int(open(p).read())
        except Exception:
            pass
    return None


def peak_rss_mb():
    """Peak resident memory of this process in MB (Linux/macOS: getrusage; Windows: PeakWorkingSetSize)."""
    if resource is not None:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / (1024 * 1024) if ok else float("nan")
    except Exception:
        return float("nan")


def bench(args):
    import cv2
    pipe = FacePipeline(args.model, args.model_dir, args.threads)
    imgs = [read_img(p) for p in args.bench_images]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        sys.exit("no benchmark images")
    G = {n: np.random.default_rng(0).normal(size=(n, 512)) for n in (10, 100, 1000)}
    for g in G.values():
        g /= np.linalg.norm(g, axis=1, keepdims=True)
    for _ in range(args.warmup):
        pipe.acquire(imgs[_ % len(imgs)])
    e0, t0 = read_rapl(), time.perf_counter()
    tim, tot, match = [], [], {n: [] for n in G}
    ok = 0
    for r in range(args.runs):
        img = imgs[r % len(imgs)]
        t = time.perf_counter()
        tm = []
        e, why = pipe.acquire(img, tm)
        if e is not None:
            ok += 1
            for n, g in G.items():
                t1 = time.perf_counter(); s = g @ e; _ = s.argmax(); match[n].append(time.perf_counter() - t1)
            tim.append(tm[-1])
        tot.append(time.perf_counter() - t)
    wall = time.perf_counter() - t0
    e1 = read_rapl()
    rss_mb = peak_rss_mb()
    if not tim:
        sys.exit("benchmark: no successful acquisition on the supplied images")
    tim = np.array(tim)
    pc = lambda a, q: float(np.percentile(a, q) * 1000)
    res = dict(model=args.model, threads=args.threads, runs=args.runs, successful_acquisitions=ok,
               cpu=platform.processor() or platform.machine(), platform=platform.platform(), python=sys.version.split()[0],
               detect_ms=dict(median=pc(tim[:, 0], 50), p95=pc(tim[:, 0], 95)),
               align_quality_ms=dict(median=pc(tim[:, 1], 50), p95=pc(tim[:, 1], 95)),
               embed_ms=dict(median=pc(tim[:, 2], 50), p95=pc(tim[:, 2], 95)),
               acquisition_to_embedding_ms=dict(median=pc(tim.sum(1), 50), p95=pc(tim.sum(1), 95)),
               match_ms={str(n): dict(median=pc(v, 50), p95=pc(v, 95)) for n, v in match.items()},
               acquisition_to_decision_ms_G100=dict(median=pc(tim.sum(1) + np.array(match[100][:len(tim)]), 50),
                                                    p95=pc(tim.sum(1) + np.array(match[100][:len(tim)]), 95)),
               peak_rss_mb=rss_mb,
               model_files_mb={os.path.basename(p): os.path.getsize(p) / 1e6 for p in pipe.paths},
               energy=dict())
    if e0 is not None and e1 is not None and e1 >= e0:
        res["energy"] = dict(method="Intel RAPL package energy (includes idle + other processes)", joule_per_attempt=(e1 - e0) / 1e6 / args.runs)
    elif args.measured_energy_j is not None:
        res["energy"] = dict(method="external power meter (user supplied)", joule_per_attempt=args.measured_energy_j / max(1, args.measured_attempts))
    else:
        res["energy"] = dict(method="not measured (no RAPL access; use a USB power meter and pass --measured-energy-j / --measured-attempts)")
    print(json.dumps(res))


# ============================================================ self test (no data, no models)
def selftest():
    rng = np.random.default_rng(1)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok &= bool(cond)

    # Clopper-Pearson vs known values
    lo, hi = clopper_pearson(0, 100); check("CP(0/100) upper ~ 0.0362", abs(hi - 0.03620) < 1e-3 and lo == 0.0)
    lo, hi = clopper_pearson(50, 100); check("CP(50/100) ~ [0.398,0.602]", abs(lo - 0.3983) < 2e-3 and abs(hi - 0.6017) < 2e-3)
    # perfectly separated scores
    g, i = rng.uniform(0.6, 1, 500), rng.uniform(-1, 0.4, 500)
    tau = best_accuracy_threshold(g, i); check("perfect separation: acc=1", ((g >= tau).sum() + (i < tau).sum()) == 1000)
    e, _ = eer(g, i); check("perfect separation: EER=0", e == 0.0)
    # identical distributions => EER ~ 0.5
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    check("identical distributions: EER ~ 0.5", abs(eer(a, b)[0] - 0.5) < 0.03)
    # threshold for FMR target
    imp = rng.normal(size=100000); t = threshold_for_fmr(imp, 0.01)
    check("threshold_for_fmr gives FMR <= target", (imp >= t).mean() <= 0.01 + 1e-9 and (imp >= t).mean() > 0.0085)
    # identification: well separated synthetic embeddings
    d, nid = 64, 400
    centers = rng.normal(size=(nid, d)); centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def sample(c, sd):
        x = c + sd * rng.normal(size=c.shape) / math.sqrt(d); return x / np.linalg.norm(x)

    people = {f"p{k}": np.array([sample(centers[k], 0.15) for _ in range(4)]) for k in range(nid)}
    r = eval_identification(people, sizes=(2, 10, 50), trials=10)
    check("identification separable: FPIR small", r[50]["FPIR"]["mean"] <= 0.03)
    check("identification separable: FNIR small", r[50]["FNIR"]["mean"] <= 0.05)
    people_n = {f"p{k}": np.array([sample(centers[k], 3.0) for _ in range(4)]) for k in range(nid)}
    r = eval_identification(people_n, sizes=(50,), trials=10)
    check("identification on noise: FNIR high", r[50]["FNIR"]["mean"] > 0.8)
    # Eq. (10): tied / duplicate gallery entries must be rejected when delta > 0, single-entry gallery disabled
    G = np.array([centers[0], centers[0]]); acc, _ = identify(centers[:1], G, 0.5, 0.05); check("tied top-2 => ambiguous => reject", not acc[0])
    acc, _ = identify(centers[:1], centers[:1], 0.5, 0.05); check("|G|=1 => identification unavailable", not acc[0])
    acc, top = identify(centers[:1], centers[:5], 0.5, 0.05); check("unique match accepted", acc[0] and top[0] == 0)
    # verification protocol on synthetic pairs (paths -> embeddings via fake cache)
    pairs, cache = [], {}
    for f in range(10):
        for k in range(300):
            c = centers[(f * 300 + k) % nid]; a_, b_ = sample(c, 0.15), sample(c, 0.15)
            pa, pb = f"a{f}_{k}", f"b{f}_{k}"; cache[pa] = (a_, "ok"); cache[pb] = (b_, "ok"); pairs.append((pa, pb, 1, f, "x", "x"))
        for k in range(300):
            c1, c2 = centers[rng.integers(nid)], centers[rng.integers(nid)]
            pa, pb = f"c{f}_{k}", f"d{f}_{k}"; cache[pa] = (sample(c1, 0.15), "ok"); cache[pb] = (sample(c2, 0.15), "ok"); pairs.append((pa, pb, 0, f, "x", "y"))
    out, *_ = eval_verification(pairs, cache)
    check("verification synthetic: accuracy > 0.97", out["accuracy_10fold"]["mean"] > 0.97)
    cache[pairs[0][0]] = (None, "no_face")
    out, *_ = eval_verification(pairs, cache)
    check("acquisition failure counted separately (1/6000 pairs)", out["pair_acquisition_failure"]["k"] == 1)
    print("SELFTEST", "OK" if ok else "FAILED")
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfw-dir", default="lfw_data")
    ap.add_argument("--download", action="store_true", help="download LFW funneled + pairs.txt if missing")
    ap.add_argument("--model", choices=list(MODELS), default="mbf")
    ap.add_argument("--model-dir", default="models_face")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op threads (1 = edge-like)")
    ap.add_argument("--out", default="table2_results")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true", help="internal: latency/memory benchmark (run in fresh process)")
    ap.add_argument("--bench-images", nargs="*", default=[])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measured-energy-j", type=float, default=None)
    ap.add_argument("--measured-attempts", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=0, help="debug: use only the first N pairs of each fold")
    ap.add_argument("--degradation-pairs", type=int, default=600)
    ap.add_argument("--ident-sizes", type=int, nargs="*", default=[2, 10, 50, 100, 500])
    ap.add_argument("--skip-identification", action="store_true")
    ap.add_argument("--skip-bench", action="store_true", help="skip latency/memory benchmark")
    ap.add_argument("--groups-csv", default=None, help="optional CSV 'person,group' for per-group verification errors")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.bench:
        return bench(args)

    img_dir, pairs_txt = ensure_lfw(args.lfw_dir, args.download)
    pipe = FacePipeline(args.model, args.model_dir, args.threads)
    pairs = parse_pairs(pairs_txt, img_dir)
    if args.max_pairs:
        pairs = [p for f in range(10) for p in [q for q in pairs if q[3] == f][:args.max_pairs] ]
    paths = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    print(f"{len(pairs)} pairs, {len(paths)} unique images; embedding with {args.model} ...", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cache = embed_all(pipe, paths, args.out + f"_{args.model}_cache.npz")
    reasons = {}
    for p in paths:
        reasons[cache[p][1]] = reasons.get(cache[p][1], 0) + 1
    res = dict(config=dict(model=args.model, threads=args.threads, det_size=pipe.det_size, min_face=pipe.min_face,
                           min_laplacian_var=pipe.min_lap, dark=pipe.dark, bright=pipe.bright, det_thr=pipe.det_thr))
    ver, scores, lab, ok = eval_verification(pairs, cache)
    res["verification"] = ver
    res["acquisition_failures"] = dict(clean_images=dict(images=len(paths), reasons=reasons,
                                                         FTA=rate(sum(v for k, v in reasons.items() if k != "ok"), len(paths))))
    tau = ver["operating_points"]["validation_fmr_target_0.01"]["tau_per_fold"]
    tau = float(np.mean(tau))
    res["acquisition_failures"]["degradations"] = eval_degradations(pipe, pairs, cache, tau, args.degradation_pairs)
    if not args.skip_identification:
        people = {}
        for nm in sorted(os.listdir(img_dir)):
            if not os.path.isdir(os.path.join(img_dir, nm)):      # the LFW archive also holds pairs*.txt / people*.txt here
                continue
            fs = sorted(f for f in os.listdir(os.path.join(img_dir, nm)) if f.lower().endswith(".jpg"))
            if len(fs) >= 2:
                people[nm] = [os.path.join(img_dir, nm, f) for f in fs]
        allp = [p for v in people.values() for p in v]
        cache2 = embed_all(pipe, allp, args.out + f"_{args.model}_cache_ident.npz")
        emb = {nm: np.array([cache2[p][0] for p in ps if cache2[p][0] is not None]) for nm, ps in people.items()}
        emb = {k: v for k, v in emb.items() if len(v) >= 2}
        res["identification"] = eval_identification(emb, tuple(args.ident_sizes))
        res["identification"]["_protocol"] = dict(people_with_2plus_images=len(emb), note="one randomly chosen enrolment image per identity, "
                                                  "probes = other images of enrolled identities, unknown probes = identities not in the gallery; "
                                                  "LFW has no session labels, so enrolment/test session separation is not possible")
    # latency / memory / energy in a fresh process
    bench_imgs = [p for p in paths if cache[p][0] is not None][:50]
    cmd = [sys.executable, os.path.abspath(__file__), "--bench", "--model", args.model, "--model-dir", args.model_dir, "--threads", str(args.threads),
           "--runs", str(args.runs), "--warmup", str(args.warmup), "--bench-images", *bench_imgs]
    if args.measured_energy_j is not None:
        cmd += ["--measured-energy-j", str(args.measured_energy_j), "--measured-attempts", str(args.measured_attempts)]
    if not args.skip_bench:
        try:
            res["latency_resource"] = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
        except Exception as e:      # do not lose the accuracy results if only the benchmark fails
            err = getattr(e, "stderr", "") or ""
            print(f"WARNING: latency/memory benchmark failed ({e.__class__.__name__}); results written without it.\n{err[-1500:]}", flush=True)
    n_img = len(paths); n_id = len({p[4] for p in pairs} | {p[5] for p in pairs})
    res["population_coverage"] = dict(dataset="LFW (funneled) 10-fold pairs", images=n_img, identities_in_pairs=n_id, pairs=len(pairs),
                                      annotations="none for age, sex, skin tone", children="not represented", population="adult public figures (press photos)",
                                      claim_limit="results do not establish performance for children, older adults, or the target relief population")
    if args.groups_csv:
        import csv
        grp = {r["person"]: r["group"] for r in csv.DictReader(open(args.groups_csv))}
        gres = {}
        for g in sorted(set(grp.values())):
            idx = [k for k, p in enumerate(pairs) if grp.get(p[4]) == g and grp.get(p[5]) == g and ok[k]]
            gen = scores[[k for k in idx if lab[k] == 1]]; imp = scores[[k for k in idx if lab[k] == 0]]
            gres[g] = dict(genuine=len(gen), impostor=len(imp), FNMR=rate(int((gen < tau).sum()), len(gen)), FMR=rate(int((imp >= tau).sum()), len(imp)))
        res["population_coverage"]["per_group_at_tau"] = gres
    json.dump(res, open(args.out + ".json", "w"), indent=2)
    write_tables(res, args.out)
    print("wrote", args.out + ".json", args.out + ".md", args.out + ".tex")


def pct(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def write_tables(res, out):
    v = res["verification"]; op = v["operating_points"]
    L = []
    c = op["validation_fmr_target_0.01"]
    L.append(("Verification errors", f"Accuracy {pct(v['accuracy_10fold']['mean'])} $\\pm$ {pct(v['accuracy_10fold']['sd'])} (10-fold); "
              f"at validation-fixed FMR$\\le$1%: FNMR {pct(c['FNMR']['rate'])} [{pct(c['FNMR']['ci95'][0])}, {pct(c['FNMR']['ci95'][1])}], "
              f"FMR {pct(c['FMR']['rate'])}; trials {c['FNMR']['n']} genuine / {c['FMR']['n']} impostor; "
              f"pair acquisition failures {pct(v['pair_acquisition_failure']['rate'])}"))
    if "identification" in res:
        parts = []
        for G, r in res["identification"].items():
            if isinstance(G, str) or not isinstance(r, dict) or "FPIR" not in r:
                continue
            parts.append(f"$|G|$={G}: FPIR {pct(r['FPIR']['mean'])}, FNIR {pct(r['FNIR']['mean'])} ($\\tau_{{id}}$={r['tau_id']:.2f}, $\\delta$={r['delta']:.2f})")
        L.append(("Identification errors", "; ".join(parts) + " (open-set, identity-disjoint calibration)"))
    a = res["acquisition_failures"]
    deg = a["degradations"]
    L.append(("Acquisition failures", f"Clean: {pct(a['clean_images']['FTA']['rate'])}; " + "; ".join(f"{k}: {pct(d['FTA']['rate'])}" for k, d in deg.items() if k != "clean")))
    lr = res.get("latency_resource")
    if lr is None:
        lr = dict(threads=0, cpu="n/a", peak_rss_mb=float("nan"), model_files_mb={}, energy=dict(method="not measured"),
                  acquisition_to_decision_ms_G100=dict(median=float("nan"), p95=float("nan")))
    L.append(("Latency", f"Acquisition-to-decision ($|G|$=100, {lr['threads']} thread): median {lr['acquisition_to_decision_ms_G100']['median']:.0f} ms, "
              f"P95 {lr['acquisition_to_decision_ms_G100']['p95']:.0f} ms on {lr['cpu']}"))
    e = lr["energy"]
    L.append(("Resource use", f"Peak RSS {lr['peak_rss_mb']:.0f} MB; models {sum(lr['model_files_mb'].values()):.0f} MB; energy/attempt: "
              + (f"{e['joule_per_attempt']:.2f} J ({e['method']})" if "joule_per_attempt" in e else e["method"])))
    pc = res["population_coverage"]
    L.append(("Population coverage", f"{pc['dataset']}: {pc['images']} images, {pc['identities_in_pairs']} identities, {pc['pairs']} pairs; {pc['population']}; "
              f"no age/sex/skin-tone labels; children not represented"))
    def md(t):
        for k_, v_ in (("$\\pm$", "±"), ("$\\le$", "≤"), ("$\\tau_{id}$", "τ_id"), ("$\\delta$", "δ"), ("$|G|$", "G"), ("$", "")):
            t = t.replace(k_, v_)
        return t.replace("{", "").replace("}", "")
    def tex(t):
        return t.replace("%", "\\%").replace("\\\\%", "\\%").replace("_", "\\_").replace("$\\tau\\_{id}$", "$\\tau_{id}$")
    with open(out + ".md", "w") as f:
        f.write("| Outcome | Result |\n|---|---|\n")
        for k, s_ in L:
            f.write(f"| {k} | {md(s_)} |\n")
    with open(out + ".tex", "w") as f:
        f.write("\\begin{tabular}{p{2.6cm}p{5.4cm}}\n\\hline Outcome & Result \\\\\\hline\n")
        for k, s_ in L:
            f.write(f"{k} & {tex(s_)} \\\\\n")
        f.write("\\hline\\end{tabular}\n")


if __name__ == "__main__":
    main()

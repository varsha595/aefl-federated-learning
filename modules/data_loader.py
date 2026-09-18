"""
data_loader.py — dataset download/loading (Kaggle via kagglehub, with graceful
local-cache fallback) and the Non-IID multi-hospital simulation.

Only the primary Chest X-ray Pneumonia dataset feeds federated training (so
every hospital speaks the same label space). The secondary datasets
(COVID-19 Radiography, RSNA Pneumonia) are loaded — when available — purely
to *evaluate* the trained global model out-of-distribution, which is what
"benchmarking and showing generalizability" means in practice.
"""

import os
import glob
import numpy as np

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------
# Kaggle download helpers (graceful — never crash the pipeline)
# --------------------------------------------------------------------------

def _kaggle_download(handle):
    """Try kagglehub; return a local directory path or None on any failure."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(handle)
        print(f"  ✅ Downloaded via kagglehub: {handle}")
        return path
    except Exception as e:
        print(f"  ⚠️  kagglehub download failed for {handle}: {e}")
        return None


def _find_local(candidates):
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return None


def resolve_dataset_path(handle, local_candidates):
    """Prefer an already-downloaded local copy; otherwise try kagglehub."""
    local = _find_local(local_candidates)
    if local:
        print(f"  ✅ Using cached local copy: {local}")
        return local
    return _kaggle_download(handle)


# --------------------------------------------------------------------------
# Low-level image loading
# --------------------------------------------------------------------------

def _load_image(path, image_size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype="float32") / 255.0
    return arr


def _load_folder_as_class(folder, label, image_size, limit=None):
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    files = sorted(files)
    if limit:
        files = files[:limit]
    X = np.zeros((len(files), image_size, image_size, 3), dtype="float32")
    for i, f in enumerate(files):
        try:
            X[i] = _load_image(f, image_size)
        except Exception:
            continue
    y = np.full((len(files),), label, dtype="int64")
    return X, y


# --------------------------------------------------------------------------
# Primary dataset — Chest X-ray Pneumonia (Kaggle: paultimothymooney/chest-xray-pneumonia)
# --------------------------------------------------------------------------

def load_chest_xray_pneumonia(image_size=128, seed=42, max_per_split=None):
    """Returns (X_train, y_train, X_val, y_val, X_test, y_test) with labels
    0=NORMAL, 1=PNEUMONIA. Re-splits train/val (the Kaggle 'val' folder only
    has 16 images) into an 80/10/10 stratified train/val/test partition.
    """
    root = resolve_dataset_path(
        "paultimothymooney/chest-xray-pneumonia",
        local_candidates=[
            "./data/chest_xray", "./data/chest-xray-pneumonia",
            os.path.expanduser("~/.cache/kagglehub/datasets/paultimothymooney/"
                                "chest-xray-pneumonia/versions"),
        ],
    )
    if root is None:
        print("  ❌ Chest X-ray Pneumonia dataset unavailable — cannot continue "
              "(this is the required primary dataset).")
        return None

    # kagglehub may nest a version dir, or the dataset may already contain
    # train/val/test at top level.
    chest_dir = _find_chest_xray_root(root)

    normal_files, pneumonia_files = [], []
    for split in ("train", "val", "test"):
        split_dir = os.path.join(chest_dir, split)
        if not os.path.isdir(split_dir):
            continue
        normal_files += _list_images(os.path.join(split_dir, "NORMAL"))
        pneumonia_files += _list_images(os.path.join(split_dir, "PNEUMONIA"))

    rng = np.random.RandomState(seed)
    rng.shuffle(normal_files)
    rng.shuffle(pneumonia_files)
    if max_per_split:
        normal_files = normal_files[:max_per_split]
        pneumonia_files = pneumonia_files[:max_per_split]

    def split_list(files, fracs=(0.8, 0.1, 0.1)):
        n = len(files)
        n_train = int(n * fracs[0])
        n_val = int(n * fracs[1])
        return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]

    n_tr, n_va, n_te = split_list(normal_files)
    p_tr, p_va, p_te = split_list(pneumonia_files)

    X_train, y_train = _load_and_label(n_tr, p_tr, image_size)
    X_val, y_val = _load_and_label(n_va, p_va, image_size)
    X_test, y_test = _load_and_label(n_te, p_te, image_size)

    print(f"  ✅ Chest X-ray Pneumonia: {len(X_train):,} train | "
          f"{len(X_val):,} val | {len(X_test):,} test")
    return X_train, y_train, X_val, y_val, X_test, y_test


def _find_chest_xray_root(root):
    for dirpath, dirnames, _ in os.walk(root):
        if "train" in dirnames and ("test" in dirnames or "val" in dirnames):
            return dirpath
    return root


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def _load_and_label(normal_files, pneumonia_files, image_size):
    n, p = len(normal_files), len(pneumonia_files)
    X = np.zeros((n + p, image_size, image_size, 3), dtype="float32")
    y = np.zeros((n + p,), dtype="int64")
    for i, f in enumerate(normal_files):
        X[i] = _load_image(f, image_size)
        y[i] = 0
    for i, f in enumerate(pneumonia_files):
        X[n + i] = _load_image(f, image_size)
        y[n + i] = 1
    perm = np.random.RandomState(0).permutation(len(X))
    return X[perm], y[perm]


# --------------------------------------------------------------------------
# Secondary datasets — used only for cross-dataset generalization evaluation
# --------------------------------------------------------------------------

def load_covid_radiography(image_size=128, seed=42, max_per_class=300):
    """Binary-mapped: NORMAL=0, everything else (COVID/Viral Pneumonia/Lung
    Opacity) = 1 (abnormal), so it's directly comparable to the primary task.
    """
    root = resolve_dataset_path(
        "tawsifurrahman/covid19-radiography-database",
        local_candidates=["./data/COVID-19_Radiography_Dataset",
                           "./data/covid19-radiography-database"],
    )
    if root is None:
        print("  ⚠️  COVID-19 Radiography dataset unavailable — skipping "
              "(continuing with what loaded).")
        return None

    base = _find_covid_root(root)
    normal_dir = _first_existing(base, ["Normal/images", "Normal"])
    abnormal_dirs = [
        _first_existing(base, ["COVID/images", "COVID"]),
        _first_existing(base, ["Viral Pneumonia/images", "Viral Pneumonia"]),
        _first_existing(base, ["Lung_Opacity/images", "Lung_Opacity"]),
    ]
    abnormal_dirs = [d for d in abnormal_dirs if d]
    if not normal_dir or not abnormal_dirs:
        print("  ⚠️  COVID-19 Radiography folder layout not recognized — skipping.")
        return None

    rng = np.random.RandomState(seed)
    X_n, y_n = _load_folder_as_class(normal_dir, 0, image_size, limit=max_per_class)
    abn_chunks = []
    per_dir = max(1, max_per_class // max(len(abnormal_dirs), 1))
    for d in abnormal_dirs:
        Xc, yc = _load_folder_as_class(d, 1, image_size, limit=per_dir)
        abn_chunks.append((Xc, yc))
    X_a = np.concatenate([c[0] for c in abn_chunks], axis=0)
    y_a = np.concatenate([c[1] for c in abn_chunks], axis=0)

    X = np.concatenate([X_n, X_a], axis=0)
    y = np.concatenate([y_n, y_a], axis=0)
    perm = rng.permutation(len(X))
    print(f"  ✅ COVID-19 Radiography: loaded {len(X):,} images "
          f"(Normal vs Abnormal) for generalization testing")
    return X[perm], y[perm]


def _find_covid_root(root):
    for dirpath, dirnames, _ in os.walk(root):
        if "Normal" in dirnames or "COVID" in dirnames:
            return dirpath
    return root


def _first_existing(base, options):
    for opt in options:
        p = os.path.join(base, opt)
        if os.path.isdir(p):
            return p
    return None


def load_rsna_pneumonia(image_size=128, seed=42, max_per_class=300):
    """Uses the stage 2 labels CSV: Target 0=Normal, 1=Pneumonia."""
    root = resolve_dataset_path(
        "c/rsna-pneumonia-detection-challenge",
        local_candidates=["./data/rsna-pneumonia-detection-challenge"],
    )
    if root is None:
        print("  ⚠️  RSNA Pneumonia dataset unavailable — skipping "
              "(continuing with what loaded).")
        return None

    import pandas as pd
    csv_candidates = glob.glob(os.path.join(root, "**", "*labels*.csv"), recursive=True)
    img_dir_candidates = glob.glob(os.path.join(root, "**", "stage_2_train_images"), recursive=True)
    if not csv_candidates or not img_dir_candidates:
        print("  ⚠️  RSNA folder layout not recognized — skipping.")
        return None

    df = pd.read_csv(csv_candidates[0]).drop_duplicates("patientId")
    img_dir = img_dir_candidates[0]

    rng = np.random.RandomState(seed)
    normal_ids = df[df["Target"] == 0]["patientId"].sample(
        frac=1, random_state=seed).tolist()[:max_per_class]
    pneu_ids = df[df["Target"] == 1]["patientId"].sample(
        frac=1, random_state=seed).tolist()[:max_per_class]

    X, y = [], []
    for pid in normal_ids + pneu_ids:
        dcm_path = os.path.join(img_dir, f"{pid}.dcm")
        if not os.path.exists(dcm_path):
            continue
        try:
            import pydicom
            arr = pydicom.dcmread(dcm_path).pixel_array
            from PIL import Image
            img = Image.fromarray(arr).convert("RGB").resize((image_size, image_size))
            X.append(np.asarray(img, dtype="float32") / 255.0)
            y.append(int(df.loc[df["patientId"] == pid, "Target"].iloc[0]))
        except Exception:
            continue

    if not X:
        print("  ⚠️  RSNA: no readable DICOM images found — skipping.")
        return None

    X, y = np.array(X), np.array(y)
    perm = rng.permutation(len(X))
    print(f"  ✅ RSNA Pneumonia: loaded {len(X):,} images for generalization testing")
    return X[perm], y[perm]


# --------------------------------------------------------------------------
# Non-IID hospital simulation — Part 4 of the spec
# --------------------------------------------------------------------------

HOSPITAL_CONFIG = {
    0: {"name": "Apollo Hospital", "normal_pct": 0.75, "data_fraction": 0.25, "comm_score": 0.90},
    1: {"name": "AIIMS Delhi", "normal_pct": 0.30, "data_fraction": 0.20, "comm_score": 0.70},
    2: {"name": "Fortis Hospital", "normal_pct": 0.55, "data_fraction": 0.22, "comm_score": 0.85},
    3: {"name": "Manipal Hospital", "normal_pct": 0.85, "data_fraction": 0.18, "comm_score": 0.60},
    4: {"name": "Medanta Hospital", "normal_pct": 0.40, "data_fraction": 0.15, "comm_score": 0.75},
}


def _apply_min_class_constraint(n_normal, n_pneumonia, n_total, min_class_pct):
    """Clip a hospital's (n_normal, n_pneumonia) pair so both classes hold
    at least `min_class_pct` of n_total, preserving n_total.
    """
    floor = int(np.ceil(min_class_pct * n_total))
    n_normal = max(floor, min(n_normal, n_total - floor))
    n_pneumonia = n_total - n_normal
    return n_normal, n_pneumonia


def create_hospital_splits(X, y, hospital_config, seed=42, min_class_pct=0.10,
                            local_val_frac=0.15):
    """Splits (X, y) into Non-IID per-hospital shards with skewed class
    balance and unequal sizes, honouring `hospital_config`. Each hospital
    also gets a small internal validation split (used to compute its own
    A_i participation-score term without exposing raw data to the server).
    """
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y == 0)[0]
    pneumonia_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    rng.shuffle(pneumonia_idx)
    total_normal, total_pneumonia = len(normal_idx), len(pneumonia_idx)
    N = len(X)

    ids = sorted(hospital_config.keys())
    n_i = {i: int(round(hospital_config[i]["data_fraction"] * N)) for i in ids}

    # Shrink hospital sizes up front if they ask for more than exists in total.
    total_requested = sum(n_i.values())
    total_available = total_normal + total_pneumonia
    if total_requested > total_available:
        scale = total_available / total_requested
        n_i = {i: max(2, int(n_i[i] * scale)) for i in ids}

    # The floor must be reserved for BOTH classes before any proportional
    # allocation happens, otherwise sequential consumption can starve a
    # later hospital below the minimum even though the math "on paper"
    # looked fine. Shrink hospital sizes (rare, only for tiny pools) until
    # every hospital's floor is simultaneously satisfiable from both pools.
    floor_i = {i: max(1, int(np.ceil(min_class_pct * n_i[i]))) for i in ids}
    for _ in range(50):
        if sum(floor_i.values()) <= total_normal and sum(floor_i.values()) <= total_pneumonia:
            break
        n_i = {i: max(2, int(n_i[i] * 0.9)) for i in ids}
        floor_i = {i: max(1, int(np.ceil(min_class_pct * n_i[i]))) for i in ids}

    # Phase 1 — reserve exactly `floor_i` images of each class per hospital.
    reserved_normal = {i: floor_i[i] for i in ids}
    reserved_pneumonia = {i: floor_i[i] for i in ids}
    remaining_normal = total_normal - sum(reserved_normal.values())
    remaining_pneumonia = total_pneumonia - sum(reserved_pneumonia.values())

    # Phase 2 — distribute what's left according to each hospital's desired
    # normal_pct, scaled proportionally to fit whatever pool remains.
    bonus_total = {i: max(0, n_i[i] - 2 * floor_i[i]) for i in ids}
    bonus_normal_want = {
        i: int(round(bonus_total[i] * hospital_config[i]["normal_pct"])) for i in ids
    }
    bonus_pneumonia_want = {i: bonus_total[i] - bonus_normal_want[i] for i in ids}

    def _scale_to_pool(wants, pool):
        total_want = sum(wants.values())
        if total_want <= pool or total_want == 0:
            return dict(wants)
        scale = pool / total_want
        return {i: int(w * scale) for i, w in wants.items()}

    bonus_normal = _scale_to_pool(bonus_normal_want, remaining_normal)
    bonus_pneumonia = _scale_to_pool(bonus_pneumonia_want, remaining_pneumonia)

    take_normal = {i: reserved_normal[i] + bonus_normal[i] for i in ids}
    take_pneumonia = {i: reserved_pneumonia[i] + bonus_pneumonia[i] for i in ids}

    hospitals = []
    n_ptr, p_ptr = 0, 0
    for i in ids:
        cfg = hospital_config[i]
        idx_normal = normal_idx[n_ptr:n_ptr + take_normal[i]]
        idx_pneumonia = pneumonia_idx[p_ptr:p_ptr + take_pneumonia[i]]
        n_ptr += take_normal[i]
        p_ptr += take_pneumonia[i]

        h_idx = np.concatenate([idx_normal, idx_pneumonia])
        rng.shuffle(h_idx)
        X_h, y_h = X[h_idx], y[h_idx]

        n_val = max(1, int(len(X_h) * local_val_frac))
        X_val_h, y_val_h = X_h[:n_val], y_h[:n_val]
        X_tr_h, y_tr_h = X_h[n_val:], y_h[n_val:]

        n_total_h = len(y_h)
        actual_normal_pct = float(np.mean(y_h == 0)) if n_total_h else 0.0
        hospitals.append({
            "id": i,
            "name": cfg["name"],
            "comm_score": cfg["comm_score"],
            "X": X_tr_h, "y": y_tr_h,
            "X_val": X_val_h, "y_val": y_val_h,
            "n_total": n_total_h,
            "actual_normal_pct": actual_normal_pct,
        })

    for h in hospitals:
        pct_normal = h["actual_normal_pct"] * 100
        pct_pneumonia = 100 - pct_normal
        ok = (h["actual_normal_pct"] >= min_class_pct - 1e-6 and
              (1 - h["actual_normal_pct"]) >= min_class_pct - 1e-6)
        print(f"  {h['name']:<18}: {h['n_total']:,} images "
              f"(Normal: {pct_normal:.0f}% | Pneumonia: {pct_pneumonia:.0f}%) "
              f"{'✅' if ok else '⚠️'}")

    return hospitals

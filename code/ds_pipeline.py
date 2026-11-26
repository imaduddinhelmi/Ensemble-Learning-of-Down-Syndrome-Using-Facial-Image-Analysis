"""Reproducible retraining pipeline for the BEEI revision (Paper ID 13183).

Addresses reviewer requests: fixed seed + saved split (B2), near-duplicate grouping (B1),
augmentation ablation (B3), extra ensemble pairs/triples (B5), ROC/PR/AUC, bootstrap CI,
McNemar, repeated CV (B6), compute cost (C3) and Grad-CAM (C1).
Fixes the original bug: sigmoid output + BinaryCrossentropy(from_logits=True).

TensorFlow is imported lazily so the data/statistics parts can be tested without it.
Positive class: Down syndrome (y = 1).
"""
import itertools
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

POS_DIR, NEG_DIR = "downSyndrome", "healty"   # folder names in the Kaggle archive
IMG = 224
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# name -> (keras.applications constructor name, preprocessing mode, last conv layer for Grad-CAM)
BACKBONES = {
    "DenseNet201": ("DenseNet201", "torch", "relu"),
    "Xception": ("Xception", "tf", "block14_sepconv2_act"),
    "InceptionV3": ("InceptionV3", "tf", "mixed10"),
    "InceptionResNetV2": ("InceptionResNetV2", "tf", "conv_7b_ac"),
    # modern CNN comparator (A2/C2); has built-in preprocessing, so raw 0..255 input; None = use base.output
    "ConvNeXtTiny": ("ConvNeXtTiny", "none", None),
}
# default model names Keras gives each application (used to find the nested backbone)
BASE_LAYER = {"DenseNet201": "densenet201", "Xception": "xception", "InceptionV3": "inception_v3",
              "InceptionResNetV2": "inception_resnet_v2", "ConvNeXtTiny": "convnext_tiny"}
SHORT = {"DenseNet201": "D", "Xception": "X", "InceptionV3": "I3", "InceptionResNetV2": "IR",
         "ConvNeXtTiny": "CNT"}


def run_name(names, policy):
    return "+".join(SHORT[n] for n in names) + f"__{policy}"


# ----------------------------------------------------------------------------- data

def list_images(root):
    rows = []
    for folder, y in ((POS_DIR, 1), (NEG_DIR, 0)):
        d = Path(root) / folder
        assert d.is_dir(), f"missing class folder: {d}"
        rows += [(str(p), y) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return pd.DataFrame(rows, columns=["path", "y"])


def _popcount64(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    return np.unpackbits(a.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def group_near_duplicates(hashes, max_dist=4):
    """Union-find over 64-bit perceptual hashes; images within max_dist bits share a group."""
    h = np.asarray(hashes, dtype=np.uint64)
    parent = np.arange(len(h))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h) - 1):
        for j in (np.nonzero(_popcount64(h[i] ^ h[i + 1:]) <= max_dist)[0] + i + 1):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(len(h))])


def add_groups(df, max_dist=4):
    """Adds phash + group columns; drops groups whose images carry conflicting labels."""
    import imagehash
    from PIL import Image
    df = df.copy()
    df["phash"] = [int(str(imagehash.phash(Image.open(p).convert("L"))), 16) for p in df.path]
    df["group"] = group_near_duplicates(df.phash.values, max_dist)
    conflict = df.groupby("group").y.nunique() > 1
    report = {
        "images": len(df),
        "groups": int(df.group.nunique()),
        "images_in_duplicate_groups": int(df.group.duplicated(keep=False).sum()),
        "cross_label_groups_dropped": int(conflict.sum()),
        "max_hamming_distance": max_dist,
    }
    df = df[~df.group.isin(conflict[conflict].index)].reset_index(drop=True)
    report["images_after_drop"] = len(df)
    return df, report


def split_groups(df, seed, val=0.15, test=0.15):
    """Stratified, group-aware train/val/test split; whole duplicate groups stay in one subset."""
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=df.index)
    for y in (0, 1):
        sizes = df[df.y == y].groupby("group").size()
        order = rng.permutation(sizes.index.values)
        cum = sizes.loc[order].cumsum() / sizes.sum()
        test_g = order[cum.values <= test]
        val_g = order[(cum.values > test) & (cum.values <= test + val)]
        out[df.group.isin(test_g) & (df.y == y)] = "test"
        out[df.group.isin(val_g) & (df.y == y)] = "val"
    return out


def cv_folds(df, k=5, repeats=1, seed=42, val=0.15):
    """Repeated stratified group k-fold. Yields (repeat, fold, split Series)."""
    from sklearn.model_selection import StratifiedGroupKFold
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(sgkf.split(df, df.y, df.group)):
            split = pd.Series("train", index=df.index)
            split.iloc[te] = "test"
            inner = split_groups(df.iloc[tr], seed + 1000 * r + f, val=val, test=0.0)
            split.iloc[tr] = inner.values
            yield r, f, split


# ----------------------------------------------------------------------------- statistics

def confusion(y, p, thr):
    yhat = (np.asarray(p) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    return tp, fn, fp, tn


def _div(a, b):
    return a / b if b else float("nan")


def metrics(y, p, thr=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    tp, fn, fp, tn = confusion(y, p, thr)
    prec, sens = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "threshold": thr,
        "accuracy": _div(tp + tn, tp + tn + fp + fn), "precision": prec, "sensitivity": sens,
        "specificity": _div(tn, tn + fp), "npv": _div(tn, tn + fn),
        "f1": _div(2 * prec * sens, prec + sens),
        "auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
        "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
    }


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bootstrap_ci(y, p, thr, keys=("accuracy", "sensitivity", "specificity", "f1", "auc", "ap"),
                 B=2000, seed=0):
    """Percentile bootstrap CIs, resampling test images (stratified by class)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    vals = {k: [] for k in keys}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = metrics(y[idx], p[idx], thr)
        for k in keys:
            vals[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def mcnemar_exact(y, pa, pb, thr_a=0.5, thr_b=0.5):
    """Exact McNemar on paired correctness; returns (b, c, p-value)."""
    from scipy.stats import binomtest
    ca = (np.asarray(pa) >= thr_a).astype(int) == np.asarray(y)
    cb = (np.asarray(pb) >= thr_b).astype(int) == np.asarray(y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(pval)


def paired_auc_diff(y, pa, pb, B=2000, seed=0):
    """Paired bootstrap 95% CI of AUC(a) - AUC(b) on the same test images."""
    from sklearn.metrics import roc_auc_score
    y, pa, pb = map(np.asarray, (y, pa, pb))
    rng = np.random.default_rng(seed)
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    d = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def soft_vote(preds, members):
    """preds: DataFrame with columns path,y,split and one probability column per run."""
    return preds[members].mean(axis=1)


def collect_preds(out_dir, tags):
    """Merges preds.csv of several runs (same split) into one wide DataFrame."""
    merged = None
    for t in tags:
        p = pd.read_csv(Path(out_dir) / t / "preds.csv")
        merged = p if merged is None else merged.merge(p, on=["path", "y", "split"], how="inner")
    return merged


def summarize(preds, configs, B=2000):
    """One row per config: metrics at 0.5 and at the validation-Youden threshold, with CIs."""
    val, test = preds[preds.split == "val"], preds[preds.split == "test"]
    rows = []
    for name in configs:
        thr_y = youden_threshold(val.y, val[name])
        for label, thr in (("0.5", 0.5), ("youden_val", thr_y)):
            m = metrics(test.y, test[name], thr)
            ci = bootstrap_ci(test.y, test[name], thr, B=B)
            row = {"config": name, "threshold_rule": label, **m}
            for k, (lo, hi) in ci.items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
            row["acc_wilson_lo"], row["acc_wilson_hi"] = wilson(m["TP"] + m["TN"], len(test))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- TensorFlow

def set_seed(seed, deterministic=False):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:  # slower; some GPU ops may refuse to run deterministically
        tf.config.experimental.enable_op_determinism()


def make_dataset(df, training, policy="conservative", bs=32, seed=42):
    import tensorflow as tf
    from tensorflow.keras import layers

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG, IMG), antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8), y

    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.y.values.astype("float32")))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(bs).map(lambda x, y: (tf.cast(x, tf.float32), y))
    if training and policy != "none":
        aug = augmenter(policy, seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def augmenter(policy, seed):
    from tensorflow import keras
    from tensorflow.keras import layers
    if policy == "original":  # settings of the submitted manuscript
        ops = [layers.RandomTranslation(0.5, 0.5, fill_mode="constant", fill_value=0.0, seed=seed),
               layers.RandomFlip("horizontal_and_vertical", seed=seed + 1),
               layers.RandomZoom(0.15, fill_mode="constant", fill_value=0.0, seed=seed + 2)]
    elif policy == "conservative":  # landmark-preserving: no vertical flip, small shifts
        ops = [layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
               layers.RandomFlip("horizontal", seed=seed + 1),
               layers.RandomRotation(10 / 360, fill_mode="reflect", seed=seed + 2),
               layers.RandomZoom(0.1, fill_mode="reflect", seed=seed + 3)]
    else:
        raise ValueError(policy)
    return keras.Sequential(ops, name=f"aug_{policy}")


def build_model(names, dropout_single=0.2, dropout_fusion=0.3):
    """1 backbone -> GAP-dropout-sigmoid; >=2 backbones -> feature concatenation head."""
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input((IMG, IMG, 3), name="image")  # raw pixels 0..255
    feats = []
    for n in names:
        ctor, mode, _ = BACKBONES[n]
        base = getattr(keras.applications, ctor)(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3))
        if mode == "none":
            x = inp
        elif mode == "tf":
            x = layers.Rescaling(1 / 127.5, offset=-1.0, name=f"{n}_pre")(inp)
        else:
            x = layers.Rescaling(1 / 255.0, name=f"{n}_pre")(inp)
            x = layers.Normalization(mean=[0.485, 0.456, 0.406],
                                     variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], name=f"{n}_norm")(x)
        feats.append(layers.GlobalAveragePooling2D(name=f"{n}_gap")(base(x)))
    if len(names) == 1:
        x = layers.Dropout(dropout_single, name="drop")(feats[0])
    else:
        x = layers.Concatenate(name="concat")(feats)
        x = layers.Dense(256, activation="relu", name="fc256")(x)
        x = layers.Dropout(dropout_fusion, name="drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="out")(x)
    return keras.Model(inp, out, name=run_name(names, "model"))


def train_run(names, df, split, out_dir, policy="conservative", seed=42, epochs=30, lr=1e-5,
              bs=32, patience=5, tag=None):
    """Trains one configuration; writes preds.csv, history.csv, meta.json, weights. Resumable."""
    import tensorflow as tf
    from tensorflow import keras
    tag = tag or run_name(names, policy)
    run_dir = Path(out_dir) / tag
    if (run_dir / "preds.csv").exists():
        print(f"[skip] {tag} already done")
        return pd.read_csv(run_dir / "preds.csv")
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    tr, va = df[split == "train"], df[split == "val"]
    model = build_model(names)
    model.compile(optimizer=keras.optimizers.Adamax(lr),
                  loss=keras.losses.BinaryCrossentropy(),  # output is already a sigmoid probability
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           keras.callbacks.CSVLogger(str(run_dir / "history.csv"))]
    t0 = time.time()
    model.fit(make_dataset(tr, True, policy, bs, seed), validation_data=make_dataset(va, False, bs=bs),
              epochs=epochs, callbacks=cbs, verbose=2)
    train_s = time.time() - t0
    model.save_weights(str(run_dir / "model.weights.h5"))
    ev = df[split.isin(["val", "test"])].copy()
    ev["split"] = split[ev.index]
    ev[tag] = model.predict(make_dataset(ev, False, bs=bs), verbose=0).ravel()
    ev[["path", "y", "split", tag]].to_csv(run_dir / "preds.csv", index=False)
    meta = {"names": names, "policy": policy, "seed": seed, "lr": lr, "batch_size": bs,
            "epochs_run": len(pd.read_csv(run_dir / "history.csv")), "train_seconds": train_s,
            "params": int(model.count_params()), "tf_version": tf.__version__,
            "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    keras.backend.clear_session()
    return ev[["path", "y", "split", tag]]


def load_model(names, run_dir):
    model = build_model(names)
    model.load_weights(str(Path(run_dir) / "model.weights.h5"))
    return model


def benchmark(model, n=50, tflite_dir=None):
    """Params, fp32 size, batch-1 latency (GPU if present, and CPU), optional TFLite size."""
    import tensorflow as tf
    x = np.random.uniform(0, 255, (1, IMG, IMG, 3)).astype("float32")
    f = tf.function(lambda t: model(t, training=False))
    res = {"params_M": model.count_params() / 1e6, "fp32_MB": model.count_params() * 4 / 2 ** 20}
    devices = ["/CPU:0"] + (["/GPU:0"] if tf.config.list_physical_devices("GPU") else [])
    for dev in devices:
        with tf.device(dev):
            for _ in range(5):
                f(x)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter(); f(x).numpy(); ts.append(time.perf_counter() - t0)
        res[f"latency_ms_{dev.strip('/').split(':')[0]}"] = 1000 * float(np.median(ts))
    if tflite_dir:
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range int8 weights
            p = Path(tflite_dir) / f"{model.name}_int8.tflite"
            p.write_bytes(conv.convert())
            res["tflite_int8_MB"] = p.stat().st_size / 2 ** 20
        except Exception as e:  # conversion of very large graphs can fail on Colab RAM
            res["tflite_error"] = str(e)[:200]
    return res


def gradcam(model, names, img_uint8, branch):
    """Grad-CAM for one backbone branch; img is HxWx3 raw pixels. Returns 224x224 map in [0,1] and p(DS)."""
    import tensorflow as tf
    from tensorflow import keras
    x = tf.convert_to_tensor(np.asarray(img_uint8, dtype="float32")[None])
    with tf.GradientTape() as tape:
        feats, conv_target = [], None
        for n in names:
            ctor, mode, conv_name = BACKBONES[n]
            base = model.get_layer(BASE_LAYER[n])
            h = x if mode == "none" else model.get_layer(f"{n}_pre")(x)
            if mode == "torch":
                h = model.get_layer(f"{n}_norm")(h)
            conv_out = base.output if conv_name is None else base.get_layer(conv_name).output
            fm = keras.Model(base.inputs, [conv_out, base.output])
            conv, out = fm(h, training=False)
            if n == branch:
                conv_target = conv
                tape.watch(conv_target)
            feats.append(model.get_layer(f"{n}_gap")(out))
        if len(names) == 1:
            z = feats[0]
        else:
            z = model.get_layer("fc256")(model.get_layer("concat")(feats))
        p = model.get_layer("out")(model.get_layer("drop")(z, training=False))[:, 0]
    g = tape.gradient(p, conv_target)[0]
    w = tf.reduce_mean(g, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(conv_target[0] * w, axis=-1))
    cam = tf.image.resize(cam[..., None], (IMG, IMG))[..., 0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(p.numpy()[0])


# ----------------------------------------------------------------------------- self-check (no TF needed)

def _selfcheck():
    rng = np.random.default_rng(0)
    # duplicate grouping: two near-identical hashes + one far
    g = group_near_duplicates([0b1111, 0b1110, 2 ** 63], max_dist=1)
    assert g[0] == g[1] != g[2]
    # group split keeps groups intact and is stratified
    df = pd.DataFrame({"y": rng.integers(0, 2, 600), "group": rng.integers(0, 400, 600)})
    df["y"] = df.groupby("group").y.transform("first")
    s = split_groups(df, seed=1)
    assert df.assign(s=s).groupby("group").s.nunique().max() == 1
    frac = s.value_counts(normalize=True)
    assert abs(frac["test"] - 0.15) < 0.05 and abs(frac["val"] - 0.15) < 0.05
    assert split_groups(df, seed=1).equals(s)  # reproducible
    for _, _, sp in cv_folds(df, k=5, repeats=1):
        assert df.assign(s=sp).groupby("group").s.nunique().max() == 1
    # metrics reproduce the manuscript's verified concatenation confusion matrix
    y = np.r_[np.ones(225), np.zeros(226)]
    p = np.r_[np.ones(223), np.zeros(2), np.ones(57), np.zeros(169)] * 0.9 + 0.05
    m = metrics(y, p)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (223, 2, 57, 169)
    assert round(m["sensitivity"], 3) == 0.991 and round(m["specificity"], 3) == 0.748
    lo, hi = wilson(392, 451)
    assert round(lo, 3) == 0.835 and round(hi, 3) == 0.897
    b, c, pv = mcnemar_exact(y, p, 1 - p)
    assert b == 392 and c == 59 and pv < 1e-6
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck()

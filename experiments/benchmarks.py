"""Load the saved outputs of the seven frozen models as GUARD conditions.

Each loader reads files from ``$GUARD_ARTIFACTS`` (default ``data/artifacts``),
fetched by ``data/download.sh``, and returns a ``Condition``.
``CONDITIONS`` lists the conditions and seeds used in the paper.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from guard import Condition
from guard.losses import softmax

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("GUARD_ARTIFACTS", ROOT / "data" / "artifacts"))


def _thirds(n_pool, n_deploy, seed):
    """Pool first, then the deployment rows split at random into fit, conf, test."""
    perm = np.random.default_rng(seed).permutation(n_deploy) + n_pool
    k = n_deploy // 3
    return np.arange(n_pool), perm[:k], perm[k:2 * k], perm[2 * k:]


# CMU-MOSEI, CMAD. The model outputs a sentiment score, read as a logit.
# Accuracy is computed on non-neutral clips, as in the published protocol.

def mosei(mask, seed):
    d = ARTIFACTS / "mosei_cmad/dumps"
    p = np.load(d / "student_preds.npz", allow_pickle=True)
    r = np.load(d / "raw_features.npz", allow_pickle=True)
    raw = r["test_y"].reshape(-1)
    feats = {"a": r["test_ac"].astype(np.float64), "v": r["test_vis"].astype(np.float64)}

    def probs(m):
        e = 1 / (1 + np.exp(-p[f"test_{m}"].astype(np.float64)))
        return np.stack([1 - e, e], 1)

    split = tuple(np.array_split(np.random.default_rng(seed).permutation(len(raw)), 4))
    return Condition(probs(mask), np.concatenate([feats[m] for m in mask], 1), (raw > 0).astype(int),
                     split, richer=probs("tav"), keep=raw != 0, targets=("hard", "cross"))


# IEMOCAP, MoMKE. Five folds; each fold's evaluation videos are split in thirds.
# Every mask file stores rows in its own order, so rows are matched by
# (video, occurrence). The files store logits.

def _row_keys(vid):
    seen, out = {}, []
    for v in vid:
        seen[v] = seen.get(v, 0) + 1
        out.append((v, seen[v] - 1))
    return out


def _reorder(ref_keys, keys):
    at = {k: i for i, k in enumerate(ref_keys)}
    return np.array([at[k] for k in keys])


def iemocap(fold, mask, cut):
    d = ARTIFACTS / "iemocap_momke/folds"
    ref = np.load(d / f"fold{fold}_atv.npz", allow_pickle=True)
    cur = np.load(d / f"fold{fold}_{mask}.npz", allow_pickle=True)
    n = len(ref["labels"])

    # evaluation rows in the order of the full-mask file
    idx = _reorder(_row_keys(ref["vid"]), _row_keys(cur["vid"]))
    probs = np.zeros((n, cur["probs"].shape[1]))
    probs[idx] = softmax(cur["probs"])
    feats = np.zeros((n, cur["feats"].shape[1]))
    feats[idx] = cur["feats"]
    richer = softmax(ref["probs"])

    # pool rows in the order of this mask's file
    pool_idx = _reorder(_row_keys(ref["pool_vid"]), _row_keys(cur["pool_vid"]))
    assert np.array_equal(ref["pool_labels"][pool_idx], cur["pool_labels"]), "pool rows do not align"
    n_pool = len(cur["pool_labels"])

    videos = np.unique(ref["vid"])
    groups = np.array_split(videos[np.random.default_rng(cut).permutation(len(videos))], 3)
    fit, conf, test = (np.where(np.isin(ref["vid"], g))[0] + n_pool for g in groups)
    return Condition(
        probs=np.concatenate([softmax(cur["pool_probs"]), probs]),
        features=np.concatenate([cur["pool_feats"].astype(np.float64), feats]),
        labels=np.concatenate([cur["pool_labels"], ref["labels"]]),
        split=(np.arange(n_pool), fit, conf, test),
        richer=np.concatenate([softmax(ref["pool_probs"])[pool_idx], richer]),
        targets=("hard", "cross"))


# OPPORTUNITY, DeepConvLSTM. Windows overlap in time, so the deployment stream is
# cut into blocks of 200 windows and whole blocks are assigned to the four roles.

def _block_split(n, seed, block=200, gap=2):
    rng = np.random.default_rng(seed)
    blocks = [np.arange(i, min(i + block, n) - gap) for i in range(0, n, block)]
    blocks = [b for b in blocks if len(b)]
    roles = (np.arange(len(blocks)) % 4)[rng.permutation(len(blocks))]
    return tuple(np.concatenate([blocks[j] for j in np.flatnonzero(roles == r)]) for r in range(4))


def opportunity(config, seed):
    d = ARTIFACTS / "opportunity_dcl_v2"
    y = np.load(d / "deploy_y.npy")
    return Condition(
        probs=np.load(d / f"probs_condition_specialist_{config}_deploy_s{seed}.npy").astype(np.float64),
        features=np.load(d / f"retfeat_{config}_deploy.npy").astype(np.float64),
        labels=y, split=_block_split(len(y), seed),
        richer=np.load(d / f"richer_deploy_s{seed}.npy").astype(np.float64),
        targets=("hard", "cross"))


# AVE, AV-att (released weights). The pool is the released training set.

def ave(mask, seed):
    d = ARTIFACTS / "ave_av_att/dumps"
    z = np.load(d / f"AV_att_{mask}.npz", allow_pickle=True)
    r = np.load(d / f"AV_att_{mask}_retrieval_paper.npz", allow_pickle=True)
    probs = z["probs"].reshape(-1, z["probs"].shape[-1]).astype(np.float64)
    labels = z["labels"].reshape(-1, z["labels"].shape[-1]).argmax(1)
    n_pool = len(r["pool_labels"])
    return Condition(
        probs=np.concatenate([r["pool_probs"].astype(np.float64), probs]),
        features=np.concatenate([r["pool_features"].astype(np.float64), r["deploy_features"].astype(np.float64)]),
        labels=np.concatenate([r["pool_labels"], labels]),
        split=_thirds(n_pool, len(labels), seed))


# NinaPro DB5, sEMG CNN. One model per subject; the condition is the number of
# electrodes kept. The pool is the subject's training repetitions.

def ninapro_subjects():
    """The first ten subjects of each training seed."""
    out = []
    for seed_dir in sorted(glob.glob(str(ARTIFACTS / "ninapro_cnn/seed*"))):
        for s in sorted(glob.glob(f"{seed_dir}/subject*"))[:10]:
            out.append(os.path.relpath(s, ARTIFACTS))
    return out


def ninapro(subject, electrodes):
    p = np.load(ARTIFACTS / subject / "preds.npz", allow_pickle=True)
    e = np.load(ARTIFACTS / subject / "masked_embeddings.npz", allow_pickle=True)
    probs = np.concatenate([p[f"train_{electrodes}"], p[f"sess1_{electrodes}"]]).astype(np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum(1, keepdims=True)
    return Condition(
        probs=probs,
        features=np.concatenate([e[f"train_{electrodes}"], e[f"sess1_{electrodes}"]]).astype(np.float64),
        labels=np.concatenate([p["train_y"], p["sess1_y"]]).astype(int),
        split=_thirds(len(p["train_y"]), len(p["sess1_y"]), seed=0))


# PTB-XL, resnet1d_wang. Multi-label, scored per label with the Bernoulli loss.
# ``ladder`` is a lead subset; ``feat`` names the retrieval features used for it.

def ptbxl(ladder, feat, seed):
    d = ARTIFACTS / "ptbxl_resnet1d_wang"
    p = np.load(d / "preds.npz", allow_pickle=True)
    r = np.load(d / "raw_features.npz", allow_pickle=True)
    return Condition(
        probs=np.concatenate([p[f"train_{ladder}"], p[f"sess1_{ladder}"]]).astype(np.float64),
        features=np.concatenate([r[f"train_{feat}"], r[f"sess1_{feat}"]]).astype(np.float64),
        labels=np.concatenate([r["train_y"], r["sess1_y"]]).astype(np.float64),
        split=_thirds(len(r["train_y"]), len(r["sess1_y"]), seed),
        loss="bernoulli")


# DrugBAN. The protein is truncated to 25% or 50%, or the drug is reduced to its
# Murcko scaffold with 50% of the protein. The released calibration rows are split
# in half into fit and conf.

def drugban(dataset, degradation, seed):
    d = ARTIFACTS / "drugban_processed" / dataset
    z = np.load(d / f"{degradation}.npz", allow_pickle=True)
    full = np.load(d / "full.npz", allow_pickle=True)
    parts = ("pool", "calib", "test")
    n_pool, n_cal, n_test = (len(z[f"{s}_labels"]) for s in parts)
    perm = np.random.default_rng(seed).permutation(n_cal) + n_pool
    return Condition(
        probs=np.concatenate([z[f"{s}_probs"] for s in parts]).astype(np.float64),
        features=np.concatenate([z[f"{s}_feats"] for s in parts]).astype(np.float64),
        labels=np.concatenate([z[f"{s}_labels"] for s in parts]),
        split=(np.arange(n_pool), perm[:n_cal // 2], perm[n_cal // 2:], np.arange(n_test) + n_pool + n_cal),
        richer=np.concatenate([full[f"{s}_probs"] for s in parts]).astype(np.float64),
        targets=("hard", "cross"))


LOADERS = dict(mosei=mosei, iemocap=iemocap, opportunity=opportunity, ave=ave,
               ninapro=ninapro, ptbxl=ptbxl, drugban=drugban)

_MASKS = ("a", "t", "v", "at", "av", "tv", "atv")
_DRUGBAN = ([(f"drugban_{d}_random_s42", c) for d in ("biosnap", "bindingdb")
             for c in ("prot25", "prot50", "scaffold")]
            + [("drugban_human_random_s42", c) for c in ("prot25", "prot50")]
            + [("drugban_biosnap_cluster_s42", "prot50")])


def conditions(name):
    """Keyword arguments of every condition and seed of one benchmark."""
    if name == "mosei":
        return [dict(mask=m, seed=s) for m in ("a", "v", "av") for s in range(10)]
    if name == "iemocap":
        return [dict(fold=f, mask=m, cut=c) for f in range(5) for m in _MASKS for c in range(5)]
    if name == "opportunity":
        configs = ("low_cost_accels_only", "no_imu_family", "no_shoes", "severe_three_sensors")
        return [dict(config=c, seed=s) for c in configs for s in range(3)]
    if name == "ave":
        return [dict(mask=m, seed=s) for m in ("audio_only", "visual_only") for s in range(3)]
    if name == "ninapro":
        return [dict(subject=s, electrodes=e) for s in ninapro_subjects() for e in ("12", "8", "6", "4")]
    if name == "ptbxl":
        pairs = (("a", "a"), ("t", "t"), ("v", "v"), ("at", "a"), ("av", "a"), ("tv", "t"))
        return [dict(ladder=l, feat=f, seed=s) for l, f in pairs for s in range(3)]
    if name == "drugban":
        return [dict(dataset=d, degradation=c, seed=s) for d, c in _DRUGBAN for s in range(3)]
    raise KeyError(f"unknown benchmark {name!r}; available: {sorted(LOADERS)}")

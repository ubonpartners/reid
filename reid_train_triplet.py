"""
ReID adapter training with upgrades from UTRACK_REVIEW §9.2:

  (1) P/K batch-hard triplet sampler  -- Hermans, Beyer, Leibe, arXiv 2017.
  (2) Classification loss alongside triplet.
  (4) BNNeck -- Luo et al., CVPRW 2019.
  (5) ArcFace / AM-Softmax classifier head -- Deng et al. 2019; Wang et al. 2018.
  (6) Cross-Batch Memory (XBM) -- Wang et al., CVPR 2020.
  (7) Soft-margin triplet via softplus.
  (8) Composite checkpoint selection (R@1 + d-prime).

Only the ReIDAdapter state_dict is saved; BNNeck/classifier/XBM live in this
file and are discarded at checkpoint time. The deployed tracker sees a vanilla
ReIDAdapter exactly as before.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
import math
import stuff
from tqdm import tqdm
import argparse
import src.reid_eval as reid_eval
from ultralytics.nn.modules.head import ReIDAdapter, ReIDAdapterV2
from torch.optim.lr_scheduler import LambdaLR
import warnings
import os
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")


# -------------------------------------------------------------------
# Dataset + P/K sampler (item 1)
# -------------------------------------------------------------------

class ReIDFeatureDataset(Dataset):
    """
    Plain (feature, label) dataset. The P/K sampler below produces
    identity-balanced batches; no per-sample triplet construction here.
    """
    def __init__(self, feats, labels):
        self.features = feats.astype(np.float32)
        self.labels = np.asarray(labels)
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            self.label_to_indices.setdefault(int(label), []).append(idx)
        # Only multi-sample identities can contribute to triplet / CE training.
        self.valid_labels = [lbl for lbl, idxs in self.label_to_indices.items()
                             if len(idxs) >= 2]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return torch.from_numpy(self.features[i]), int(self.labels[i])


class PKSampler(Sampler):
    """
    Samples P identities and K samples per identity per batch -> batch_size = P*K.

    Yields a flat stream of indices; DataLoader with batch_size=P*K,
    drop_last=True, shuffle=False consumes them in-order.
    """
    def __init__(self, label_to_indices, valid_labels, P=16, K=4,
                 num_batches=1000, seed=None):
        self.label_to_indices = label_to_indices
        self.valid_labels = list(valid_labels)
        self.P = P
        self.K = K
        self.num_batches = num_batches
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_batches * self.P * self.K

    def __iter__(self):
        for _ in range(self.num_batches):
            pids = self.rng.sample(self.valid_labels, self.P) \
                if len(self.valid_labels) >= self.P \
                else self.rng.choices(self.valid_labels, k=self.P)
            for pid in pids:
                idxs = self.label_to_indices[pid]
                if len(idxs) >= self.K:
                    chosen = self.rng.sample(idxs, self.K)
                else:
                    chosen = self.rng.choices(idxs, k=self.K)
                for i in chosen:
                    yield i


# -------------------------------------------------------------------
# BNNeck + classifier head (items 2, 4, 5)
# -------------------------------------------------------------------

class BNNeckClassifier(nn.Module):
    """
    BNNeck (Luo 2019): triplet loss consumes the pre-BN feature; classifier
    loss consumes BN(feature). At inference only the upstream ReIDAdapter is
    used, so this module is training-only.

    `mode` selects the classification loss:
      * "linear"     -- plain Linear -> CE. Simple, good baseline (item 2).
      * "am_softmax" -- additive cosine margin (Wang 2018). cos(theta) - m, scaled by s.
      * "arcface"    -- additive angular margin (Deng 2019). cos(theta + m), scaled by s.

    `am_softmax` / `arcface` also implement BNNeck's "bias-free BN" trick: the
    affine beta is frozen at 0 so the feature centroid is preserved.
    """
    def __init__(self, emb_dim, num_classes, mode="am_softmax",
                 scale=30.0, margin=0.35):
        super().__init__()
        self.mode = mode
        self.scale = float(scale)
        self.margin = float(margin)

        self.bn = nn.BatchNorm1d(emb_dim)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        self.bn.bias.requires_grad_(False)

        if mode == "linear":
            self.fc = nn.Linear(emb_dim, num_classes, bias=False)
            nn.init.normal_(self.fc.weight, std=0.01)
        elif mode in ("am_softmax", "arcface"):
            self.W = nn.Parameter(torch.empty(num_classes, emb_dim))
            nn.init.xavier_uniform_(self.W)
        else:
            raise ValueError(f"Unknown BNNeckClassifier mode {mode!r}")

    def forward(self, emb, labels=None):
        """
        emb: [B, D] pre-BN feature from the adapter (or an L2-normalised version of it).
        labels: [B] int64 class indices. Required for margin-loss modes during training.
        Returns logits [B, C].
        """
        feat = self.bn(emb)
        if self.mode == "linear":
            return self.fc(feat)

        feat_u = F.normalize(feat, dim=1)
        W_u = F.normalize(self.W, dim=1)
        cos = feat_u @ W_u.t()
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None or not self.training:
            return self.scale * cos

        onehot = F.one_hot(labels, num_classes=cos.size(1)).float()
        if self.mode == "am_softmax":
            logits = self.scale * (cos - self.margin * onehot)
        else:  # arcface
            theta = torch.acos(cos)
            theta_m = theta + self.margin * onehot
            logits = self.scale * torch.cos(theta_m)
        return logits


# -------------------------------------------------------------------
# Cross-Batch Memory (item 6)
# -------------------------------------------------------------------

class XBMQueue:
    """
    FIFO queue of (embedding, label) pairs from recent batches. Used to
    enlarge the negative pool for batch-hard mining without the memory
    cost of a larger actual batch.

    Embeddings are stored detached (no grad flows back through the queue);
    only the current batch's embeddings carry gradient.
    """
    def __init__(self, emb_dim, max_size=8192, device="cuda"):
        self.max_size = int(max_size)
        self.emb_dim = int(emb_dim)
        self.device = device
        self.embeddings = torch.empty(0, self.emb_dim, device=device)
        self.labels = torch.empty(0, dtype=torch.long, device=device)

    def size(self):
        return self.embeddings.size(0)

    @torch.no_grad()
    def enqueue(self, emb, labels):
        emb = emb.detach().to(self.device)
        labels = labels.detach().to(self.device)
        self.embeddings = torch.cat([self.embeddings, emb], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        if self.embeddings.size(0) > self.max_size:
            excess = self.embeddings.size(0) - self.max_size
            self.embeddings = self.embeddings[excess:]
            self.labels = self.labels[excess:]


# -------------------------------------------------------------------
# Losses (items 1, 7)
# -------------------------------------------------------------------

def _batch_hard_pair(query_emb_u, query_labels, mem_emb_u, mem_labels):
    """
    Shared cosine-distance + masking for batch-hard mining. Uses cosine rather
    than Euclidean to avoid the sqrt(0) backward-pass NaN when the adapter
    collapses identities onto the same point.

    Returns (cos_ap, cos_an, valid) where:
      cos_ap = cosine similarity to the HARDEST positive (smallest cos).
      cos_an = cosine similarity to the HARDEST negative (largest cos).
      valid  = rows that have both a positive and a negative available.

    On the unit sphere cos and Euclidean distance are monotonically related
    (d² = 2 - 2·cos), so training dynamics are equivalent to the L2 form used
    in the paper -- just numerically stabler.
    """
    if mem_emb_u is not None and mem_emb_u.numel() > 0:
        all_emb = torch.cat([query_emb_u, mem_emb_u], dim=0)
        all_lbl = torch.cat([query_labels, mem_labels], dim=0)
    else:
        all_emb, all_lbl = query_emb_u, query_labels

    Q = query_emb_u.size(0)
    cos = query_emb_u @ all_emb.t()

    same = query_labels.unsqueeze(1) == all_lbl.unsqueeze(0)
    self_mask = torch.zeros_like(same)
    self_mask[torch.arange(Q, device=same.device),
              torch.arange(Q, device=same.device)] = True
    same_no_self = same & ~self_mask

    valid = same_no_self.any(dim=1) & (~same).any(dim=1)
    cos_ap = cos.masked_fill(~same_no_self, float("inf")).min(dim=1).values
    cos_an = cos.masked_fill(same,           float("-inf")).max(dim=1).values
    return cos_ap, cos_an, valid


def batch_hard_triplet_soft(query_emb_u, query_labels,
                            mem_emb_u=None, mem_labels=None):
    """
    Batch-hard triplet with soft margin (softplus), Hermans-Beyer-Leibe 2017.

    Positives/negatives are selected from the union of batch + memory
    (XBM, item 6), excluding each query's own position.

        loss = softplus(cos_an - cos_ap).mean()

    softplus gives the same saturating shape as relu(d_ap - d_an + margin)
    with no margin hyperparameter, and contributes non-zero gradient even
    once cos_ap > cos_an.
    """
    cos_ap, cos_an, valid = _batch_hard_pair(query_emb_u, query_labels,
                                             mem_emb_u, mem_labels)
    if not valid.any():
        return torch.zeros((), device=query_emb_u.device, requires_grad=True)
    return F.softplus(cos_an[valid] - cos_ap[valid]).mean()


def batch_hard_triplet_hard(query_emb_u, query_labels,
                            mem_emb_u=None, mem_labels=None, margin=0.3):
    """Hard-margin variant kept for ablation against the soft form (in cosine space)."""
    cos_ap, cos_an, valid = _batch_hard_pair(query_emb_u, query_labels,
                                             mem_emb_u, mem_labels)
    if not valid.any():
        return torch.zeros((), device=query_emb_u.device, requires_grad=True)
    return F.relu(cos_an[valid] - cos_ap[valid] + margin).mean()


# -------------------------------------------------------------------
# Checkpoint selection (item 8)
# -------------------------------------------------------------------

def composite_score(recalls, dprime):
    """
    Checkpoint-selection metric: equal weight on R@1 (ranking) and a
    normalised d-prime (separation). d-prime is clamped into [0, 3] and
    scaled to [0, 1] -- values above 3 are saturated because the tracker
    cannot exploit them above that point.
    """
    r1 = float(recalls.get("R@1", 0.0))
    d = float(max(0.0, min(3.0, dprime))) / 3.0
    return 0.5 * r1 + 0.5 * d


# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------

def train_triplet_model(train_feats, train_labels,
                        val_feats, val_labels,
                        batch_size=128, lr=2e-2, epochs=50,
                        patience=10,
                        device="cuda", output="out.pth",
                        margin_start=0.01, margin_end=0.35,
                        hidden1=160, hidden2=192, emb=96,
                        # adapter version: 1 = legacy ReIDAdapter (fp16-unsafe),
                        # 2 = ReIDAdapterV2 (interleaved LN, SiLU, no scale, fp16-safe)
                        adapter_version=2,
                        # item 1: PK sampler
                        pk_P=64, pk_K=4, pk_num_batches=None,
                        # item 7: soft margin
                        soft_margin=True,
                        # items 2, 4, 5: classifier + BNNeck + margin head
                        ce_enabled=True,
                        ce_mode="am_softmax",     # "linear" | "am_softmax" | "arcface"
                        ce_weight=1.0,
                        ce_scale=30.0,
                        ce_margin=0.35,
                        # item 6: XBM
                        xbm_enabled=True,
                        xbm_size=8192,
                        xbm_start_epoch=3,
                        # item 8: checkpoint selection
                        checkpoint_metric="r1"):  # "r1" | "composite" | "avg_recall" | "dprime"
    """
    Train a ReIDAdapter with triplet + (optional) classification + (optional) XBM.

    Checkpoint saved contains ONLY the adapter state_dict; the BNNeck classifier
    and XBM queue are training-only scaffolding. Deployment loads the adapter
    exactly as before.
    """
    train_ds = ReIDFeatureDataset(train_feats, train_labels)
    in_dim = train_feats[0].shape[0]

    # --- label remapping: raw ID -> [0, num_classes) for the classifier ---
    uniq_labels = sorted({int(l) for l in train_ds.labels})
    label_to_cls = {lbl: i for i, lbl in enumerate(uniq_labels)}
    num_classes = len(uniq_labels)

    # --- P/K sampler (item 1) ---
    effective_batch = pk_P * pk_K
    if pk_num_batches is None:
        # Size one "epoch" so we see ~= len(train_ds) total samples per pass.
        pk_num_batches = max(1, len(train_ds) // effective_batch)
    sampler = PKSampler(train_ds.label_to_indices, train_ds.valid_labels,
                        P=pk_P, K=pk_K, num_batches=pk_num_batches)
    train_loader = DataLoader(train_ds, batch_size=effective_batch, sampler=sampler,
                              drop_last=True, num_workers=0, pin_memory=False)

    if int(adapter_version) == 2:
        adapter_cls = ReIDAdapterV2
    elif int(adapter_version) == 1:
        adapter_cls = ReIDAdapter
    else:
        raise ValueError(f"Unknown adapter_version: {adapter_version} (expected 1 or 2)")

    print(f"{adapter_cls.__name__} in_dim={in_dim}  hidden=({hidden1},{hidden2})  emb={emb}")
    print(f"train: N={len(train_ds)}  classes={num_classes}  "
          f"PK=({pk_P},{pk_K}) -> batch={effective_batch}  batches/epoch={pk_num_batches}")
    print(f"loss: triplet({'soft' if soft_margin else 'hard'}), "
          f"ce_enabled={ce_enabled} mode={ce_mode} weight={ce_weight}  "
          f"xbm_enabled={xbm_enabled} size={xbm_size} start_epoch={xbm_start_epoch}  "
          f"ckpt_metric={checkpoint_metric}")

    model = adapter_cls(in_dim=in_dim, hidden1=hidden1, hidden2=hidden2, emb=emb).to(device)

    # V1 has a learnable `scale` that can collapse and force MLP weights to grow
    # huge, overflowing fp16 at inference. Clamp it into a safe range when
    # training V1. V2 has no scale parameter (its interleaved LN bounds every
    # intermediate activation to O(1) regardless of weight magnitude), so the
    # clamp is a no-op for V2.
    SCALE_MIN, SCALE_MAX = 0.1, 20.0

    head = None
    if ce_enabled:
        head = BNNeckClassifier(emb_dim=emb, num_classes=num_classes,
                                mode=ce_mode, scale=ce_scale, margin=ce_margin).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if head is not None:
        params += [p for p in head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    warmup_epochs = min(5, max(1, epochs))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        if epochs <= warmup_epochs:
            return 1.0
        t = epoch - warmup_epochs
        T = max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * t / T))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    xbm = XBMQueue(emb_dim=emb, max_size=xbm_size, device=device) if xbm_enabled else None

    # Baseline evaluation (item 8: report d-prime alongside recall from the start).
    recalls = reid_eval.evaluate_recall_faiss(None, val_feats, val_labels,
                                              device=device, name="val", do_print=False)
    dp = reid_eval.evaluate_dprime(None, val_feats, val_labels, device=device, name="val")
    print("Eval with no REID model       " +
          " ".join(f"{r}={recalls[r]:0.4f}" for r in recalls) +
          f"  d'={dp['d_prime']:.3f}  gap={dp['pos_mean']-dp['neg_mean']:.3f}")

    best_score = -float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if head is not None:
            head.train()
        total_tri = 0.0
        total_ce = 0.0
        counted = 0
        n_nan_steps = 0

        hard_margin = margin_start + (epoch - 1) / max(1, epochs - 1) * (margin_end - margin_start)
        xbm_active = xbm is not None and epoch >= xbm_start_epoch

        for feats, raw_labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            feats = feats.to(device)
            # Remap raw IDs -> contiguous class indices.
            cls_labels = torch.tensor(
                [label_to_cls[int(l)] for l in raw_labels.tolist()],
                dtype=torch.long, device=device)

            emb_out = model(feats)                       # [B, emb] -- scale*unit
            emb_u = F.normalize(emb_out, p=2, dim=1)     # unit for triplet / XBM

            # Triplet (item 7: soft margin, item 1: batch-hard mining,
            # item 6: XBM augments the negative pool)
            mem_emb_u, mem_lbl = None, None
            if xbm_active and xbm.size() > 0:
                mem_emb_u = xbm.embeddings
                mem_lbl = xbm.labels
            if soft_margin:
                loss_tri = batch_hard_triplet_soft(emb_u, cls_labels, mem_emb_u, mem_lbl)
            else:
                loss_tri = batch_hard_triplet_hard(emb_u, cls_labels, mem_emb_u, mem_lbl,
                                                   margin=hard_margin)

            # Classification loss (items 2, 4, 5: BNNeck + AM-softmax/ArcFace)
            loss_ce = torch.zeros((), device=device)
            if head is not None:
                logits = head(emb_out, cls_labels)
                loss_ce = F.cross_entropy(logits, cls_labels)

            loss = loss_tri + (ce_weight * loss_ce if head is not None else 0.0)

            optimizer.zero_grad()
            loss.backward()

            # NaN guard: a single non-finite loss or grad poisons Adam's moment
            # buffers permanently. Skip step + enqueue + stats when that happens
            # so a transient numerical hiccup can't corrupt the run.
            bad = (not torch.isfinite(loss)) or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in params
            )
            if bad:
                n_nan_steps += 1
                continue

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Keep scale away from the fp16-overflow regime (see SCALE_MIN/MAX above).
            if hasattr(model, "scale") and isinstance(model.scale, nn.Parameter):
                with torch.no_grad():
                    model.scale.clamp_(min=SCALE_MIN, max=SCALE_MAX)

            # Enqueue current batch embeddings (detached) for next iteration.
            if xbm is not None:
                xbm.enqueue(emb_u, cls_labels)

            total_tri += float(loss_tri.item()) * feats.size(0)
            total_ce  += float(loss_ce.item())  * feats.size(0)
            counted   += feats.size(0)

        scheduler.step()
        avg_tri = total_tri / max(1, counted)
        avg_ce  = total_ce  / max(1, counted)

        # --- Evaluation ---
        recalls = reid_eval.evaluate_recall_faiss(model, val_feats, val_labels,
                                                  device=device, name="val", do_print=False)
        dp = reid_eval.evaluate_dprime(model, val_feats, val_labels,
                                       device=device, name="val")
        avg_recall = sum(recalls.values()) / max(1, len(recalls))
        score_composite = composite_score(recalls, dp["d_prime"])

        # Pick the metric that drives checkpoint / early stopping.
        if checkpoint_metric == "r1":
            score = float(recalls.get("R@1", 0.0))
        elif checkpoint_metric == "avg_recall":
            score = avg_recall
        elif checkpoint_metric == "dprime":
            score = float(dp["d_prime"])
        else:
            score = score_composite

        lr_now = optimizer.param_groups[0]['lr']
        scale_bit = ""
        if hasattr(model, "scale") and isinstance(model.scale, nn.Parameter):
            scale_bit = f" scale={float(model.scale.item()):.2f}"
        txt = (f"Epoch {epoch:02d} - L_tri={avg_tri:.4f} L_ce={avg_ce:.4f} "
               f"LR={lr_now:.2e}{scale_bit} "
               + " ".join(f"{r}={recalls[r]:0.4f}" for r in recalls) +
               f" AVG={avg_recall:0.4f} d'={dp['d_prime']:.3f} "
               f"gap={dp['pos_mean']-dp['neg_mean']:.3f} "
               f"comp={score_composite:.4f}")
        if n_nan_steps > 0:
            txt += f" [skipped {n_nan_steps} non-finite steps]"

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), output)
            txt += f" **NEW BEST** ({checkpoint_metric}={score:.4f})"
        print(txt)

        if epoch >= best_epoch + patience:
            print(f"Stopping at epoch {epoch} — no improvement in "
                  f"{checkpoint_metric} for {patience} epochs")
            break

    if os.path.exists(output):
        model.load_state_dict(torch.load(output, map_location=device))
    return model


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='reid_train_triplet.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default="/mldata/config/reid/reid_train.yaml", help="config")
    opt = parser.parse_args()

    config = stuff.load_dictionary(opt.config)

    data = np.load(config["reid_dataset"])
    arrays = data.files
    assert "train_labels" in arrays
    assert "train_feats" in arrays

    epochs = config.get("train_epochs", 20)
    batch_size = config.get("train_batch_size", 128)

    feats = data["train_feats"]
    labels = data["train_labels"]
    val_feats = data["val_feats"]
    val_labels = data["val_labels"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_triplet_model(
        feats, labels, val_feats, val_labels,
        device=device,
        epochs=epochs,
        patience=config.get("train_patience", 10),
        batch_size=batch_size,
        lr=config.get("train_lr0", 0.01),
        output=config["reid_model"],
        hidden1=config.get("hidden1", 160),
        hidden2=config.get("hidden2", 192),
        emb=config.get("emb", 96),
        adapter_version=config.get("adapter_version", 2),
        pk_P=config.get("pk_P", 16),
        pk_K=config.get("pk_K", 4),
        pk_num_batches=config.get("pk_num_batches", None),
        soft_margin=config.get("soft_margin", True),
        ce_enabled=config.get("ce_enabled", True),
        ce_mode=config.get("ce_mode", "am_softmax"),
        ce_weight=config.get("ce_weight", 1.0),
        ce_scale=config.get("ce_scale", 30.0),
        ce_margin=config.get("ce_margin", 0.35),
        xbm_enabled=config.get("xbm_enabled", True),
        xbm_size=config.get("xbm_size", 8192),
        xbm_start_epoch=config.get("xbm_start_epoch", 3),
        checkpoint_metric=config.get("checkpoint_metric", "composite"),
    )
    print("evaluate recall...")
    for v in arrays:
        if v.startswith("val") and v.endswith("_labels"):
            name = v[:-len("_labels")]
            val_labels_key = v
            val_feats_key = name + "_feats"
            reid_eval.evaluate_recall_faiss(model, data[val_feats_key], data[val_labels_key],
                                            device=device, name=name)
            reid_eval.evaluate_dprime(model, data[val_feats_key], data[val_labels_key],
                                      device=device, name=name, do_print=True)



# Epoch 92 - Triplet Loss: 0.3136 - Margin: 0.323 - LR: 5.22e-05 - Triplets used: 110304 R@1=0.4891 R@5=0.7256 R@10=0.8089 R@20=0.8800 AVG=0.7259 **NEW BEST** 0.7259
# Epoch 47 - Triplet Loss: 0.1675 - Margin: 0.168 - LR: 1.77e-03 - Triplets used: 278371 R@1=0.4133 R@5=0.6596 R@10=0.7505 R@20=0.8309 AVG=0.6635 **NEW BEST** 0.6635
# Epoch 70 - Triplet Loss: 0.2462 - Margin: 0.247 - LR: 6.80e-04 - Triplets used: 260147 R@1=0.4249 R@5=0.6750 R@10=0.7661 R@20=0.8353 AVG=0.6753 **NEW BEST** 0.6753
# Epoch 45 - Triplet Loss: 0.1612 - Margin: 0.161 - LR: 1.87e-03 - Triplets used: 268144 R@1=0.4208 R@5=0.6861 R@10=0.7714 R@20=0.8407 AVG=0.6797 **NEW BEST** 0.6797
# Epoch 67 - Triplet Loss: 0.2361 - Margin: 0.237 - LR: 8.08e-04 - Triplets used: 240164 R@1=0.4090 R@5=0.6613 R@10=0.7499 R@20=0.8263 AVG=0.6616 **NEW BEST** 0.6616 v9s-160825
# Epoch 82 - Triplet Loss: 0.2871 - Margin: 0.288 - LR: 2.58e-04 - Triplets used: 243700 R@1=0.4385 R@5=0.6818 R@10=0.7749 R@20=0.8519 AVG=0.6868 **NEW BEST** 0.6868 v8l

#Yolo26l
#Epoch 55 - Triplet Loss: 0.1947 - Margin: 0.195 - LR: 2.29e-03 - Triplets used: 196852 R@1=0.5139 R@5=0.7309 R@10=0.8043 R@20=0.8669 AVG=0.7290 **NEW BEST** 0.7290
#Epoch 86 - Triplet Loss: 0.2997 - Margin: 0.302 - LR: 5.26e-04 - Triplets used: 138285 R@1=0.5863 R@5=0.7906 R@10=0.8530 R@20=0.9027 AVG=0.7832 **NEW BEST** 0.7832

# New training
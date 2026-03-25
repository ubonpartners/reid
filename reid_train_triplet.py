import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import stuff
from tqdm import tqdm
import argparse
import src.reid_eval as reid_eval
from ultralytics.nn.modules.head import ReIDAdapter
from torch.optim.lr_scheduler import LambdaLR
import warnings
import os
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

class TripletReIDDataset(Dataset):
    def __init__(self, feats, labels):
        self.features = feats.astype(np.float32)
        self.labels = labels
        self.label_to_indices = self._build_label_index()

        # Filter out indices of singleton classes
        self.valid_indices = [
            idx for idx, label in enumerate(self.labels)
            if len(self.label_to_indices[label]) > 1
        ]

    def _build_label_index(self):
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label_to_indices.setdefault(label, []).append(idx)
        return label_to_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, i):
        idx = self.valid_indices[i]
        anchor_feat = self.features[idx]  # np.ndarray
        anchor_label = self.labels[idx]

        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.label_to_indices[anchor_label])
        pos_feat = self.features[pos_idx]

        neg_label = random.choice([l for l in self.label_to_indices if l != anchor_label])
        neg_feat = self.features[random.choice(self.label_to_indices[neg_label])]

        return (
            torch.from_numpy(anchor_feat),
            torch.from_numpy(pos_feat),
            torch.from_numpy(neg_feat),
        )

def train_triplet_model(train_feats, train_labels,
                        val_feats, val_labels,
                        batch_size=128, lr=1e-2, epochs=50,
                        patience=10,
                        device="cuda", output="out.pth",
                        margin_start=0.01, margin_end=0.35):

    train_ds = TripletReIDDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    in_dim=train_feats[0].shape[0]
    print(f"Training in embedding feature size is {in_dim} - should be 520+nc")
    model = ReIDAdapter(in_dim=in_dim, hidden1=160, hidden2=192, emb=80).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Chainable warmup+cosine scheduler without SequentialLR to avoid
    # torch's epoch-parameter deprecation warning in newer versions.
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

    recalls = reid_eval.evaluate_recall_faiss(None, val_feats, val_labels, device=device, name="val", do_print=False)
    txt = "Eval with no REID model                       "
    for r in recalls:
        txt += f" {r}={recalls[r]:0.4f}"
    print(txt)
    best_avg_recall=-1.0
    best_epoch=0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        valid_triplets = 0

        # Annealed margin for this epoch
        margin = margin_start + (epoch - 1) / max(1, epochs - 1) * (margin_end - margin_start)

        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Triplet concatenation
            triplets = torch.cat([anchor, positive, negative], dim=0)
            embeddings = model(triplets)  # L2-normalized inside model
            out_a, out_p, out_n = embeddings.chunk(3, dim=0)

            # Compute distances
            d_ap = F.pairwise_distance(out_a, out_p, p=2)
            d_an = F.pairwise_distance(out_a, out_n, p=2)

            # Filter hard triplets: negatives that are too close
            mask = (d_an < d_ap + margin).float()
            if mask.sum() == 0:
                continue

            loss = (F.relu(d_ap - d_an + margin) * mask).sum() / mask.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * anchor.size(0)
            valid_triplets += mask.sum().item()

        avg_loss = total_loss / len(train_loader.dataset)
        scheduler.step()

        recalls = reid_eval.evaluate_recall_faiss(model, val_feats, val_labels, device=device, name="val", do_print=False)
        txt = f"Epoch {epoch:02d} - Triplet Loss: {avg_loss:.4f} - Margin: {margin:.3f} - LR: {optimizer.param_groups[0]['lr']:.2e} - Triplets used: {int(valid_triplets)}"
        tot = 0
        for r in recalls:
            txt += f" {r}={recalls[r]:0.4f}"
            tot+=recalls[r]
        avg_recall=tot/len(recalls)
        txt += f" AVG={avg_recall:0.4f}"
        if avg_recall>best_avg_recall:
            best_avg_recall=avg_recall
            txt+= f" **NEW BEST** {best_avg_recall:0.4f}"
            best_epoch=epoch
            torch.save(model.state_dict(), output)
        print(txt)
        if epoch>best_epoch+patience:
            print(f"Stopping training at epoch {epoch} as no improvement in last {patience} epochs")
            break

    # Load best checkpoint (if saved); otherwise keep last weights.
    if os.path.exists(output):
        model.load_state_dict(torch.load(output, map_location=device))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='reid_train_triplet.py')
    parser.add_argument('--logging', type=str, default='info', help="Logging config: level[:console|file]")
    parser.add_argument('--config', type=str, default="/mldata/config/reid/reid_train.yaml", help="config")
    opt = parser.parse_args()

    config=stuff.load_dictionary(opt.config)

    data = np.load(config["reid_dataset"])

    arrays=data.files
    assert "train_labels" in arrays
    assert "train_feats" in arrays

    epochs=config.get("train_epochs", 20)
    batch_size=config.get("train_batch_size", 128)

    feats=data["train_feats"]
    labels=data["train_labels"]
    val_feats=data["val_feats"]
    val_labels=data["val_labels"]

    #for v in arrays:
    #    print(f"data {v} {len(data[v])}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_triplet_model(feats,
                                labels,
                                val_feats,
                                val_labels,
                                device=device,
                                epochs=epochs,
                                patience=config.get("train_patience", 10),
                                batch_size=batch_size,
                                lr=config.get("train_lr0", 0.01),
                                output=config["reid_model"])
    print("evaluate recall...")
    for v in arrays:
        if v.startswith("val") and v.endswith("_labels"):
            name=v[:-len("_labels")]
            val_labels=v
            val_feats=name+"_feats"
            reid_eval.evaluate_recall_faiss(model, data[val_feats], data[val_labels], device=device, name=name)

# Epoch 92 - Triplet Loss: 0.3136 - Margin: 0.323 - LR: 5.22e-05 - Triplets used: 110304 R@1=0.4891 R@5=0.7256 R@10=0.8089 R@20=0.8800 AVG=0.7259 **NEW BEST** 0.7259
# Epoch 47 - Triplet Loss: 0.1675 - Margin: 0.168 - LR: 1.77e-03 - Triplets used: 278371 R@1=0.4133 R@5=0.6596 R@10=0.7505 R@20=0.8309 AVG=0.6635 **NEW BEST** 0.6635
# Epoch 70 - Triplet Loss: 0.2462 - Margin: 0.247 - LR: 6.80e-04 - Triplets used: 260147 R@1=0.4249 R@5=0.6750 R@10=0.7661 R@20=0.8353 AVG=0.6753 **NEW BEST** 0.6753
# Epoch 45 - Triplet Loss: 0.1612 - Margin: 0.161 - LR: 1.87e-03 - Triplets used: 268144 R@1=0.4208 R@5=0.6861 R@10=0.7714 R@20=0.8407 AVG=0.6797 **NEW BEST** 0.6797
# Epoch 67 - Triplet Loss: 0.2361 - Margin: 0.237 - LR: 8.08e-04 - Triplets used: 240164 R@1=0.4090 R@5=0.6613 R@10=0.7499 R@20=0.8263 AVG=0.6616 **NEW BEST** 0.6616 v9s-160825
# Epoch 82 - Triplet Loss: 0.2871 - Margin: 0.288 - LR: 2.58e-04 - Triplets used: 243700 R@1=0.4385 R@5=0.6818 R@10=0.7749 R@20=0.8519 AVG=0.6868 **NEW BEST** 0.6868 v8l

#Yolo26l
#Epoch 55 - Triplet Loss: 0.1947 - Margin: 0.195 - LR: 2.29e-03 - Triplets used: 196852 R@1=0.5139 R@5=0.7309 R@10=0.8043 R@20=0.8669 AVG=0.7290 **NEW BEST** 0.7290

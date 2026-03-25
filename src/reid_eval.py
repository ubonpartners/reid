import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import faiss

def evaluate_recall_at_k(model, features, labels, device="cuda", ks=(1, 5, 10), name="??", do_print=True):
    model.eval()
    features = features.astype(np.float32)
    feats_tensor = torch.tensor(features).to(device)

    if model is None:
        embeddings=feats_tensor
    else:
        with torch.no_grad():
            embeddings = model(feats_tensor).cpu().numpy()

    dist_matrix = pairwise_distances(embeddings, embeddings, metric="cosine")
    np.fill_diagonal(dist_matrix, np.inf)

    recalls = {}
    for k in ks:
        correct = 0
        for i in range(len(labels)):
            nearest = np.argsort(dist_matrix[i])[:k]
            if any(labels[nearest[j]] == labels[i] for j in range(k)):
                correct += 1
        recalls[f"R@{k}"] = correct / len(labels)

    if do_print:
        print(f"🔍 {name} : Normal Recall@K:")
        for k in ks:
            print(f"  Recall@{k}: {recalls[f'R@{k}']:.4f}")
    return recalls

def evaluate_recall_faiss(model, features, labels, device="cuda", ks=(1, 5, 10, 20), name="??", do_print=True):

    feats_tensor = torch.tensor(features.astype(np.float32)).to(device)

    if model is None:
        embeddings=feats_tensor.cpu()
    else:
        model.eval()
        with torch.no_grad():
            embeddings = model(feats_tensor).cpu().numpy()

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product == cosine sim if normalized
    index.add(embeddings)

    D, I = index.search(embeddings, max(ks)+1)  # Include self
    recalls = {}
    for k in ks:
        correct = 0
        for i in range(len(labels)):
            top_k = [j for j in I[i][1:k+1]]  # skip self (i)
            if any(labels[j] == labels[i] for j in top_k):
                correct += 1
        recalls[f"R@{k}"] = correct / len(labels)

    if do_print:
        print(f"🔍 {name} : Faiss Recall@K [{len(labels)} labels]:")
        ksum=0
        for k in ks:
            print(f"  Recall@{k}: {recalls[f'R@{k}']:.4f}")
            ksum+=recalls[f'R@{k}']
        avg=ksum/len(ks)
        print(f"  Avg {avg:.4f}")

    return recalls


def evaluate_standard_reid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    query_camids: np.ndarray | None = None,
    gallery_camids: np.ndarray | None = None,
    ks=(1, 5, 10, 20),
) -> dict[str, float]:
    """
    Evaluate a standard ReID protocol over explicit query/gallery splits.

    Metrics:
    - CMC (Rank-K)
    - mAP

    If camids are provided, same pid + same cam matches are treated as junk and
    excluded from ranking for each query (common in person-ReID benchmarks).
    """
    if len(query_indices) == 0 or len(gallery_indices) == 0:
        out = {f"Rank-{k}": 0.0 for k in ks}
        out["mAP"] = 0.0
        out["num_query_total"] = 0.0
        out["num_query_valid"] = 0.0
        return out

    emb = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    gallery_indices = np.asarray(gallery_indices, dtype=np.int64)

    # Cosine search by L2-normalizing then using inner-product.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-12)

    query_emb = emb[query_indices]
    gallery_emb = emb[gallery_indices]
    gallery_labels = labels[gallery_indices]

    index = faiss.IndexFlatIP(gallery_emb.shape[1])
    index.add(gallery_emb)
    _, rank_idx = index.search(query_emb, len(gallery_indices))

    cmc_hits = np.zeros(len(ks), dtype=np.float64)
    ap_sum = 0.0
    valid_queries = 0

    use_cam_filter = query_camids is not None and gallery_camids is not None

    for qi, q_global_idx in enumerate(query_indices):
        q_label = labels[q_global_idx]
        ranked_gallery_global = gallery_indices[rank_idx[qi]]
        ranked_gallery_labels = labels[ranked_gallery_global]

        valid_mask = np.ones_like(ranked_gallery_global, dtype=bool)
        if use_cam_filter:
            q_cam = query_camids[qi]
            rg_cam = gallery_camids[rank_idx[qi]]
            junk = (ranked_gallery_labels == q_label) & (rg_cam == q_cam)
            valid_mask &= ~junk

        ranked_gallery_global = ranked_gallery_global[valid_mask]
        ranked_gallery_labels = ranked_gallery_labels[valid_mask]

        positive_mask = ranked_gallery_labels == q_label
        num_pos = int(np.sum(positive_mask))
        if num_pos == 0:
            continue

        valid_queries += 1
        pos_ranks = np.flatnonzero(positive_mask)

        # CMC
        for k_i, k in enumerate(ks):
            if np.any(pos_ranks < k):
                cmc_hits[k_i] += 1.0

        # AP
        hit_count = 0
        precisions = []
        for r, is_pos in enumerate(positive_mask, start=1):
            if is_pos:
                hit_count += 1
                precisions.append(hit_count / r)
        ap_sum += float(np.mean(precisions)) if precisions else 0.0

    out = {f"Rank-{k}": 0.0 for k in ks}
    out["mAP"] = 0.0
    out["num_query_total"] = float(len(query_indices))
    out["num_query_valid"] = float(valid_queries)
    if valid_queries > 0:
        for k_i, k in enumerate(ks):
            out[f"Rank-{k}"] = float(cmc_hits[k_i] / valid_queries)
        out["mAP"] = float(ap_sum / valid_queries)
    return out
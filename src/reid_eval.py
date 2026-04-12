import torch
import numpy as np
import faiss


def evaluate_recall_faiss(model, features, labels, device="cuda", ks=(1, 5, 10, 20), name="??", do_print=True):
    """
    Compute Recall@K using FAISS inner-product search on L2-normalised embeddings
    (equivalent to cosine similarity).

    model: ReIDAdapter or None. If None, raw features are evaluated directly.
    features: numpy array [N, D]
    labels: array-like [N] — integer identity labels
    """
    feats_tensor = torch.tensor(features.astype(np.float32)).to(device)

    if model is None:
        embeddings = feats_tensor.cpu().numpy()
    else:
        model.eval()
        with torch.no_grad():
            embeddings = model(feats_tensor).cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    # IndexFlatIP == cosine similarity when vectors are L2-normalised.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Retrieve k+1 neighbours; the first result is always the query itself.
    D, I = index.search(embeddings, max(ks) + 1)

    recalls = {}
    for k in ks:
        correct = 0
        for i in range(len(labels)):
            top_k = I[i][1:k+1]  # skip self at position 0
            if any(labels[j] == labels[i] for j in top_k):
                correct += 1
        recalls[f"R@{k}"] = correct / len(labels)

    if do_print:
        print(f"🔍 {name} : Faiss Recall@K [{len(labels)} labels]:")
        ksum = 0
        for k in ks:
            print(f"  Recall@{k}: {recalls[f'R@{k}']:.4f}")
            ksum += recalls[f'R@{k}']
        print(f"  Avg {ksum / len(ks):.4f}")

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

    Returns CMC (Rank-K) and mAP.

    If camids are provided, same-identity same-camera gallery entries are treated as
    junk and excluded from ranking (standard person-ReID benchmark convention).
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

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-12)

    query_emb = emb[query_indices]
    gallery_emb = emb[gallery_indices]

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

        if use_cam_filter:
            q_cam = query_camids[qi]
            rg_cam = gallery_camids[rank_idx[qi]]
            # Exclude same-identity same-camera entries (junk).
            valid_mask = ~((ranked_gallery_labels == q_label) & (rg_cam == q_cam))
            ranked_gallery_global = ranked_gallery_global[valid_mask]
            ranked_gallery_labels = ranked_gallery_labels[valid_mask]

        positive_mask = ranked_gallery_labels == q_label
        if not np.any(positive_mask):
            continue  # query has no positive in gallery — skip

        valid_queries += 1
        pos_ranks = np.flatnonzero(positive_mask)

        for k_i, k in enumerate(ks):
            if np.any(pos_ranks < k):
                cmc_hits[k_i] += 1.0

        # Average precision over ranked positive hits.
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

from typing import List, Tuple
import math


def calc_precision_at_k(
    top_k: List[Tuple[float, float]], k: int, threshold: float
) -> float:
    """
    Calculates Precision@K given the top K items.
    """

    if not top_k:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / k


def calc_recall_at_k(
    top_k: List[Tuple[float, float]], n_rel_total: int, threshold: float
) -> float:
    """
    Calculates Recall@K given top K items and total relevant count.
    """

    if n_rel_total == 0:
        return 0.0

    n_rel_k = sum(true_r >= threshold for (_, true_r) in top_k)
    return n_rel_k / n_rel_total


def calc_hit_rate_at_k(top_k: List[Tuple[float, float]], threshold: float) -> float:
    """
    Calculates HitRate@K (1.0 if any relevant item is in top K, else 0.0).
    """

    is_hit = any(true_r >= threshold for (_, true_r) in top_k)
    return 1.0 if is_hit else 0.0


def calc_ndcg_at_k(
    top_k: List[Tuple[float, float]], n_rel_total: int, k: int, threshold: float
) -> float:
    """
    Calculates NDCG@K using binary relevance.
    """

    dcg = 0.0

    # Calculate DCG (based on predicted rank in top_k)
    for i, (_, true_r) in enumerate(top_k):
        if true_r >= threshold:
            dcg += 1.0 / math.log2(i + 2)

    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    num_ideal_relevant = min(n_rel_total, k)

    for i in range(num_ideal_relevant):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0

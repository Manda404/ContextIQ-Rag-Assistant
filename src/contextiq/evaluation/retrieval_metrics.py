"""Métriques de classement pour une question ayant des sources attendues."""

from pydantic import BaseModel, ConfigDict, Field

from contextiq.logging import log_operation


class RetrievalScores(BaseModel):
    """Scores entre 0 et 1, calculés pour une valeur de K donnée."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)


@log_operation
def compute_retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: list[str],
    top_k: int,
) -> RetrievalScores:
    """Calculer Precision@K, Recall@K, Hit Rate@K et MRR@K.

    Precision@K utilise K au dénominateur, même avec moins de K résultats.
    Une question sans source attendue doit être analysée séparément.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k doit être un entier strictement positif.")
    if not gold_ids:
        raise ValueError("Les questions hors périmètre sont évaluées séparément.")
    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError("Un document ne doit pas être retourné plusieurs fois.")
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Les sources attendues doivent être uniques.")

    gold = set(gold_ids)
    relevant_ranks = [
        rank
        for rank, document_id in enumerate(retrieved_ids[:top_k], start=1)
        if document_id in gold
    ]
    return RetrievalScores(
        precision=len(relevant_ranks) / top_k,
        recall=len(relevant_ranks) / len(gold),
        hit_rate=float(bool(relevant_ranks)),
        mrr=1 / relevant_ranks[0] if relevant_ranks else 0.0,
    )

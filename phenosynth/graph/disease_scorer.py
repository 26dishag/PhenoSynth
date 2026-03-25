"""
Disease Scorer.

Given a list of HPO term IDs extracted from a patient description,
scores every disease in the annotation database by how well its known
symptom profile overlaps with the patient's symptoms.

Scoring approach: inverse document frequency (IDF) weighting.

    A symptom that appears in only 10 diseases is very specific.
    Matching on it is strong evidence.

    A symptom that appears in 3000 diseases tells you almost nothing.
    Matching on it barely moves the score.

    weight(term) = log(total_diseases / diseases_with_this_term)

This is the same core idea used in search engines to rank documents.
Here, diseases are the documents and HPO terms are the keywords.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DiseaseMatch:
    disease_id: str
    disease_name: str
    score: float                        # Raw weighted overlap score
    match_percent: float                # Score normalized to 0-100
    matched_terms: list[str]            # HPO IDs that matched
    total_disease_terms: int            # How many HPO terms this disease has in total
    coverage: float                     # Fraction of patient terms that matched


@dataclass
class ScoringResult:
    query_hpo_ids: list[str]
    matches: list[DiseaseMatch] = field(default_factory=list)

    @property
    def top(self) -> list[DiseaseMatch]:
        return self.matches[:5]


class DiseaseScorer:
    """
    Scores diseases against a patient HPO term set using IDF-weighted overlap.

    Usage:
        from phenosynth.graph.disease_loader import load_disease_index
        from phenosynth.graph.disease_scorer import DiseaseScorer

        index = load_disease_index()
        scorer = DiseaseScorer(index)

        result = scorer.score(["HP:0003701", "HP:0001638", "HP:0003236"])
        for match in result.top:
            print(match.match_percent, match.disease_name)
    """

    def __init__(self, disease_index: dict):
        self._disease_to_hpo: dict[str, dict] = disease_index["disease_to_hpo"]
        self._hpo_to_diseases: dict[str, list[str]] = disease_index["hpo_to_diseases"]
        self._total_diseases = len(self._disease_to_hpo)

        # Pre-compute IDF weight for every HPO term in the database
        self._idf: dict[str, float] = {}
        for hpo_id, disease_list in self._hpo_to_diseases.items():
            count = len(disease_list)
            if count > 0:
                self._idf[hpo_id] = math.log(self._total_diseases / count)

        logger.info(
            "DiseaseScorer ready: %d diseases, %d IDF weights computed.",
            self._total_diseases,
            len(self._idf),
        )

    def _term_weight(self, hpo_id: str) -> float:
        """
        Return the IDF weight for an HPO term.
        Higher weight = rarer term = more informative match.
        Terms not found in the database get a moderate default weight.
        """
        return self._idf.get(hpo_id, math.log(self._total_diseases / 5))

    def score(self, hpo_ids: list[str], top_n: int = 10) -> ScoringResult:
        """
        Score all diseases against the patient's HPO term set.

        Args:
            hpo_ids:  List of HPO term IDs from the symptom extractor.
            top_n:    Number of top-scoring diseases to return.

        Returns:
            ScoringResult with ranked DiseaseMatch list.
        """
        if not hpo_ids:
            return ScoringResult(query_hpo_ids=hpo_ids)

        query_set = set(hpo_ids)

        # Compute total possible score for normalization:
        # the theoretical max score is the sum of all term weights in the query
        max_possible = sum(self._term_weight(h) for h in hpo_ids)
        if max_possible == 0:
            return ScoringResult(query_hpo_ids=hpo_ids)

        scores: list[DiseaseMatch] = []

        for disease_id, info in self._disease_to_hpo.items():
            disease_terms = set(info["hpo_terms"])
            matched = query_set & disease_terms

            if not matched:
                continue

            # Weighted score: sum of IDF weights for each matching term
            raw_score = sum(self._term_weight(h) for h in matched)

            # Normalize to 0-100 against the theoretical max for this query
            match_percent = round((raw_score / max_possible) * 100, 1)

            # Coverage: how many of the patient's terms were matched
            coverage = round(len(matched) / len(query_set), 3)

            scores.append(DiseaseMatch(
                disease_id=disease_id,
                disease_name=info["name"],
                score=round(raw_score, 6),
                match_percent=match_percent,
                matched_terms=sorted(matched),
                total_disease_terms=len(disease_terms),
                coverage=coverage,
            ))

        # Sort by match_percent descending, then by coverage as a tiebreaker
        scores.sort(key=lambda m: (m.match_percent, m.coverage), reverse=True)

        return ScoringResult(
            query_hpo_ids=hpo_ids,
            matches=scores[:top_n],
        )

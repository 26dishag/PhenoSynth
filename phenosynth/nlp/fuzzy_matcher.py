"""
HPO Fuzzy Matcher.

Takes a medical entity string (extracted by SciSpaCy) and finds the
best-matching HPO term in the ontology index using string similarity scoring.

Why fuzzy matching instead of exact lookup:
    SciSpaCy might extract "proximal limb weakness" from a clinical note.
    The HPO index contains "proximal muscle weakness" (HP:0003325).
    These are not identical strings, but they are close enough to match
    with high confidence. Fuzzy matching handles this gracefully.

The scorer used is token_sort_ratio from rapidfuzz, which:
    1. Splits both strings into tokens (words)
    2. Sorts the tokens alphabetically
    3. Compares the sorted versions

This means word order does not matter:
    "weakness muscle" scores the same as "muscle weakness"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# Minimum similarity score (0-100) to accept a match.
# Below this threshold the match is considered too weak and discarded.
DEFAULT_THRESHOLD = 72


@dataclass
class FuzzyMatch:
    hpo_id: str
    label: str
    query: str              # The phrase we searched for
    matched_term: str       # The HPO term string that was closest
    score: float            # Similarity score from 0 to 100
    confidence: float       # Normalized score from 0.0 to 1.0


class HPOFuzzyMatcher:
    """
    Matches free-text medical phrases to HPO terms using fuzzy string similarity.

    Usage:
        from phenosynth.graph.hpo_loader import load_hpo_index
        from phenosynth.nlp.fuzzy_matcher import HPOFuzzyMatcher

        index = load_hpo_index()
        matcher = HPOFuzzyMatcher(index)

        result = matcher.match("proximal limb weakness")
        if result:
            print(result.hpo_id, result.label, result.confidence)
            # HP:0003325  Proximal muscle weakness  0.94
    """

    def __init__(
        self,
        hpo_index: dict[str, tuple[str, str]],
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Args:
            hpo_index: Output of load_hpo_index(). Maps term text to (HPO ID, label).
            threshold: Minimum similarity score (0-100) to accept a match.
        """
        self._index = hpo_index
        self._keys = list(hpo_index.keys())
        self._threshold = threshold
        logger.info("HPOFuzzyMatcher initialized with %d index entries.", len(self._keys))

    def match(self, query: str) -> Optional[FuzzyMatch]:
        """
        Find the closest HPO term to a query phrase.

        Args:
            query: A medical phrase, e.g. "proximal limb weakness".

        Returns:
            FuzzyMatch if a term is found above threshold, else None.
        """
        if not query or not query.strip():
            return None

        normalized_query = query.strip().lower()

        result = process.extractOne(
            normalized_query,
            self._keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self._threshold,
        )

        if result is None:
            return None

        matched_term, score, _ = result
        hpo_id, label = self._index[matched_term]

        return FuzzyMatch(
            hpo_id=hpo_id,
            label=label,
            query=query,
            matched_term=matched_term,
            score=score,
            confidence=round(score / 100.0, 3),
        )

    def match_many(self, queries: list[str]) -> list[FuzzyMatch]:
        """
        Match a list of phrases, deduplicate by HPO ID, and sort by confidence.

        Args:
            queries: List of medical phrases from SciSpaCy NER.

        Returns:
            Deduplicated list of FuzzyMatch results, highest confidence first.
        """
        seen_ids: set[str] = set()
        results: list[FuzzyMatch] = []

        for query in queries:
            match = self.match(query)
            if match and match.hpo_id not in seen_ids:
                results.append(match)
                seen_ids.add(match.hpo_id)

        return sorted(results, key=lambda m: m.confidence, reverse=True)

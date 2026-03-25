"""
SymptomExtractor: maps free-text clinical descriptions to HPO term IDs.

Two modes:

  Phase 1 (seed-only, no dependencies):
    Uses a small built-in dictionary of ~30 common symptom phrases.
    Fast to start, no downloads needed. Good for testing.

  Phase 2 (full NLP, default):
    Uses SciSpaCy (a biomedical NLP model trained on millions of medical
    papers) to extract medical entity spans from raw text, then fuzzy-matches
    those spans against the full HPO ontology (17,000+ terms and synonyms).
    Understands paraphrases, word order variation, and partial matches.
    Requires the HPO index to be downloaded on first use (~10 MB, one-time).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 seed dictionary (fallback / fast mode)
# ---------------------------------------------------------------------------

HPO_SEED: dict[str, tuple[str, str]] = {
    "muscle weakness": ("HP:0003326", "Myalgia"),
    "myalgia": ("HP:0003326", "Myalgia"),
    "elevated creatine kinase": ("HP:0003236", "Elevated serum creatine kinase"),
    "high creatine kinase": ("HP:0003236", "Elevated serum creatine kinase"),
    "cardiomyopathy": ("HP:0001638", "Cardiomyopathy"),
    "heart muscle disease": ("HP:0001638", "Cardiomyopathy"),
    "seizures": ("HP:0001250", "Seizure"),
    "seizure": ("HP:0001250", "Seizure"),
    "intellectual disability": ("HP:0001249", "Intellectual disability"),
    "cognitive impairment": ("HP:0100543", "Cognitive impairment"),
    "short stature": ("HP:0004322", "Short stature"),
    "growth retardation": ("HP:0001510", "Growth delay"),
    "vision loss": ("HP:0000572", "Visual loss"),
    "hearing loss": ("HP:0000365", "Hearing impairment"),
    "ataxia": ("HP:0001251", "Ataxia"),
    "spasticity": ("HP:0001257", "Spasticity"),
    "hypotonia": ("HP:0001252", "Muscular hypotonia"),
    "low muscle tone": ("HP:0001252", "Muscular hypotonia"),
    "hepatomegaly": ("HP:0002240", "Hepatomegaly"),
    "enlarged liver": ("HP:0002240", "Hepatomegaly"),
    "splenomegaly": ("HP:0001744", "Splenomegaly"),
    "enlarged spleen": ("HP:0001744", "Splenomegaly"),
    "fatigue": ("HP:0012378", "Fatigue"),
    "jaundice": ("HP:0000952", "Jaundice"),
    "yellow skin": ("HP:0000952", "Jaundice"),
    "joint pain": ("HP:0002829", "Arthralgia"),
    "arthralgia": ("HP:0002829", "Arthralgia"),
    "scoliosis": ("HP:0002650", "Scoliosis"),
    "nystagmus": ("HP:0000639", "Nystagmus"),
    "dystonia": ("HP:0001332", "Dystonia"),
    "tremor": ("HP:0001337", "Tremor"),
    "ptosis": ("HP:0000508", "Ptosis"),
    "drooping eyelid": ("HP:0000508", "Ptosis"),
    "proteinuria": ("HP:0000093", "Proteinuria"),
}


# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedPhenotype:
    hpo_id: str
    label: str
    matched_phrase: str
    confidence: float = 1.0
    source: str = "seed"   # "seed" or "scispacy"


@dataclass
class ExtractionResult:
    input_text: str
    phenotypes: list[ExtractedPhenotype] = field(default_factory=list)
    mode: str = "seed"

    @property
    def hpo_ids(self) -> list[str]:
        return [p.hpo_id for p in self.phenotypes]

    def __repr__(self) -> str:
        lines = [f"[{self.mode}] Input: {self.input_text[:80]}"]
        for p in self.phenotypes:
            lines.append(
                f"  {p.hpo_id}  {p.label}  "
                f"(phrase: '{p.matched_phrase}', conf: {p.confidence:.2f}, src: {p.source})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: seed-only extractor (no ML dependencies)
# ---------------------------------------------------------------------------

class SeedExtractor:
    """
    Fast dictionary-based extractor. Used as fallback when SciSpaCy is
    unavailable, and as a supplementary pass in Phase 2.
    """

    def __init__(self, custom_terms: Optional[dict[str, tuple[str, str]]] = None):
        self._terms = {**HPO_SEED, **(custom_terms or {})}
        self._sorted_terms = sorted(self._terms.keys(), key=len, reverse=True)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract(self, text: str) -> list[ExtractedPhenotype]:
        normalized = self._normalize(text)
        seen_ids: set[str] = set()
        results: list[ExtractedPhenotype] = []

        for phrase in self._sorted_terms:
            if phrase in normalized:
                hpo_id, label = self._terms[phrase]
                if hpo_id not in seen_ids:
                    results.append(ExtractedPhenotype(
                        hpo_id=hpo_id,
                        label=label,
                        matched_phrase=phrase,
                        confidence=1.0,
                        source="seed",
                    ))
                    seen_ids.add(hpo_id)

        return results


# ---------------------------------------------------------------------------
# Phase 2: SciSpaCy + HPO fuzzy matching extractor
# ---------------------------------------------------------------------------

class ScispaCyExtractor:
    """
    Full NLP extractor using SciSpaCy for biomedical NER and HPO fuzzy matching.

    On first use this downloads the HPO ontology file (~10 MB) and builds
    the term index. Subsequent runs load from disk in seconds.
    """

    def __init__(self, threshold: float = 72.0):
        """
        Args:
            threshold: Minimum fuzzy match score (0-100) to accept a match.
                       Lower values return more results but more false positives.
                       72 is a good balance for clinical text.
        """
        self._threshold = threshold
        self._nlp = None
        self._matcher = None

    def _lazy_load(self) -> None:
        """Load SciSpaCy model and HPO index on first use (not at import time)."""
        if self._nlp is not None:
            return

        import spacy
        from phenosynth.graph.hpo_loader import load_hpo_index
        from phenosynth.nlp.fuzzy_matcher import HPOFuzzyMatcher

        logger.info("Loading SciSpaCy model en_core_sci_sm...")
        self._nlp = spacy.load("en_core_sci_sm")

        logger.info("Loading HPO index...")
        hpo_index = load_hpo_index()

        self._matcher = HPOFuzzyMatcher(hpo_index, threshold=self._threshold)
        logger.info("ScispaCyExtractor ready.")

    def extract(self, text: str) -> list[ExtractedPhenotype]:
        """Extract HPO phenotypes from text using SciSpaCy NER + fuzzy matching."""
        self._lazy_load()

        # Words that SciSpaCy sometimes tags as entities but are not symptoms
        STOPWORDS = {
            "patient", "patients", "history", "examination", "findings",
            "presentation", "onset", "diagnosis", "treatment", "family",
            "lab", "labs", "test", "tests", "result", "results", "note",
            "complaint", "complaints",
        }

        doc = self._nlp(text)
        entity_texts = [
            ent.text for ent in doc.ents
            if ent.text.lower().strip() not in STOPWORDS and len(ent.text.strip()) > 3
        ]

        fuzzy_matches = self._matcher.match_many(entity_texts)

        return [
            ExtractedPhenotype(
                hpo_id=m.hpo_id,
                label=m.label,
                matched_phrase=m.query,
                confidence=m.confidence,
                source="scispacy",
            )
            for m in fuzzy_matches
        ]


# ---------------------------------------------------------------------------
# Unified SymptomExtractor (public API)
# ---------------------------------------------------------------------------

class SymptomExtractor:
    """
    Main entry point for symptom extraction.

    Runs SciSpaCy (Phase 2) by default. Falls back to seed dictionary
    if SciSpaCy is not available. In Phase 2 mode, seed matches are
    merged in to catch any terms the fuzzy matcher missed.

    Usage:
        extractor = SymptomExtractor()
        result = extractor.extract("Patient has muscle weakness and seizures.")
        print(result.hpo_ids)
    """

    def __init__(
        self,
        use_scispacy: bool = True,
        threshold: float = 72.0,
        custom_terms: Optional[dict[str, tuple[str, str]]] = None,
    ):
        self._seed = SeedExtractor(custom_terms=custom_terms)
        self._scispacy: Optional[ScispaCyExtractor] = None
        self._use_scispacy = use_scispacy

        if use_scispacy:
            try:
                self._scispacy = ScispaCyExtractor(threshold=threshold)
            except Exception as exc:
                logger.warning(
                    "SciSpaCy not available (%s). Falling back to seed extractor.", exc
                )
                self._use_scispacy = False

    def extract(self, text: str) -> ExtractionResult:
        """Extract HPO phenotypes from a clinical text description."""
        if not text or not text.strip():
            return ExtractionResult(input_text=text, mode="seed")

        if self._use_scispacy and self._scispacy is not None:
            return self._extract_scispacy(text)
        else:
            return self._extract_seed(text)

    def _extract_seed(self, text: str) -> ExtractionResult:
        phenotypes = self._seed.extract(text)
        return ExtractionResult(input_text=text, phenotypes=phenotypes, mode="seed")

    def _extract_scispacy(self, text: str) -> ExtractionResult:
        scispacy_results = self._scispacy.extract(text)
        seed_results = self._seed.extract(text)

        # Merge: start with SciSpaCy results, fill in any seed matches not already found
        seen_ids = {p.hpo_id for p in scispacy_results}
        merged = list(scispacy_results)

        for p in seed_results:
            if p.hpo_id not in seen_ids:
                merged.append(p)
                seen_ids.add(p.hpo_id)

        return ExtractionResult(input_text=text, phenotypes=merged, mode="scispacy")

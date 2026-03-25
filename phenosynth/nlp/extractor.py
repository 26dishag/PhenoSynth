"""
SymptomExtractor: maps free-text clinical descriptions to HPO term IDs.

Uses a rule-based + dictionary lookup approach in Phase 1.
Will be upgraded to scispaCy NER in Phase 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Minimal seed dictionary: symptom phrase → HPO ID + label
# This grows significantly in Phase 2 when we load the full HPO OBO file.
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
    "curved spine": ("HP:0002650", "Scoliosis"),
    "nystagmus": ("HP:0000639", "Nystagmus"),
    "involuntary eye movement": ("HP:0000639", "Nystagmus"),
    "dystonia": ("HP:0001332", "Dystonia"),
    "tremor": ("HP:0001337", "Tremor"),
    "ptosis": ("HP:0000508", "Ptosis"),
    "drooping eyelid": ("HP:0000508", "Ptosis"),
    "proteinuria": ("HP:0000093", "Proteinuria"),
    "protein in urine": ("HP:0000093", "Proteinuria"),
}


@dataclass
class ExtractedPhenotype:
    hpo_id: str
    label: str
    matched_phrase: str
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    input_text: str
    phenotypes: list[ExtractedPhenotype] = field(default_factory=list)

    @property
    def hpo_ids(self) -> list[str]:
        return [p.hpo_id for p in self.phenotypes]

    def __repr__(self) -> str:
        lines = [f"Input: {self.input_text[:80]}..."]
        for p in self.phenotypes:
            lines.append(f"  {p.hpo_id}  {p.label}  (matched: '{p.matched_phrase}')")
        return "\n".join(lines)


class SymptomExtractor:
    """
    Phase 1: dictionary-based HPO term extractor.

    Normalizes input text, scans for known symptom phrases,
    and returns deduplicated HPO term matches.

    Usage:
        extractor = SymptomExtractor()
        result = extractor.extract("Patient has muscle weakness and seizures.")
        print(result.hpo_ids)  # ['HP:0003326', 'HP:0001250']
    """

    def __init__(self, custom_terms: Optional[dict[str, tuple[str, str]]] = None):
        self._terms = {**HPO_SEED, **(custom_terms or {})}
        # Sort by phrase length descending so longer matches take priority
        self._sorted_terms = sorted(self._terms.keys(), key=len, reverse=True)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract(self, text: str) -> ExtractionResult:
        """Extract HPO phenotypes from free-text clinical description."""
        result = ExtractionResult(input_text=text)
        normalized = self._normalize(text)
        seen_ids: set[str] = set()

        for phrase in self._sorted_terms:
            if phrase in normalized:
                hpo_id, label = self._terms[phrase]
                if hpo_id not in seen_ids:
                    result.phenotypes.append(
                        ExtractedPhenotype(
                            hpo_id=hpo_id,
                            label=label,
                            matched_phrase=phrase,
                        )
                    )
                    seen_ids.add(hpo_id)

        return result

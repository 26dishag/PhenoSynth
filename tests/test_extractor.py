"""Tests for SymptomExtractor."""

import pytest
from phenosynth.nlp.extractor import SymptomExtractor


@pytest.fixture
def extractor():
    return SymptomExtractor()


def test_basic_extraction(extractor):
    result = extractor.extract(
        "Patient presents with muscle weakness, elevated creatine kinase, and cardiomyopathy."
    )
    assert "HP:0003326" in result.hpo_ids  # muscle weakness
    assert "HP:0003236" in result.hpo_ids  # elevated creatine kinase
    assert "HP:0001638" in result.hpo_ids  # cardiomyopathy


def test_case_insensitive(extractor):
    result = extractor.extract("SEIZURES and Ataxia observed.")
    assert "HP:0001250" in result.hpo_ids  # seizure
    assert "HP:0001251" in result.hpo_ids  # ataxia


def test_no_duplicates(extractor):
    result = extractor.extract("muscle weakness and myalgia and muscle weakness again")
    ids = result.hpo_ids
    assert len(ids) == len(set(ids)), "Duplicate HPO IDs found"


def test_empty_text(extractor):
    result = extractor.extract("")
    assert result.phenotypes == []


def test_no_matches(extractor):
    result = extractor.extract("The weather is nice today.")
    assert result.phenotypes == []


def test_hpo_ids_property(extractor):
    result = extractor.extract("fatigue and jaundice")
    assert isinstance(result.hpo_ids, list)
    assert all(id.startswith("HP:") for id in result.hpo_ids)


def test_custom_terms():
    custom = {"purple toenails": ("HP:9999999", "Purple toenails")}
    extractor = SymptomExtractor(custom_terms=custom)
    result = extractor.extract("Patient has purple toenails.")
    assert "HP:9999999" in result.hpo_ids

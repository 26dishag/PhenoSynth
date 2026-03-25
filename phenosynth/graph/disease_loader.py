"""
Disease Annotation Loader.

Downloads the HPO disease annotation file (phenotype.hpoa) from the official
HPO project and builds two indexes:

1. disease_to_hpo:  disease ID -> list of associated HPO term IDs
   Used to score diseases against a patient's symptom set.

2. hpo_to_diseases: HPO term ID -> list of disease IDs that have this term
   Used to calculate term specificity (how rare a symptom is across diseases).

Data source: Human Phenotype Ontology project
File: phenotype.hpoa (~12,000 diseases from OMIM and Orphanet)
URL: https://github.com/obophenotype/human-phenotype-ontology/releases
"""

from __future__ import annotations

import csv
import json
import logging
import urllib.request
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

HPOA_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-04-26/phenotype.hpoa"
HPOA_FALLBACK_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/phenotype.hpoa"

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
HPOA_PATH = DATA_DIR / "phenotype.hpoa"
DISEASE_INDEX_PATH = DATA_DIR / "disease_index.json"


def download_hpoa(force: bool = False) -> Path:
    """Download phenotype.hpoa if not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if HPOA_PATH.exists() and not force:
        logger.info("Disease annotation file already exists, skipping download.")
        return HPOA_PATH

    print("Downloading disease annotation file (~8 MB). This only happens once...")

    for url in [HPOA_URL, HPOA_FALLBACK_URL]:
        try:
            urllib.request.urlretrieve(url, HPOA_PATH)
            size_mb = HPOA_PATH.stat().st_size / 1_000_000
            print(f"Downloaded phenotype.hpoa ({size_mb:.1f} MB)")
            return HPOA_PATH
        except Exception as exc:
            logger.warning("Failed to download from %s: %s", url, exc)

    raise RuntimeError(
        "Could not download phenotype.hpoa. "
        "Check your internet connection or manually place the file in data/."
    )


def build_disease_index(force: bool = False) -> dict:
    """
    Parse phenotype.hpoa and build the disease and term indexes.

    Returns a dict with two keys:
        "disease_to_hpo": { disease_id: { "name": str, "hpo_terms": [hpo_id, ...] } }
        "hpo_to_diseases": { hpo_id: [disease_id, ...] }
    """
    if DISEASE_INDEX_PATH.exists() and not force:
        logger.info("Loading cached disease index.")
        with open(DISEASE_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    if not HPOA_PATH.exists():
        raise FileNotFoundError(
            f"Disease annotation file not found at {HPOA_PATH}. "
            "Run download_hpoa() first."
        )

    print("Building disease index from phenotype.hpoa...")

    disease_to_hpo: dict[str, dict] = {}
    hpo_to_diseases: dict[str, list[str]] = defaultdict(list)

    with open(HPOA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comment and header lines
            if line.startswith("#") or line.startswith("database_id"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue

            # Columns: database_id, disease_name, qualifier, hpo_id,
            #          reference, evidence, onset, frequency, sex, modifier, aspect, biocuration
            disease_id   = parts[0]   # e.g. OMIM:310200
            disease_name = parts[1]   # e.g. Duchenne muscular dystrophy
            hpo_id       = parts[3]   # e.g. HP:0003701
            aspect       = parts[10]  # P=phenotype, I=inheritance, C=clinical course

            # Only include phenotype annotations, skip inheritance and onset rows
            if aspect != "P":
                continue

            if not hpo_id.startswith("HP:"):
                continue

            if disease_id not in disease_to_hpo:
                disease_to_hpo[disease_id] = {
                    "name": disease_name,
                    "hpo_terms": [],
                }

            if hpo_id not in disease_to_hpo[disease_id]["hpo_terms"]:
                disease_to_hpo[disease_id]["hpo_terms"].append(hpo_id)
                hpo_to_diseases[hpo_id].append(disease_id)

    index = {
        "disease_to_hpo": disease_to_hpo,
        "hpo_to_diseases": dict(hpo_to_diseases),
    }

    total_diseases = len(disease_to_hpo)
    total_terms = len(hpo_to_diseases)
    print(f"Disease index built: {total_diseases:,} diseases, {total_terms:,} unique HPO terms.")

    with open(DISEASE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Disease index cached to {DISEASE_INDEX_PATH}")

    return index


def load_disease_index(force_download: bool = False, force_rebuild: bool = False) -> dict:
    """
    Top-level convenience function. Downloads and parses phenotype.hpoa,
    returns the complete disease index ready to use.
    """
    download_hpoa(force=force_download)
    return build_disease_index(force=force_rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index = load_disease_index()
    d2h = index["disease_to_hpo"]

    # Show a sample disease
    sample_id = "OMIM:310200"
    if sample_id in d2h:
        disease = d2h[sample_id]
        print(f"\n{sample_id}: {disease['name']}")
        print(f"  HPO terms ({len(disease['hpo_terms'])}):", disease["hpo_terms"][:5], "...")

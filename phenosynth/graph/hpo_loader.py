"""
HPO Ontology Loader.

Downloads the Human Phenotype Ontology (hp.obo) from the official HPO project,
parses it using the pronto library, and builds a flat term index mapping every
known label and synonym to its HPO ID.

The index looks like this:
    {
        "muscle weakness":         ("HP:0003326", "Myalgia"),
        "myalgia":                 ("HP:0003326", "Myalgia"),
        "pain in muscles":         ("HP:0003326", "Myalgia"),
        "muscular pain":           ("HP:0003326", "Myalgia"),
        ...17,000 more entries...
    }

This index is what the fuzzy matcher searches against in the next module.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import pronto

logger = logging.getLogger(__name__)

# Official HPO release URL (stable, versioned releases from the HPO project)
HPO_OBO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2024-04-26/hp.obo"
HPO_OBO_FALLBACK_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"

# Local paths
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
HPO_OBO_PATH = DATA_DIR / "hp.obo"
HPO_INDEX_PATH = DATA_DIR / "hpo_index.json"


def download_hpo(force: bool = False) -> Path:
    """
    Download hp.obo from the HPO project if not already cached locally.

    Args:
        force: Re-download even if the file already exists.

    Returns:
        Path to the downloaded .obo file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if HPO_OBO_PATH.exists() and not force:
        logger.info("HPO file already exists at %s, skipping download.", HPO_OBO_PATH)
        return HPO_OBO_PATH

    logger.info("Downloading HPO ontology from %s ...", HPO_OBO_URL)
    print("Downloading HPO ontology file (~10 MB). This only happens once...")

    for url in [HPO_OBO_URL, HPO_OBO_FALLBACK_URL]:
        try:
            urllib.request.urlretrieve(url, HPO_OBO_PATH)
            size_mb = HPO_OBO_PATH.stat().st_size / 1_000_000
            print(f"Downloaded hp.obo ({size_mb:.1f} MB) to {HPO_OBO_PATH}")
            return HPO_OBO_PATH
        except Exception as exc:
            logger.warning("Failed to download from %s: %s", url, exc)

    raise RuntimeError(
        "Could not download hp.obo from any URL. "
        "Check your internet connection or manually place hp.obo in the data/ folder."
    )


def build_index(obo_path: Optional[Path] = None, force: bool = False) -> dict[str, tuple[str, str]]:
    """
    Parse the HPO OBO file and build a flat label+synonym index.

    Each entry maps a normalized text string to (hpo_id, primary_label).
    Synonyms, exact synonyms, and related synonyms are all included.

    Args:
        obo_path: Path to hp.obo. Defaults to the standard data/ location.
        force: Rebuild even if the cached index JSON already exists.

    Returns:
        dict mapping lowercase term/synonym text to (HPO ID, primary label).
    """
    if obo_path is None:
        obo_path = HPO_OBO_PATH

    # Use cached JSON index if available
    if HPO_INDEX_PATH.exists() and not force:
        logger.info("Loading cached HPO index from %s", HPO_INDEX_PATH)
        with open(HPO_INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: tuple(v) for k, v in raw.items()}  # type: ignore[return-value]

    if not obo_path.exists():
        raise FileNotFoundError(
            f"HPO OBO file not found at {obo_path}. "
            "Run download_hpo() first or call load_hpo_index() which does this automatically."
        )

    print("Parsing HPO ontology. This takes about 20-30 seconds on first run...")
    ontology = pronto.Ontology(str(obo_path))

    index: dict[str, tuple[str, str]] = {}
    skipped = 0

    for term in ontology.terms():
        # Only include phenotype terms (HP:XXXXXXX), skip metadata terms
        if not term.id.startswith("HP:"):
            skipped += 1
            continue

        hpo_id = term.id
        primary_label = term.name if term.name else hpo_id

        # Index the primary label
        _add_to_index(index, primary_label, hpo_id, primary_label)

        # Index all synonyms
        for synonym in term.synonyms:
            _add_to_index(index, synonym.description, hpo_id, primary_label)

    total = len(index)
    print(f"HPO index built: {total:,} searchable entries from {len(list(ontology.terms()))} terms.")

    # Cache to disk so future loads are instant
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HPO_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Index cached to {HPO_INDEX_PATH} for fast future loads.")

    return index


def _add_to_index(
    index: dict[str, tuple[str, str]],
    text: str,
    hpo_id: str,
    primary_label: str,
) -> None:
    """Normalize text and add to index, skipping empty or very short strings."""
    if not text:
        return
    normalized = text.strip().lower()
    if len(normalized) < 3:
        return
    # Only add if not already present (first definition wins)
    if normalized not in index:
        index[normalized] = (hpo_id, primary_label)


def load_hpo_index(force_download: bool = False, force_rebuild: bool = False) -> dict[str, tuple[str, str]]:
    """
    High-level convenience function. Downloads HPO if needed, builds the index,
    and returns it ready to use.

    Args:
        force_download: Re-download the OBO file even if already cached.
        force_rebuild: Rebuild the index even if a cached JSON exists.

    Returns:
        dict mapping lowercase term/synonym text to (HPO ID, primary label).
    """
    download_hpo(force=force_download)
    return build_index(force=force_rebuild)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index = load_hpo_index()
    print(f"\nSample entries:")
    for key in list(index.keys())[:10]:
        hpo_id, label = index[key]
        print(f"  '{key}' -> {hpo_id} ({label})")

"""
PhenoSynth API server.

Serves the web UI and exposes the /analyze endpoint.
The /analyze endpoint accepts a plain-text symptom description,
extracts HPO phenotype terms, and returns ranked disease matches.

Run with:
    uvicorn phenosynth.api.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from phenosynth.nlp.extractor import SymptomExtractor, ExtractionResult
from phenosynth.graph.disease_loader import load_disease_index
from phenosynth.graph.disease_scorer import DiseaseScorer

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhenoSynth",
    description="AI-powered rare disease differential diagnosis via HPO graph reasoning.",
    version="0.2.0",
)

# Serve static files (HTML, CSS, JS) from the top-level /static directory
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load all components once at startup
_extractor = SymptomExtractor()
_disease_index = load_disease_index()
_scorer = DiseaseScorer(_disease_index)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Free-text clinical description or patient-reported symptoms.",
        examples=["Patient presents with muscle weakness, seizures, and elevated creatine kinase."],
    )


class PhenotypeMatch(BaseModel):
    hpo_id: str
    label: str
    matched_phrase: str
    confidence: float


class DiseaseResult(BaseModel):
    disease_id: str
    disease_name: str
    match_percent: float
    matched_terms: list[str]
    total_disease_terms: int
    coverage: float


class AnalyzeResponse(BaseModel):
    input_text: str
    phenotypes: list[PhenotypeMatch]
    diseases: list[DiseaseResult]
    total_phenotype_matches: int
    total_disease_matches: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the main web UI."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            content="<h1>UI not found. Run from project root.</h1>",
            status_code=404,
        )
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_symptoms(request: AnalyzeRequest):
    """
    Full analysis pipeline:
    1. Extract HPO phenotype terms from free text using SciSpaCy
    2. Score all diseases against the extracted HPO terms
    3. Return ranked phenotypes and ranked disease matches
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty or whitespace.")

    # Step 1: extract HPO terms
    extraction: ExtractionResult = _extractor.extract(request.text)

    phenotypes = [
        PhenotypeMatch(
            hpo_id=p.hpo_id,
            label=p.label,
            matched_phrase=p.matched_phrase,
            confidence=p.confidence,
        )
        for p in extraction.phenotypes
    ]

    # Step 2: score diseases
    hpo_ids = [p.hpo_id for p in extraction.phenotypes]
    scoring = _scorer.score(hpo_ids, top_n=10)

    diseases = [
        DiseaseResult(
            disease_id=m.disease_id,
            disease_name=m.disease_name,
            match_percent=m.match_percent,
            matched_terms=m.matched_terms,
            total_disease_terms=m.total_disease_terms,
            coverage=m.coverage,
        )
        for m in scoring.matches
    ]

    return AnalyzeResponse(
        input_text=request.text,
        phenotypes=phenotypes,
        diseases=diseases,
        total_phenotype_matches=len(phenotypes),
        total_disease_matches=len(diseases),
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return JSONResponse({"status": "ok", "version": "0.2.0"})

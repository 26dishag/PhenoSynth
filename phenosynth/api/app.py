"""
PhenoSynth API server.

Serves the web UI and exposes the /analyze endpoint.
The /analyze endpoint accepts a plain-text symptom description
and returns a list of matched HPO phenotype terms.

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PhenoSynth",
    description="AI-powered rare disease differential diagnosis via HPO graph reasoning.",
    version="0.1.0",
)

# Serve static files (HTML, CSS, JS) from the top-level /static directory
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Single shared extractor instance (loaded once at startup)
_extractor = SymptomExtractor()


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
    hpo_id: str = Field(description="HPO term identifier, e.g. HP:0003326")
    label: str = Field(description="Human-readable HPO term label")
    matched_phrase: str = Field(description="The phrase in the input text that triggered this match")
    confidence: float = Field(description="Confidence score between 0 and 1")


class AnalyzeResponse(BaseModel):
    input_text: str
    phenotypes: list[PhenotypeMatch]
    total_matches: int


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
    Analyze a free-text symptom description and return matched HPO phenotypes.

    - Normalizes the input text
    - Scans for known HPO symptom phrases (Phase 1: dictionary-based)
    - Returns deduplicated matches sorted by HPO ID
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty or whitespace.")

    result: ExtractionResult = _extractor.extract(request.text)

    phenotypes = [
        PhenotypeMatch(
            hpo_id=p.hpo_id,
            label=p.label,
            matched_phrase=p.matched_phrase,
            confidence=p.confidence,
        )
        for p in result.phenotypes
    ]

    return AnalyzeResponse(
        input_text=request.text,
        phenotypes=phenotypes,
        total_matches=len(phenotypes),
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return JSONResponse({"status": "ok", "version": "0.1.0"})

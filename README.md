# PhenoSynth 🧬

> AI-powered rare disease differential diagnosis via Human Phenotype Ontology (HPO) graph reasoning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/status-active-brightgreen.svg)]()

---

## The Problem

Patients with rare diseases wait an average of **5–7 years** to receive a correct diagnosis. These diseases are hard to identify because:
- Training data is scarce (by definition, rare diseases affect few people)
- Symptoms overlap across hundreds of conditions
- Most AI tools only work well on common diseases

## What PhenoSynth Does

PhenoSynth takes unstructured clinical text (a doctor's note or patient symptom description) and:

1. **Extracts symptoms** using a medical NLP pipeline
2. **Maps them** to the Human Phenotype Ontology (HPO) — a standardized graph of 17,000+ phenotype terms
3. **Reasons over the graph** using a Graph Neural Network (GNN) to find matching rare diseases — even ones it has never seen before (zero-shot)
4. **Generates synthetic patients** using a conditional VAE to augment training data for ultra-rare conditions
5. **Returns a ranked differential diagnosis** with natural language explanations

---

## Architecture

```
phenosynth/
├── nlp/          # Symptom extraction + HPO term mapping
├── graph/        # HPO graph loading + GNN reasoning
├── synthesis/    # Synthetic patient data generation (cVAE)
├── api/          # FastAPI inference endpoint
└── eval/         # Benchmarks against OMIM / Orphanet
```

---

## Quickstart

```bash
git clone https://github.com/26dishag/PhenoSynth.git
cd PhenoSynth
pip install -r requirements.txt
python -m phenosynth.api.app
```

---

## Example

```python
from phenosynth.nlp.extractor import SymptomExtractor

extractor = SymptomExtractor()
phenotypes = extractor.extract("Patient presents with muscle weakness, elevated creatine kinase, and cardiomyopathy.")
print(phenotypes)
# ['HP:0003326', 'HP:0003236', 'HP:0001638']
```

---

## Roadmap

- [x] Project structure + README
- [ ] NLP symptom extraction pipeline
- [ ] HPO graph loader + GNN model
- [ ] Synthetic patient generator (cVAE)
- [ ] FastAPI inference endpoint
- [ ] Evaluation benchmarks
- [ ] Research paper (LaTeX)

---

## Clinical Relevance

PhenoSynth is designed to support — not replace — clinical decision making. It is intended as a research tool and has not been validated for direct clinical use.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

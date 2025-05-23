# AgGRU-Check

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13147167.svg)](https://doi.org/10.5281/zenodo.13147167)

Identify and visualize amyloidogenic domains in proteins using gated recurrent units. Try it live at [aggru-check.fly.dev](https://aggru-check.fly.dev)!

Read more about this model in [details.pdf](details.pdf).

## Overview

AgGRU-Check uses an ensemble of bidirectional GRUs to predict and analyze amyloidogenic regions in protein sequences. The model combines sliding window analysis with ensemble predictions to provide:

- Position-specific amyloid propensity scores
- Confidence measures based on ensemble agreement
- Interactive visualization of amyloidogenic domains
- Detailed window-based structural analysis
- Exportable results in CSV format

## Features

### Advanced Analysis
- Multi-scale window analysis (15-27 residues)
- Ensemble prediction with confidence scoring
- Position-specific saliency mapping

### Interactive Visualization
- Color-coded propensity scores
- Detailed residue-level tooltips
- Window analysis overlays
- Confidence indicators
- Sequence position markers

### Input Support
- Raw amino acid sequences
- FASTA format with headers
- Multiple sequence analysis
- Batch processing with streaming results

## Project Structure

```
.
├── deployment/                  # Deployment-ready code
│   ├── backend/                # FastAPI server
│   │   └── main.py            # Core analysis engine
│   └── frontend/              # SvelteKit UI
│       └── src/
│           ├── routes/        # Main pages
│           └── lib/           # Shared components
├── details.pdf                # Technical paper
└── amyloidClassification.ipynb # Training notebook
```

## Quick Start

### Using the Web Interface

1. Visit [aggru-check.fly.dev](https://aggru-check.fly.dev)
2. Paste your protein sequence(s)
3. Click "Predict" to see interactive results
4. Hover over residues for detailed predictions
5. Download results as CSV if needed

Example input:
```
>TDP-43 (amyloid domain) [318-343]
INPAMMAAAQAALKSSWGMMGMLASQ
```

### Local Development

#### Backend
```bash
cd deployment/backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend
```bash
cd deployment/frontend
npm install
npm run dev
```

## API Usage

The API accepts POST requests and streams real-time analysis results:

```bash
curl -X POST https://aggru-check.fly.dev/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sequenceList": [">TDP-43\nINPAMMAAA"]}'
```

The API streams results as Server-Sent Events (SSE) including:
- Real-time position-specific scores
- Per-model analysis progress
- Ensemble agreement metrics
- Position-specific confidence scores
- Detailed window analysis results

Example response events:
```json
{"type": "sequence_start", "sequence_index": 0, "total_models": 5}
{"type": "position_result", "position": 10, "saliency": 0.85, "confidence": 0.92}
{"type": "model_complete", "sequence_index": 0, "model_index": 1}
```

## Training Data

Before running the training notebooks, download these datasets:

- GPD_proteome.faa
- GPD_proteome_orthology_assignment.txt

Available at: [Gut Phage Database](https://datacommons.cyverse.org/browse/iplant/home/shared/iVirus/Gut_Phage_Database)

## Technical Details

- **Model Architecture**: 
  - Bidirectional GRU ensemble
  - Multi-scale window analysis
  
- **Analysis Methodology**:
  - **Adaptive Window Analysis**: Uses multiple window sizes (15, 21, 27 residues) to capture both local and broader structural patterns
  - **Context-Aware Scoring**:
    - For amyloid sequences: Identifies critical regions by masking windows and measuring prediction impact
    - For non-amyloid sequences: Requires stronger evidence with higher thresholds for positive predictions
  - **Position Scoring**:
    - Gaussian weighting around window centers
    - Nonlinear impact scaling to emphasize strong signals
    - Conservative score aggregation using weighted max/mean combinations
  - **Confidence Calculation**:
    - Based on ensemble agreement and prediction variance
    - Scales with number of models completed
    - Incorporates standard deviation of predictions

- **Stack**: 
  - Backend: PyTorch, FastAPI
  - Frontend: SvelteKit, TypeScript
  - Deployment: Docker + Fly.io

## Citation

If you use AgGRU-Check in your research, please cite:

```bibtex
@software{aggru_check_2024,
  author = {Mockler, Evan},
  doi = {10.5281/zenodo.13147167},
  title = {AgGRU-Check: Interactive Analysis of Amyloidogenic Domains},
  url = {https://github.com/ejmockler/AgGRU-Check}
}
```

## License

MIT

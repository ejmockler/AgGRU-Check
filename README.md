# AgGRU-Check

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13147167.svg)](https://doi.org/10.5281/zenodo.13147167)

Identify and visualize amyloidogenic domains in proteins using gated recurrent units. Try it live at [aggru-check.fly.dev](https://aggru-check.fly.dev)

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
- Permutation-based significance testing
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
- FASTQ format
- Multiple sequence analysis (up to 5)
- Batch processing capability

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

The API accepts POST requests and streams results:

```bash
curl -X POST https://aggru-check.fly.dev/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sequenceList": [">TDP-43\nINPAMMAAA"]}'
```

Response includes:
- Position-specific scores
- Window analysis results
- Confidence measures
- Model ensemble agreement

## Training Data

Before running the training notebooks, download these datasets:

- GPD_proteome.faa
- GPD_proteome_orthology_assignment.txt

Available at: [Gut Phage Database](https://datacommons.cyverse.org/browse/iplant/home/shared/iVirus/Gut_Phage_Database)

## Technical Details

- **Model Architecture**: 
  - Bidirectional GRU ensemble
  - Multi-scale window analysis
  - Permutation-based significance testing
  
- **Analysis Features**:
  - Position-specific scoring
  - Confidence calculation
  - Window-based domain detection
  - Ensemble agreement metrics

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

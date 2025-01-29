# AgGRU-Check

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13147167.svg)](https://doi.org/10.5281/zenodo.13147167)

Gated recurrent units classify amyloidogenic proteins. Try it live at [aggru-check.fly.dev](https://aggru-check.fly.dev)

## Overview

AgGRU-Check uses an ensemble of bidirectional GRUs to predict amyloidogenic regions in protein sequences. The model was trained on a curated dataset of known amyloids and validated against experimental data.

## Project Structure

```
.
├── deployment/                  # Deployment-ready code
│   ├── backend/                 # FastAPI server
│   └── frontend/                # SvelteKit UI
├── details.pdf                  # Technical paper
└── amyloidClassification.ipynb  # Training notebook
```

## Quick Start

### Using the Web Interface

1. Visit [aggru-check.fly.dev](https://aggru-check.fly.dev)
2. Paste your protein sequence(s) in FASTA format
3. Click "Predict"

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

The API accepts POST requests with protein sequences:

```bash
curl -X POST https://aggru-check.fly.dev/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sequences": [">TDP-43\nINPAMMAAA"]}'
```

## Training Data

Before running the training notebooks, download these datasets:

- GPD_proteome.faa
- GPD_proteome_orthology_assignment.txt

Available at: [Gut Phage Database](https://datacommons.cyverse.org/browse/iplant/home/shared/iVirus/Gut_Phage_Database)

## Technical Details

- **Model**: Bidirectional GRU ensemble
- **Input**: Protein sequences (FASTA/FASTQ)
- **Output**: Amyloidogenic probability (0-1)
- **Stack**: PyTorch Lightning, FastAPI, SvelteKit
- **Deployment**: Docker + Fly.io

## Citation

If you use AgGRU-Check in your research, please cite:

```bibtex
@software{aggru_check_2024,
  author = {Mockler, Evan},
  doi = {10.5281/zenodo.13147167},
  title = {AgGRU-Check},
  url = {https://github.com/ejmockler/AgGRU-Check}
}
```

## License

MIT

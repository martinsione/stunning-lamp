# Vendor Clustering Service

A service that clusters and standardizes vendor names using pattern matching, fuzzy string matching, and optional LLM assistance.

## Features

- Pattern matching for known vendor formats
- Fuzzy string matching using Levenshtein distance
- DBSCAN clustering for grouping similar vendors
- OpenAI GPT-4o-mini for cluster naming
- Health check endpoint
- Vercel deployment ready

## Setup

1. Set the Python version using pyenv:
```bash
pyenv local 3.12.7  # or your preferred Python version
```

2. Create a virtual environment:
```bash
~/.pyenv/versions/3.12.7/bin/python -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the service:
```bash
python api/main.py
```

The service will be available at http://localhost:8000

## API Usage

### GET /health

Health check endpoint that returns the service status.

Response:
```json
{
    "status": "healthy"
}
```

### POST /api/process-vendors

Process a list of vendor names and get clustering results.

Request body:
```json
{
    "vendor_names": [
        "AMAZON MKTPL*XM1EJ9M33",
        "AMAZON.COM",
        "AMAZON WEB SERVICES"
    ]
}
```

Response:
```json
[
    {
        "vendor_name": "AMAZON MKTPL*XM1EJ9M33",
        "cluster": "Amazon Retail",
        "recommendation": "Amazon Retail",
        "confidence": 1.0
    },
    {
        "vendor_name": "AMAZON.COM",
        "cluster": "Amazon Retail",
        "recommendation": "Amazon Retail",
        "confidence": 1.0
    },
    {
        "vendor_name": "AMAZON WEB SERVICES",
        "cluster": "Amazon Web Services",
        "recommendation": "Amazon Web Services",
        "confidence": 1.0
    }
]
```

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── vendor_clustering.py # Clustering logic
├── requirements.txt
├── README.md
└── vercel.json             # Vercel deployment config
```

## Deployment

### Vercel Deployment

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel
```

4. Set environment variables:
```bash
vercel env add OPENAI_API_KEY
```

## How it Works

1. **Pattern Matching**: First tries to match vendors against known patterns (e.g., "AMAZON MKTPL*" → "Amazon Retail")
2. **Fuzzy Clustering**: For unmatched vendors, uses DBSCAN clustering with Levenshtein distance
3. **LLM Naming**: Uses GPT-4o-mini to suggest standardized names for clusters
4. **Confidence Scoring**: 
   - Pattern matches: 1.0 confidence
   - Fuzzy clusters: Average similarity between cluster members
   - Unmatched: 0.5 confidence

## Adding New Patterns

Edit the `patterns` dictionary in `api/vendor_clustering.py` to add new vendor patterns:

```python
self.patterns = {
    r"PATTERN_REGEX": "Standardized Name",
    ...
} 
```
FROM python:3.10-slim

LABEL maintainer="hiddenlayers-predictive-sales"
LABEL description="Confidence-Gated Neuro-Symbolic Hybrid — B2B SaaS sales predictor"

# System dependencies for matplotlib / PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Smoke test — fail build if imports break
RUN python -c "\
import pandas, numpy, sklearn, torch, matplotlib, seaborn, transformers, tqdm; \
print('All imports OK')"

# Default: list available notebooks
CMD ["python", "-c", "\
import os; \
nbs = sorted(f for f in os.listdir('notebooks') if f.endswith('.ipynb')); \
print('Available notebooks:'); \
[print(f'  notebooks/{nb}') for nb in nbs]"]

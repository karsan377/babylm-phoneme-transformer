FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false USE_TF=0
CMD ["python", "scripts/validate_setup.py", "--skip-data-check"]

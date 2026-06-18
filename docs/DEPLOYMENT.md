# Deployment Guide

## Local Streamlit Demo

```bash
pip install -r requirements.txt
python train.py --data_dir dataset/ --epochs 30 --fine_tune
streamlit run app/streamlit_app.py
```

## Streamlit Community Cloud

1. Push this repository to GitHub (without `dataset/` or large model files — see `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app/streamlit_app.py`.
4. Add model weights:
   - Option A: Train in a one-off job and upload `models/brain_tumor_model.h5` to cloud storage, download in `app/streamlit_app.py` on startup.
   - Option B: Commit a release artifact via GitHub Releases and fetch by URL (keep under size limits).
5. Set Python version to 3.11.

## Hugging Face Spaces (Streamlit SDK)

1. Create a new Space with SDK **Streamlit**.
2. Push this codebase (or mirror the repo).
3. Set `app_file` to `app/streamlit_app.py` in the Space README metadata:

```yaml
---
title: Brain Tumor MRI Classifier
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
license: mit
---
```

4. Include `requirements.txt` at the repository root.
5. Upload trained weights to the Space (via Git LFS or `huggingface_hub`) or document that users must train first.

## Docker (optional local serving)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t brain-tumor-ai .
docker run -p 8501:8501 -v $(pwd)/models:/app/models brain-tumor-ai
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `BRAIN_TUMOR_MODEL_PATH` | Optional override for model file (not wired by default; extend `brain_tumor/config.py` if needed) |

## CI

GitHub Actions runs unit tests on push (see `.github/workflows/ci.yml`). Full TensorFlow inference tests are skipped in CI when model weights are absent.

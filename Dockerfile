FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements-embeddings.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-embeddings.txt
COPY label_rag ./label_rag
ENV INDEX_DIR=/app/data/index
EXPOSE 8000
# Build the index first (see README) and mount it: -v $(pwd)/data:/app/data
CMD ["uvicorn", "label_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

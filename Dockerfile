FROM node:22-alpine AS web-build
WORKDIR /web
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PKA_CONFIG=/app/config/docker.yaml
COPY requirements.txt ./
RUN pip install -r requirements.txt
ARG INSTALL_LOCAL_MODELS=false
RUN if [ "$INSTALL_LOCAL_MODELS" = "true" ]; then pip install --extra-index-url https://download.pytorch.org/whl/cpu 'torch>=2,<3' 'sentence-transformers>=3,<6'; fi
COPY backend/ ./backend/
COPY config/docker.yaml ./config/docker.yaml
COPY --from=web-build /web/dist ./static
RUN mkdir -p /app/data /app/KnowledgeBase
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "backend.api.web:app", "--host", "0.0.0.0", "--port", "8000"]

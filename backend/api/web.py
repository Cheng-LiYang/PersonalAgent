"""Serve the browser client and API from one container and one origin."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.server import app as api

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/api", api)
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[2] / "static", html=True), name="web")

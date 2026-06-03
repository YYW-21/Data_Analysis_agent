from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from datapilot.api import datasets, jobs
from datapilot.core.config import settings
from datapilot.core.storage import ensure_storage_dirs

ensure_storage_dirs()

app = FastAPI(title="DataPilot", version="0.1.0")
app.include_router(datasets.router)
app.include_router(jobs.router)
app.mount("/storage", StaticFiles(directory=str(settings.data_dir)), name="storage")
app.mount("/web", StaticFiles(directory="src/datapilot/web"), name="web")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse("src/datapilot/web/index.html")

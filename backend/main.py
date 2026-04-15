from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import laps, track, validation, visualization

app = FastAPI(title="FSAE Sim API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(laps.router)
app.include_router(track.router)
app.include_router(validation.router)
app.include_router(visualization.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}

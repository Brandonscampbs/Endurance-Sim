"""FastAPI entry point for the FSAE Sim webapp backend.

Request flow (outermost to innermost):

1. :class:`backend.middleware.RequestIdMiddleware` -- assigns/propagates
   ``X-Request-ID``, binds it to a :mod:`contextvars` slot so log records
   and error responses pick it up automatically, and emits a one-line
   access log with method, path, status, and duration.
2. :class:`fastapi.middleware.cors.CORSMiddleware` -- handles the browser
   preflight and adds ``Access-Control-Allow-*`` headers. It lives inside
   the request-ID middleware so that failures originating here are still
   tagged with a request ID.
3. The FastAPI application itself, which dispatches to the routers defined
   under :mod:`backend.routers`. Exceptions raised inside the app are
   translated into structured JSON responses by the handlers registered in
   :mod:`backend.errors`.

Logging is configured via ``logging.basicConfig`` with a format that includes
the active request ID (``-`` when no request is in flight). The level is
controlled by the ``LOG_LEVEL`` environment variable and defaults to
``INFO``.
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.errors import register_exception_handlers
from backend.middleware import RequestIdLogFilter, RequestIdMiddleware
from backend.routers import cache, laps, track, validation, visualization


def _configure_logging() -> logging.Logger:
    """Initialise root logging with a request-ID aware format.

    Using :func:`logging.basicConfig` with ``force=True`` keeps configuration
    idempotent when uvicorn reloads or when the module is imported under a
    test harness that has already touched the root logger.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s [%(levelname)s] [%(name)s] "
            "[req=%(request_id)s] %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    # Attach the request-ID filter to the root logger so every handler --
    # including uvicorn's own -- receives the attribute before formatting.
    request_id_filter = RequestIdLogFilter()
    root = logging.getLogger()
    root.addFilter(request_id_filter)
    for handler in root.handlers:
        handler.addFilter(request_id_filter)

    return logging.getLogger(__name__)


logger = _configure_logging()

app = FastAPI(title="FSAE Sim API", version="0.1.0")

# Order matters: the middleware added LAST via ``add_middleware`` runs
# FIRST for incoming requests. We want request-ID to wrap CORS so that a
# CORS-rejected preflight still carries a correlation ID in the logs.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)

app.include_router(cache.router)
app.include_router(laps.router)
app.include_router(track.router)
app.include_router(validation.router)
app.include_router(visualization.router)


@app.on_event("startup")
async def _on_startup() -> None:
    logger.info("backend starting (log_level=%s)", logging.getLevelName(logger.getEffectiveLevel()))


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    logger.info("backend shutting down")


@app.get("/api/health")
def health():
    return {"status": "ok"}

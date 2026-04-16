"""Centralised FastAPI exception handlers producing structured error bodies.

All handlers return a JSON body containing at minimum:

* ``detail`` -- human readable message (the frontend's ``ApiError`` reads this)
* ``status`` -- numeric HTTP status code
* ``request_id`` -- the ID assigned by :mod:`backend.middleware`, enabling a
  user-visible error to be matched against server logs.

Validation errors include the Pydantic/FastAPI error list under ``errors``.
Unhandled exceptions are logged at ``ERROR`` level with a full traceback but
the client only ever sees a generic "Internal server error" message -- we do
not leak tracebacks over the wire.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.middleware import get_request_id

_logger = logging.getLogger("backend.errors")


def _error_payload(status: int, detail: str, **extra) -> dict:
    """Assemble the canonical error body shape."""
    payload = {
        "detail": detail,
        "status": status,
        "request_id": get_request_id(),
    }
    payload.update(extra)
    return payload


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle ``HTTPException``/``StarletteHTTPException`` with a structured body.

    FastAPI re-exports ``HTTPException`` from Starlette; registering against
    the Starlette class means this handler fires for both.
    """
    _logger.warning(
        "HTTPException %d at %s: %s",
        exc.status_code,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.status_code, str(exc.detail)),
        headers=getattr(exc, "headers", None) or None,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic request-validation failures (422).

    We surface the raw ``errors`` list so the frontend can point users at the
    specific field that failed. ``detail`` is a short stable string used by
    the toast layer.
    """
    # RequestValidationError.errors() can contain non-JSON-serialisable values
    # (e.g. ``bytes`` input). Convert defensively.
    errors = []
    for err in exc.errors():
        safe = {}
        for key, value in err.items():
            try:
                import json

                json.dumps(value)
                safe[key] = value
            except (TypeError, ValueError):
                safe[key] = repr(value)
        errors.append(safe)

    _logger.info(
        "Validation failed at %s: %d issue(s)", request.url.path, len(errors)
    )
    return JSONResponse(
        status_code=422,
        content=_error_payload(422, "Validation failed", errors=errors),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for exceptions not matched by the more specific handlers.

    The traceback is logged with ``logger.exception`` so operators can debug
    the failure from the server logs. The client payload is intentionally
    generic to avoid leaking internals (stack frames, file paths, etc.).
    """
    _logger.exception(
        "Unhandled exception at %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content=_error_payload(500, "Internal server error"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers defined in this module to ``app``.

    Keeping registration in one function lets ``main.py`` stay a thin
    composition root.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

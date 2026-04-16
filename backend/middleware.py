"""Request-ID middleware and logging helpers for the FSAE Sim backend.

The middleware assigns every incoming request a stable identifier so that
log lines emitted during that request (and any error response returned to
the client) can be correlated. When the caller supplies an ``X-Request-ID``
header we reuse it; otherwise we generate a short random hex value.

The active request ID is stored in a :class:`contextvars.ContextVar` so the
logging formatter can inject it into every record without needing the ID to
be passed around by hand. The same ID is also written back to the response
headers as ``X-Request-ID`` for client-side correlation.
"""
from __future__ import annotations

import contextvars
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# ContextVar used by both the middleware and the logging formatter. Default
# ``"-"`` keeps log output aligned when a record is emitted outside of any
# request (e.g. startup / shutdown).
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def get_request_id() -> str:
    """Return the request ID for the current context, or ``"-"`` if unset."""
    return request_id_ctx.get()


class RequestIdLogFilter(logging.Filter):
    """Logging filter that attaches ``request_id`` to every ``LogRecord``.

    The stdlib ``logging`` module raises ``KeyError`` when a format string
    references an attribute that is not present on the record. Populating
    the attribute here makes the ``%(request_id)s`` token safe to use from
    any logger in the process.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib API
        record.request_id = request_id_ctx.get()
        return True


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that tags each request with a stable ID and logs the access line.

    Responsibilities:

    * Read ``X-Request-ID`` from the inbound headers, or generate a new one.
    * Bind the ID to :data:`request_id_ctx` for the duration of the request.
    * Emit a single INFO access log with method, path, status, and duration.
    * Echo the ID back on the response via ``X-Request-ID``.

    This middleware should be registered *outermost* so that errors surfaced
    by later middleware (including CORS) are still tagged with a request ID.
    """

    def __init__(self, app: ASGIApp, logger_name: str = "backend.access") -> None:
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("X-Request-ID")
        request_id = incoming if incoming else uuid.uuid4().hex[:12]
        token = request_id_ctx.set(request_id)

        start = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            # Use `extra` to attach structured fields without breaking the
            # default format string used by non-structured consumers.
            self._logger.info(
                "%s %s -> %d (%.2f ms)",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            request_id_ctx.reset(token)

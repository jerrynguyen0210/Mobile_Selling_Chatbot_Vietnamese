"""Custom exception classes and global exception handlers for FastAPI."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class AppBaseException(Exception):
    """Base class for all application exceptions."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class NotFoundError(AppBaseException):
    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found."


class ConflictError(AppBaseException):
    status_code = status.HTTP_409_CONFLICT
    detail = "Resource already exists."


class ValidationError(AppBaseException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    detail = "Validation failed."


class AuthenticationError(AppBaseException):
    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Authentication required."


class AuthorizationError(AppBaseException):
    status_code = status.HTTP_403_FORBIDDEN
    detail = "Permission denied."


class SessionNotFoundError(NotFoundError):
    detail = "Chat session not found."


class ProductNotFoundError(NotFoundError):
    detail = "Product not found."


class LLMError(AppBaseException):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "LLM service error."


class EmbeddingError(AppBaseException):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Embedding service error."


class VectorStoreError(AppBaseException):
    status_code = status.HTTP_502_BAD_GATEWAY
    detail = "Vector store error."


class CacheError(AppBaseException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Cache service unavailable."


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _error_body(status_code: int, detail: str) -> dict[str, object]:
    return {"error": {"status_code": status_code, "detail": detail}}


async def app_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.status_code, exc.detail),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An unexpected error occurred.",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to *app*."""
    app.add_exception_handler(AppBaseException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

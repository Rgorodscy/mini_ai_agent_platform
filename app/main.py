from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config import validate_settings
from app.logger import get_logger, setup_logging
from app.routers import (
    agent_router,
    execution_router,
    knowledge_router,
    tool_router,
)

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Settings that would only fail on the first LLM call are validated
    # here instead, so a misconfigured deployment dies at boot.
    validate_settings()
    logger.info("Application started")
    yield


app = FastAPI(
    title="Mini Agent Platform",
    description="Multi-tenant AI Agent Platform",
    version="0.1.0",
    redirect_slashes=False,
    lifespan=lifespan,
)

app.include_router(tool_router.router)
app.include_router(agent_router.router)
app.include_router(execution_router.router)
app.include_router(knowledge_router.router)


@app.exception_handler(SQLAlchemyError)
def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error | path={request.url.path} error={str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "A database error occurred. Please try again later."
        },
    )


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled error | path={request.url.path} error={str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later."
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}

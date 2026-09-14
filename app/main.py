from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from app.config import validate_settings
from app.logger import get_logger, setup_logging
from app.routers import (
    agent_router,
    execution_router,
    knowledge_router,
    models_router,
    tool_router,
    usage_router,
)

UI_DIR = Path(__file__).parent / "static"

# The console renders model output and tool results, which are untrusted: a
# document pulled in by retrieval can contain markup. The page builds every
# dynamic node with textContent; this policy is the second layer, refusing
# inline script, foreign origins and framing even if markup slipped through.
UI_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class UIStaticFiles(StaticFiles):
    """
    Static files with the console's security headers.

    Headers are added here rather than by an HTTP middleware because a
    middleware would also wrap the API — including the streaming endpoint,
    whose behaviour on client disconnect depends on nothing sitting between
    it and the server.
    """

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers.update(UI_SECURITY_HEADERS)
        return response


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
app.include_router(usage_router.router)
app.include_router(models_router.router)


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


# The console's assets use relative URLs, which resolve against the wrong
# directory without the trailing slash — so both entry points redirect.
@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


@app.api_route("/ui", methods=["GET", "HEAD"], include_in_schema=False)
def ui_without_slash():
    return RedirectResponse(url="/ui/")


# Mounted last: routes are matched in registration order.
app.mount("/ui", UIStaticFiles(directory=UI_DIR, html=True), name="ui")

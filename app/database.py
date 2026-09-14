from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

# check_same_thread is a SQLite-only escape hatch: FastAPI hands a session
# to whichever worker thread serves the request. Passing it to any other
# driver raises, so it stays scoped to SQLite URLs.
connect_args = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_factory():
    """
    The factory for sessions that must outlive the request.

    A streamed run keeps going after the client disconnects, and by then the
    request's own session has been closed — so the thread that records the
    run opens its own. A dependency rather than a direct import so tests can
    point it at their database.
    """
    return SessionLocal

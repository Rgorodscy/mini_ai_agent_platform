import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.middleware.auth import get_tenant

TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(TEST_DATABASE_URL,
                       connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            db.close()

    def override_get_tenant():
        return "test_tenant"

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_tenant] = override_get_tenant

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def tenant_id():
    return "test_tenant"


@pytest.fixture
def auth_headers():
    return {"x-api-key": "key-tenant-a"}

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.schema import ROOT


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def example():
    return (ROOT / "examples/20170107T1800Z.png").read_bytes()


@pytest.fixture
def array_file():
    def encode(array):
        file = io.BytesIO()
        np.save(file, array)
        return file.getvalue()

    return encode

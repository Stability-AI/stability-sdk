import grpc
import pathlib
import pytest

from concurrent import futures
from PIL import Image

from stability_sdk.api import generation_grpc

# modified from https://github.com/justdoit0823/grpc-resolver/blob/master/tests/conftest.py

_TEST_GRPC_PORT = (
    '127.0.0.1:50031', '127.0.0.1:50032', '127.0.0.1:50033', '127.0.0.1:50034')

@pytest.fixture(scope='module')
def grpc_addr():
    return _TEST_GRPC_PORT

@pytest.fixture(scope='module')
def grpc_server(grpc_addr):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = generation_grpc.GenerationServiceServicer()
    generation_grpc.add_GenerationServiceServicer_to_server(servicer, server)
    for addr in grpc_addr:
        server.add_insecure_port(addr)
    server.start()
    yield server
    server.stop(0)

@pytest.fixture(scope='module')
def impath() -> str:
    return str(next(pathlib.Path('.').glob('**/tests/assets/*.png')).resolve())

@pytest.fixture(scope='module')
def pil_image(impath) -> Image.Image:
    return Image.open(impath)

@pytest.fixture(scope='module')
def vidpath() -> str:
    return str(next(pathlib.Path('.').glob('**/tests/assets/*.mp4')))

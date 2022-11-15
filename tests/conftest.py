from concurrent import futures

import grpc
import pytest

import logging
import pathlib
import sys

thisPath = pathlib.Path(__file__).parent.parent.resolve()
genPath = thisPath / "src/stability_sdk/interfaces/gooseai/generation"
tensPath = thisPath / "src/stability_sdk/interfaces/src/tensorizer/tensors"
assert genPath.exists()
assert tensPath.exists()

logger = logging.getLogger(__name__)
sys.path.extend([str(genPath), str(tensPath)])

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

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

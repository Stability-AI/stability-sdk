# modified from https://github.com/justdoit0823/grpc-resolver/blob/master/tests/conftest.py

from concurrent import futures

import grpc
import pytest

#from grpcresolver import EtcdServiceResolver, EtcdServiceRegistry
#import hello_pb2_grpc
#import rpc

# hopefully this add anything to the PATH we might need?
#import stability_sdk

import logging
import pathlib
import sys

# uh... let's try this
# this is necessary because of how the auto-generated code constructs its imports
#thisPath = pathlib.Path(__file__).parent.resolve()
#genPath = thisPath / "interfaces/gooseai/generation"
thisPath = pathlib.Path(__file__).parent.parent.resolve()
genPath = thisPath / "api-interfaces/gooseai/generation"

logger = logging.getLogger(__name__)
#logger.setLevel(level=logging.INFO)
#print(thisPath)
#print(genPath)
logger.critical(thisPath)
logger.critical(genPath)
sys.path.append(str(genPath))

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
#import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

_TEST_GRPC_PORT = (
    '127.0.0.1:50031', '127.0.0.1:50032', '127.0.0.1:50033', '127.0.0.1:50034')


@pytest.fixture(scope='module')
def grpc_addr():
    return _TEST_GRPC_PORT

  
#class Greeter(helloworld_pb2_grpc.GreeterServicer):
#    def SayHello(self, request, unused_context):
#        return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

@pytest.fixture(scope='module')
def grpc_server(grpc_addr):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    #hello_pb2_grpc.add_HelloServicer_to_server(rpc.HelloGRpcServer(), server)
    servicer = generation_grpc.GenerationServiceServicer()
    generation_grpc.add_GenerationServiceServicer_to_server(servicer, server)
    for addr in grpc_addr:
        server.add_insecure_port(addr)

    server.start()

    yield server

    server.stop(0)

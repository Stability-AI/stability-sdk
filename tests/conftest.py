from concurrent import futures

import grpc
import numpy as np
import pytest
from PIL import Image

import logging
import pathlib
import sys
from types import SimpleNamespace

thisPath = pathlib.Path(__file__).parent.parent.resolve()
genPath = thisPath / "src/stability_sdk/interfaces/gooseai/generation"

logger = logging.getLogger(__name__)
sys.path.append(str(genPath))

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

from stability_sdk.animation import AnimationArgs

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
def impath():
    #impath = "tests\assets\4166726513_giant__rainbow_sequoia__tree_by_hayao_miyazaki___earth_tones__a_row_of_western_cedar_nurse_trees_che.png"
    return next(pathlib.Path('.').glob('**/tests/assets/*.png'))

@pytest.fixture(scope='module')
def pil_image(impath):
    return Image.open(impath)

@pytest.fixture(scope='module')
def np_image(pil_image):
    return np.array(pil_image)

@pytest.fixture(scope='module')
def vidpath():
    return str(next(pathlib.Path('.').glob('**/tests/assets/*.mp4')))

# TO DO: get defaults from arg objects
@pytest.fixture(scope='module')
def default_anim_args(vidpath):
    args = AnimationArgs()
    args.video_init_path = vidpath
    return args


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

@pytest.fixture(scope='module')
def default_anim_args():

    #@markdown ####**Settings:**
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    sampler = 'K_euler_ancestral' #@param ["DDIM", "PLMS", "K_euler", "K_euler_ancestral", "K_heun", "K_dpm_2", "K_dpm_2_ancestral", "K_lms"]
    seed = -1 #@param
    cfg_scale = 7 #@param {type:"number"}
    clip_guidance = 'FastBlue' #@param ["None", "Simple", "FastBlue", "FastGreen"]

    #@markdown ####**Animation Settings:**
    animation_mode = '3D' #@param ['2D', '3D', 'Video Input'] {type:'string'}
    max_frames = 60 #@param {type:"number"}
    border = 'replicate' #@param ['reflect', 'replicate', 'wrap', 'zero'] {type:'string'}
    inpaint_border = False #@param {type:"boolean"}
    interpolate_prompts = False #@param {type:"boolean"}
    locked_seed = False #@param {type:"boolean"}

    #@markdown ####**Key framed value curves:**
    angle = "0:(1)" #@param {type:"string"}
    zoom = "0:(1.05)" #@param {type:"string"}
    translation_x = "0:(0)" #@param {type:"string"}
    translation_y = "0:(0)" #@param {type:"string"}
    translation_z = "0:(5)" #@param {type:"string"}
    rotation_x = "0:(0)" #@param {type:"string"}
    rotation_y = "0:(0)" #@param {type:"string"}
    rotation_z = "0:(1)" #@param {type:"string"}
    brightness_curve = "0: (1.0)" #@param {type:"string"}
    contrast_curve = "0: (1.0)" #@param {type:"string"}
    noise_curve = "0:(0.0)" # likely to be removed, still hidden here for potential experiments
    noise_scale_curve = "0:(1.02)" #@param {type:"string"}
    steps_curve = "0:(50)" #@param {type:"string"}
    strength_curve = "0:(0.65)" #@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = 'LAB' #@param ['None', 'HSV', 'LAB', 'RGB'] {type:'string'}
    diffusion_cadence_curve = "0:(4)" #@param {type:"string"}

    #@markdown ####**3D Depth Warping:**
    #use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3 #@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov_curve = "0:(25)" #@param {type:"string"}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path = '/content/video_in.mp4' #@param {type:"string"}
    extract_nth_frame = 4 #@param {type:"number"}
    video_mix_in_curve = "0:(0.02)" #@param {type:"string"}
    video_flow_warp = True #@param {type:"boolean"}

    return SimpleNamespace(**locals())

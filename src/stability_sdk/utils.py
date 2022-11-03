import bisect
import copy
import io
import logging
import mimetypes
import os
import pathlib
import re
import random
import sys
import time
from typing import Dict, Generator, List, Optional, Union, Any, Sequence, Tuple
import uuid
import warnings


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

try:
    import numpy as np
    import pandas as pd
    import cv2 # to do: add this as an installation dependency
except ImportError:
    warnings.warn(
        "Failed to import animation reqs. To use the animation toolchain, install the requisite dependencies via:" 
        "   pip install --upgrade stability_sdk[anim]"
    )
    
from PIL import Image

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

SAMPLERS: Dict[str, int] = {
    "ddim": generation.SAMPLER_DDIM,
    "plms": generation.SAMPLER_DDPM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation.SAMPLER_K_LMS,
}

GUIDANCE_PRESETS: Dict[str, int] = {
        "none": generation.GUIDANCE_PRESET_NONE,
        "simple": generation.GUIDANCE_PRESET_SIMPLE,
        "fastblue": generation.GUIDANCE_PRESET_FAST_BLUE,
        "fastgreen": generation.GUIDANCE_PRESET_FAST_GREEN,
    }

COLOR_SPACES =  {
        "hsv": generation.COLOR_MATCH_HSV,
        "lab": generation.COLOR_MATCH_LAB,
        "rgb": generation.COLOR_MATCH_RGB,
    }

BORDER_MODES_2D = {
    'replicate': generation.BORDER_REPLICATE,
    'reflect': generation.BORDER_REFLECT,
    'wrap': generation.BORDER_WRAP,
    'zero': generation.BORDER_ZERO,
}

_2d_only_modes = ['wrap']
BORDER_MODES_3D = {
    k:v for k,v in BORDER_MODES_2D.items() 
    if k not in _2d_only_modes
    }

MAX_FILENAME_SZ = int(os.getenv("MAX_FILENAME_SZ", 200))

# note: we need to decide on a convention between _str and _string

def border_mode_from_str_2d(s: str) -> generation.BorderMode:
    repr = BORDER_MODES_2D.get(s.lower().strip())
    if repr is None:
        raise ValueError(f"invalid 2d border mode {s}")
    return repr

def border_mode_from_str_3d(s: str) -> generation.BorderMode:
    repr = BORDER_MODES_3D.get(s.lower().strip())
    if repr is None:
        raise ValueError(f"invalid 3d border mode {s}")
    return repr

def color_match_from_string(s: str) -> generation.ColorMatchMode:
    repr = COLOR_SPACES.get(s.lower().strip())
    if repr is None:
        raise ValueError(f"invalid color space: {s}")
    return repr

def guidance_from_string(s: str) -> generation.GuidancePreset:
    repr = GUIDANCE_PRESETS.get(s.lower().strip())
    if repr is None:
        raise ValueError(f"invalid guidance preset: {s}")
    return repr

def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.
    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    repr = SAMPLERS.get(algorithm_key)
    if repr is None:
        raise ValueError(f"invalid sampler: {s}")
    return repr

sampler_from_string = get_sampler_from_str


def truncate_fit(prefix: str, prompt: str, ext: str, ts: int, idx: int, max: int) -> str:
    """
    Constructs an output filename from a collection of required fields.
    
    Given an over-budget threshold of `max`, trims the prompt string to satisfy the budget.
    NB: As implemented, 'max' is the smallest filename length that will trigger truncation.
    It is presumed that the sum of the lengths of the other filename fields is smaller than `max`.
    If they exceed `max`, this function will just always construct a filename with no prompt component.
    """
    post = f"_{ts}_{idx}"
    prompt_budget = max
    prompt_budget -= len(prefix)
    prompt_budget -= len(post)
    prompt_budget -= len(ext) + 1
    return f"{prefix}{prompt[:prompt_budget]}{post}{ext}"


#########################

def open_images(
    images: Union[
        Sequence[Tuple[str, generation.Artifact]],
        Generator[Tuple[str, generation.Artifact], None, None],
    ],
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Open the images from the filenames and Artifacts tuples.
    :param images: The tuples of Artifacts and associated images to open.
    :return:  A Generator of tuples of image filenames and Artifacts, intended
     for passthrough.
    """
    for path, artifact in images:
        if artifact.type == generation.ARTIFACT_IMAGE:
            if verbose:
                logger.info(f"opening {path}")
            img = Image.open(io.BytesIO(artifact.binary))
            img.show()
        yield (path, artifact)


def image_mix(img_a: np.ndarray, img_b: np.ndarray, tween: float) -> np.ndarray:
    assert(img_a.shape == img_b.shape)
    return (img_a.astype(float)*(1.0-tween) + img_b.astype(float)*tween).astype(img_a.dtype)

def image_to_jpg_bytes(image: np.ndarray, quality: int=90):
    return cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()

def image_to_png_bytes(image: np.ndarray):
    return cv2.imencode('.png', image)[1].tobytes()

def pil_image_to_png_bytes(image: Image.Image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def image_to_prompt(
        image: Union[np.ndarray, Image.Image],
        is_mask=False,
    ) -> generation.Prompt:
    if isinstance(image, np.ndarray):
        image = image_to_png_bytes(image)
    elif isinstance(image, Image.Image):
        image = pil_image_to_png_bytes(image)
    else:
        print(type(image))
        raise NotImplementedError
    
    return generation.Prompt(
        parameters=generation.PromptParameters(init=not is_mask), # is this right?
        artifact=generation.Artifact(
            type=generation.ARTIFACT_MASK if is_mask else generation.ARTIFACT_IMAGE,
            binary=image))


##############################################


def key_frame_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)
    
    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
      interp_method = 'Quadratic'    
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
      interp_method = 'Linear'
          
    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

def key_frame_parse(string, prompt_parser=None):
    pattern = r'((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])'
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()['frame'])
        param = match_object.groupdict()['param']
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


#####################################################################


def image_xform(
    stub:generation_grpc.GenerationServiceStub, 
    images:List[np.ndarray], 
    ops:List[generation.TransformOperation],
    engine_id: str = 'transform-server-v1'
) -> Tuple[List[np.ndarray], np.ndarray]:
    assert(len(images))
    transforms = generation.TransformSequence(operations=ops)
    p = [image_to_prompt(image) for image in images]
    rq = generation.Request(
        engine_id=engine_id,
        prompt=p,
        image=generation.ImageParameters(transform=generation.TransformType(sequence=transforms)),
    )

    # This whole bottom portion looks like something that should be handled by "process response" or some evolved version of that
    ######################
    # there's an input above named "images", which has nothing to do with anything below this comment.
    # this is super confusing. 
    images, mask = [], None
    for resp in stub.Generate(rq, wait_for_ready=True):
        for artifact in resp.artifacts:
            if artifact.type in (generation.ARTIFACT_IMAGE, generation.ARTIFACT_MASK):
                nparr = np.frombuffer(artifact.binary, np.uint8)
                im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if artifact.type == generation.ARTIFACT_IMAGE:
                    images.append(im)
                elif artifact.type == generation.ARTIFACT_MASK:
                    if mask is not None:
                        raise Exception(
                            "multiple masks returned in response, client implementaion currently assumes no more than one mask returned"
                        )
                    mask = im
    return images, mask

#################################
# transform ops helpers
#  - move to their own submodule
#  - add doc strings giving details on parameters

# is call signature of generation.TransformWarp_d inconsistent? 
# or did I ust shuffle them here?
def warp2d_op(
    border_mode:str,
    rotate:float,
    scale:float,
    translate_x:float,
    translate_y:float,
) -> generation.TransformOperation:
    return generation.TransformOperation(
        warp2d=generation.TransformWarp2d(
            border_mode = border_mode_from_str_2d(border_mode),
            rotate = rotate,
            scale = scale,
            translate_x = translate_x,
            translate_y = translate_y,
        ))

def warp3d_op(
    border_mode:str,
    translate_x:float,
    translate_y:float,
    translate_z:float,
    rotate_x:float,
    rotate_y:float,
    rotate_z:float,
    near_plane:float,
    far_plane:float,
    fov:float, 
) -> generation.TransformOperation:
    if not (near_plane < far_plane):
        raise ValueError(
            "Invalid camera volume: must satisfy near < far, "
            f"got near={near_plane}, far={far_plane}"
        )
    if not (fov > 0):
        raise ValueError(
            "Invalid camera volume: fov must be greater than 0, "
            f"got fov={fov}"
        )
    return generation.TransformOperation(
        warp3d=generation.TransformWarp3d(
            border_mode = border_mode_from_str_3d(border_mode),
            translate_x = translate_x,
            translate_y = translate_y,
            translate_z = translate_z,
            rotate_x = rotate_x,
            rotate_y = rotate_y,
            rotate_z = rotate_z,
            near_plane = near_plane,
            far_plane = far_plane,
            fov = fov,
            ))
    
def colormatch_op(
    palette_image:np.ndarray,
    color_mode:str='LAB',
) -> generation.TransformOperation:
    im = generation.Artifact(
        type=generation.ARTIFACT_IMAGE, 
        binary=image_to_jpg_bytes(palette_image),
    )
    return generation.TransformOperation(
        color_match=generation.TransformColorMatch(
            color_mode=color_match_from_string(color_mode),
            image= im))

# why doesn't this take an image as an argument?
# pretty confident we should parameterize this to expect an ARTIFACT_IMAGE
def depthcalc_op(
    blend_weight:float,
    export:bool,
) -> generation.TransformOperation:
    return generation.TransformOperation(                    
        depth_calc=generation.TransformDepthCalc(
            blend_weight=blend_weight,
            export=export
        )
    )

def warpflow_op(
    prev_frame:np.ndarray,
    next_frame:np.ndarray,
):
    im_prev=generation.Artifact(
        type=generation.ARTIFACT_IMAGE,
        binary=image_to_jpg_bytes(prev_frame))
    im_next=generation.Artifact(
        type=generation.ARTIFACT_IMAGE,
        binary=image_to_jpg_bytes(next_frame))
    return generation.TransformOperation(
        warp_flow=generation.TransformWarpFlow(
            prev_frame=im_prev,
            next_frame=im_next,
        )
    )
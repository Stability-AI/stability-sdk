import pathlib
import sys
import os
import uuid
import random
import io
import logging
import time
from typing import Dict, Generator, List, Optional, Union, Any, Sequence, Tuple
import mimetypes


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
        "None": generation.GUIDANCE_PRESET_NONE,
        "Simple": generation.GUIDANCE_PRESET_SIMPLE,
        "FastBlue": generation.GUIDANCE_PRESET_FAST_BLUE,
        "FastGreen": generation.GUIDANCE_PRESET_FAST_GREEN,
    }

def guidance_from_string(s: str) -> generation.GuidancePreset:
    repr = GUIDANCE_PRESETS.get(s, None)
    if repr is None:
        raise ValueError("invalid guider provided")
    return repr

    
    
MAX_FILENAME_SZ = int(os.getenv("MAX_FILENAME_SZ", 200))

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

def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.
    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    algorithm = SAMPLERS.get(algorithm_key, None)
    if algorithm is None:
        raise ValueError(f"unknown sampler {s}")
    return algorithm

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

        
import numpy as np
import cv2


def image_mix(img_a: np.ndarray, img_b: np.ndarray, tween: float) -> np.ndarray:
    assert(img_a.shape == img_b.shape)
    return (img_a.astype(float)*(1.0-tween) + img_b.astype(float)*tween).astype(img_a.dtype)

def image_to_jpg_bytes(image: np.ndarray, quality: int=90):
    return cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()

def image_to_png_bytes(image: np.ndarray):
    return cv2.imencode('.png', image)[1].tobytes()

def image_to_prompt(image: np.ndarray) -> generation.Prompt:
    return generation.Prompt(
        parameters=generation.PromptParameters(init=True),
        artifact=generation.Artifact(
            type=generation.ARTIFACT_IMAGE,
            binary=image_to_png_bytes(image)))

def image_to_prompt_mask(image: np.ndarray) -> generation.Prompt:
    mask = image_to_prompt(image)
    mask.artifact.type = generation.ARTIFACT_MASK
    return mask

def image_xform(
    stub:generation_grpc.GenerationServiceStub, 
    images:List[np.ndarray], 
    ops:List[generation.TransformOperation]
) -> Tuple[List[np.ndarray], np.ndarray]:
    assert(len(images))
    transforms = generation.TransformSequence(operations=ops)
    p = [image_to_prompt(image) for image in images]
    rq = generation.Request(
        engine_id=TRANSFORM_ENGINE_ID,
        prompt=p,
        image=generation.ImageParameters(transform=generation.TransformType(sequence=transforms)),
    )

    images, mask = [], None
    for resp in stub.Generate(rq, wait_for_ready=True):
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                nparr = np.frombuffer(artifact.binary, np.uint8)
                images.append(cv2.imdecode(nparr, cv2.IMREAD_COLOR))
            elif artifact.type == generation.ARTIFACT_MASK:
                nparr = np.frombuffer(artifact.binary, np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return images, mask

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

SAMPLERS = {
        "DDIM": generation.SAMPLER_DDIM,
        "PLMS": generation.SAMPLER_DDPM,
        "K_euler": generation.SAMPLER_K_EULER,
        "K_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
        "K_heun": generation.SAMPLER_K_HEUN,
        "K_dpm_2": generation.SAMPLER_K_DPM_2,
        "K_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
        "K_lms": generation.SAMPLER_K_LMS,
    }

def sampler_from_string(str: str) -> generation.DiffusionSampler:
    repr = SAMPLERS.get(str, None)
    if not repr:
        raise ValueError("invalid sampler provided")
    return repr

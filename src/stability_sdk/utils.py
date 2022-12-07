import io
import logging
import os
import re
from typing import Dict, Generator, List, Optional, Union, Any, Sequence, Tuple
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
    "k_dpmpp_2m": generation.SAMPLER_K_DPMPP_2M,
    "k_dpmpp_2s_ancestral": generation.SAMPLER_K_DPMPP_2S_ANCESTRAL
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

INTERP_MODES = {
    'mix': generation.INTERPOLATE_LINEAR,
    'rife': generation.INTERPOLATE_RIFE,
    'vae-lerp': generation.INTERPOLATE_VAE_LINEAR,
    'vae-slerp': generation.INTERPOLATE_VAE_SLERP,
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

def interp_mode_from_str(s: str) -> generation.InterpolateMode:
    mode = INTERP_MODES.get(s.lower().strip())
    if mode is None:
        raise ValueError(f"invalid interpolation mode: {s}")
    return mode

def artifact_type_to_str(artifact_type: generation.ArtifactType):
    """
    Convert ArtifactType to a string.
    :param artifact_type: The ArtifactType to convert.
    :return: String representation of the ArtifactType.
    """
    try:
        return generation.ArtifactType.Name(artifact_type)
    except ValueError:
        logging.warning(
            f"Received artifact of type {artifact_type}, which is not recognized in the loaded protobuf definition.\n"
            "If you are seeing this message, you might be using an old version of the client library. Please update your client via `pip install --upgrade stability-sdk`\n"
            "If updating the client does not make this warning message go away, please report this behavior to https://github.com/Stability-AI/stability-sdk/issues/new"
        )
        return "ARTIFACT_UNRECOGNIZED"

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


def image_mix(img_a: np.ndarray, img_b: np.ndarray, ratio: Union[float, np.ndarray]) -> np.ndarray:
    """
    Performs a linear interpolation between two images
    :param img_a: The first image.
    :param img_b: The second image.
    :param ratio: A float (or ndarray of per-pixel floats) for in-between ratio
    :return: The mixed image
    """
    if img_a.shape != img_b.shape:
        raise ValueError(f"img_a shape {ratio.shape} does not match img_b shape {img_a.shape}")

    if isinstance(ratio, np.ndarray):
        if ratio.shape[:2] != img_a.shape[:2]:
            raise ValueError(f"tween dimensions {ratio.shape[:2]} do not match image dimensions {img_a.shape[:2]}")
        if ratio.dtype == np.uint8:
            ratio = ratio.astype(np.float32) / 255.0
        if len(ratio.shape) == 2:
            ratio = np.repeat(ratio[:,:,None], 3, axis=2)
        
    return (img_a.astype(np.float32)*(1.0-ratio) + img_b.astype(np.float32)*ratio).astype(img_a.dtype)

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
    is_mask: bool = False
) -> generation.Prompt:
    if isinstance(image, np.ndarray):
        image = image_to_png_bytes(image)
    elif isinstance(image, Image.Image):
        image = pil_image_to_png_bytes(image)
    else:
        print(type(image))
        raise NotImplementedError
    
    return generation.Prompt(
        parameters=generation.PromptParameters(init=not is_mask),
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

# to do: defaults. None?
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

def depthcalc_op(
    blend_weight:float,
    export:bool = False,
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
) -> generation.TransformOperation:
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

def blend_op(
    amount:float,
    target:np.ndarray,
) -> generation.TransformOperation:
    im=generation.Artifact(
        type=generation.ARTIFACT_IMAGE,
        binary=image_to_jpg_bytes(target),
    )
    return generation.TransformOperation(
        blend=generation.TransformBlend(
            amount=amount, 
            target=im,
        ))

def contrast_op(
    brightness: float,
    contrast: float,
) -> generation.TransformOperation:
    return generation.TransformOperation(
        contrast=generation.TransformContrast(
            brightness=brightness,
            contrast=contrast,
        ))

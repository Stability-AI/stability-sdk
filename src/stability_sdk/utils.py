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
from stability_sdk.matrix import Matrix


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

BORDER_MODES = {
    'replicate': generation.BORDER_REPLICATE,
    'reflect': generation.BORDER_REFLECT,
    'wrap': generation.BORDER_WRAP,
    'zero': generation.BORDER_ZERO,
    'prefill': generation.BORDER_PREFILL,
}

INTERP_MODES = {
    'mix': generation.INTERPOLATE_LINEAR,
    'rife': generation.INTERPOLATE_RIFE,
    'vae-lerp': generation.INTERPOLATE_VAE_LINEAR,
    'vae-slerp': generation.INTERPOLATE_VAE_SLERP,
}

CAMERA_TYPES = {
    'perspective': generation.CAMERA_PERSPECTIVE,
    'orthographic': generation.CAMERA_ORTHOGRAPHIC,
}

MAX_FILENAME_SZ = int(os.getenv("MAX_FILENAME_SZ", 200))


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a cv2 BGR ndarray"""
    return np.array(pil_img)[:, :, ::-1]

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert a cv2 BGR ndarray to a PIL Image"""
    return Image.fromarray(cv2_img[:, :, ::-1])


# note: we need to decide on a convention between _str and _string

def border_mode_from_str(s: str) -> generation.BorderMode:
    repr = BORDER_MODES.get(s.lower().strip())
    if repr is None:
        raise ValueError(f"invalid border mode: {s}")
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

# TODO: class/decorator instead
def camera_type_from_string(s: str) -> generation.CameraType:
    cam_type = CAMERA_TYPES.get(s.lower().strip())
    if cam_type is None:
        raise ValueError(f"invalid camera type: {s}")
    return cam_type

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
        raise ValueError(f"img_a shape {img_a.shape} does not match img_b shape {img_b.shape}")

    if isinstance(ratio, np.ndarray):
        if ratio.shape[:2] != img_a.shape[:2]:
            raise ValueError(f"tween dimensions {ratio.shape[:2]} do not match image dimensions {img_a.shape[:2]}")
        if ratio.dtype == np.uint8:
            ratio = ratio.astype(np.float32) / 255.0
        if len(ratio.shape) == 2:
            ratio = np.repeat(ratio[:,:,None], 3, axis=2)
        
    return (img_a.astype(np.float32)*(1.0-ratio) + img_b.astype(np.float32)*ratio).astype(img_a.dtype)

def image_to_jpg_bytes(image: Union[Image.Image, np.ndarray], quality: int=90) -> bytes:
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return buf.getvalue()
    elif isinstance(image, np.ndarray):        
        return cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1].tobytes()
    else:
        raise TypeError(f"Expected image to be a PIL.Image.Image or numpy.ndarray, got {type(image)}")

def image_to_png_bytes(image: Union[Image.Image, np.ndarray]) -> bytes:
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    elif isinstance(image, np.ndarray):
        return cv2.imencode('.png', image)[1].tobytes()
    else:
        raise TypeError(f"Expected image to be a PIL.Image.Image or numpy.ndarray, got {type(image)}")

def image_to_prompt(
    image: Union[Image.Image, np.ndarray],
    type: generation.ArtifactType=generation.ARTIFACT_IMAGE
) -> generation.Prompt:
    png = image_to_png_bytes(image)    
    return generation.Prompt(
        parameters=generation.PromptParameters(init=True),
        artifact=generation.Artifact(type=type, binary=png)
    )


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

def blend_op(
    amount:float,
    target:np.ndarray,
) -> generation.TransformParameters:
    return generation.TransformParameters(
        blend=generation.TransformBlend(
            amount=amount, 
            target=generation.Artifact(
                type=generation.ARTIFACT_IMAGE,
                binary=image_to_jpg_bytes(target),
            )
        )
    )

def color_adjust_op(
    brightness:float=1.0,
    contrast:float=1.0,
    hue:float=0.0,
    saturation:float=1.0,
    lightness:float=0.0,
    match_image:Optional[np.ndarray]=None,
    match_mode:str='LAB',
    noise_amount:float=0.0,
    noise_seed:int=0
) -> generation.TransformParameters:
    if match_mode == 'None':
        match_mode = 'RGB'
        match_image = None
    return generation.TransformParameters(
        color_adjust=generation.TransformColorAdjust(
            brightness=brightness,
            contrast=contrast,
            hue=hue,
            saturation=saturation,
            lightness=lightness,
            match_image=generation.Artifact(
                type=generation.ARTIFACT_IMAGE,
                binary=image_to_jpg_bytes(match_image),
            ) if match_image is not None else None,
            match_mode=color_match_from_string(match_mode),
            noise_amount=noise_amount,
            noise_seed=noise_seed,
        ))

def depthcalc_op(
    blend_weight:float,
    blur_radius:int=0,
    reverse:bool=False,
) -> generation.TransformParameters:
    return generation.TransformParameters(
        depth_calc=generation.TransformDepthCalc(
            blend_weight=blend_weight,
            blur_radius=blur_radius,
            reverse=reverse
        )
    )

def resample_op(
    border_mode:str,
    transform:Matrix,
    prev_transform:Optional[Matrix]=None,
    depth_warp:float=1.0,
    export_mask:bool=False
) -> generation.TransformParameters:
    return generation.TransformParameters(
        resample=generation.TransformResample(
            border_mode=border_mode_from_str(border_mode),
            transform=generation.TransformMatrix(data=sum(transform, [])),
            prev_transform=generation.TransformMatrix(data=sum(prev_transform, [])) if prev_transform else None,
            depth_warp=depth_warp,
            export_mask=export_mask
        )
    )

def camera_pose_op(
    transform:Matrix,
    near_plane:float,
    far_plane:float,
    fov:float,
    camera_type:str='perspective',
    image_render_method:str='mesh',
    image_point_radius:Optional[float]=None,
    image_points_per_pixel:Optional[int]=None,
    image_max_mesh_edge:Optional[float]=0.04,
    mask_render_method:str='pointcloud',
    mask_point_radius:Optional[float]=0.003,
    mask_points_per_pixel:Optional[int]=4,
    mask_max_mesh_edge:Optional[float]=None,
    do_prefill:bool=True,
) -> generation.TransformParameters:
    camera_parameters = generation.CameraParameters(
        camera_type=camera_type_from_string(camera_type),
        near_plane=near_plane, far_plane=far_plane, fov=fov)
    if image_render_method == "pointcloud":
        image_render_parameters = generation.RenderParameters(
            pointcloud_parameters=generation.PointCloudRenderParameters(
                radius=image_point_radius, points_per_pixel=image_points_per_pixel))
    elif image_render_method == "mesh":
        image_render_parameters = generation.RenderParameters(
            mesh_parameters=generation.MeshRenderParameters(
                max_mesh_edge=image_max_mesh_edge))
    else:
        raise Exception("Rendering method must be one of 'pointcloud' or 'mesh'")
    if mask_render_method == "pointcloud":
        mask_render_parameters = generation.RenderParameters(
            pointcloud_parameters=generation.PointCloudRenderParameters(
                radius=mask_point_radius, points_per_pixel=mask_points_per_pixel))
    elif mask_render_method == "mesh":
        mask_render_parameters = generation.RenderParameters(
            mesh_parameters=generation.MeshRenderParameters(
                max_mesh_edge=mask_max_mesh_edge))
    else:
        raise Exception("Rendering method must be one of 'pointcloud' or 'mesh'")
    return generation.TransformParameters(
        camera_pose=generation.TransformCameraPose(
            world_to_view_matrix=generation.TransformMatrix(data=sum(transform, [])),
            camera_parameters=camera_parameters,
            image_render_parameters=image_render_parameters,
            mask_render_parameters=mask_render_parameters,
            do_prefill=do_prefill
        )
    )

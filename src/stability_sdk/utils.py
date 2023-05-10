import io
import logging
import os
import subprocess

from PIL import Image
from typing import Dict, Generator, Optional, Sequence, Tuple, Type, TypeVar, Union

from .api import generation
from .matrix import Matrix


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

MAX_FILENAME_SZ = int(os.getenv("MAX_FILENAME_SZ", 200))


#==============================================================================
# Mappings from strings to protobuf enums
#==============================================================================

BORDER_MODES = {
    'replicate': generation.BORDER_REPLICATE,
    'reflect': generation.BORDER_REFLECT,
    'wrap': generation.BORDER_WRAP,
    'zero': generation.BORDER_ZERO,
    'prefill': generation.BORDER_PREFILL,
}

CAMERA_TYPES = {
    'perspective': generation.CAMERA_PERSPECTIVE,
    'orthographic': generation.CAMERA_ORTHOGRAPHIC,
}

COLOR_MATCH_MODES = {
    "hsv": generation.COLOR_MATCH_HSV,
    "lab": generation.COLOR_MATCH_LAB,
    "rgb": generation.COLOR_MATCH_RGB,
}

GUIDANCE_PRESETS: Dict[str, int] = {
    "none": generation.GUIDANCE_PRESET_NONE,
    "simple": generation.GUIDANCE_PRESET_SIMPLE,
    "fastblue": generation.GUIDANCE_PRESET_FAST_BLUE,
    "fastgreen": generation.GUIDANCE_PRESET_FAST_GREEN,
}

INTERPOLATE_MODES = {
    'film': generation.INTERPOLATE_FILM,
    'mix': generation.INTERPOLATE_LINEAR,
    'rife': generation.INTERPOLATE_RIFE,
    'vae-lerp': generation.INTERPOLATE_VAE_LINEAR,
    'vae-slerp': generation.INTERPOLATE_VAE_SLERP,
}

RENDER_MODES = {
    'mesh': generation.RENDER_MESH,
    'pointcloud': generation.RENDER_POINTCLOUD,
}

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

T = TypeVar('T')

def _from_string(s: str, mapping: Dict[str, T], name: str, enum_cls: Type[T]) -> T:
    enum_value = mapping.get(s.lower().strip())
    if enum_value is None:
        raise ValueError(f"invalid {name}: {s}")
    return enum_value

def border_mode_from_string(s: str) -> generation.BorderMode:
    return _from_string(s, BORDER_MODES, "border mode", generation.BorderMode)

def camera_type_from_string(s: str) -> generation.CameraType:
    return _from_string(s, CAMERA_TYPES, "camera type", generation.CameraType)

def color_match_from_string(s: str) -> generation.ColorMatchMode:
    return _from_string(s, COLOR_MATCH_MODES, "color match", generation.ColorMatchMode)

def guidance_from_string(s: str) -> generation.GuidancePreset:
    return _from_string(s, GUIDANCE_PRESETS, "guidance preset", generation.GuidancePreset)

def interpolate_mode_from_string(s: str) -> generation.InterpolateMode:
    return _from_string(s, INTERPOLATE_MODES, "interpolate mode", generation.InterpolateMode)

def render_mode_from_string(s: str) -> generation.RenderMode:
    return _from_string(s, RENDER_MODES, "render mode", generation.RenderMode)

def sampler_from_string(s: str) -> generation.DiffusionSampler:
    return _from_string(s, SAMPLERS, "sampler", generation.DiffusionSampler)


#==============================================================================
# Transform helper functions
#==============================================================================

def camera_pose_transform(
    transform: Matrix,
    near_plane: float,
    far_plane: float,
    fov: float,
    camera_type: str='perspective',
    render_mode: str='mesh',
    do_prefill: bool=True,
) -> generation.TransformParameters:
    camera_parameters = generation.CameraParameters(
        camera_type=camera_type_from_string(camera_type),
        near_plane=near_plane, far_plane=far_plane, fov=fov)
    return generation.TransformParameters(
        camera_pose=generation.TransformCameraPose(
            world_to_view_matrix=generation.TransformMatrix(data=sum(transform, [])),
            camera_parameters=camera_parameters,
            render_mode=render_mode_from_string(render_mode),
            do_prefill=do_prefill
        )
    )

def color_adjust_transform(
    brightness: float=1.0,
    contrast: float=1.0,
    hue: float=0.0,
    saturation: float=1.0,
    lightness: float=0.0,
    match_image: Optional[Image.Image]=None,
    match_mode: str='LAB',
    noise_amount: float=0.0,
    noise_seed: int=0
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

def depth_calc_transform(
    blend_weight: float,
    blur_radius: int=0,
    reverse: bool=False,
) -> generation.TransformParameters:
    return generation.TransformParameters(
        depth_calc=generation.TransformDepthCalc(
            blend_weight=blend_weight,
            blur_radius=blur_radius,
            reverse=reverse
        )
    )

def resample_transform(
    border_mode: str,
    transform: Matrix,
    prev_transform: Optional[Matrix]=None,
    depth_warp: float=1.0,
    export_mask: bool=False
) -> generation.TransformParameters:
    return generation.TransformParameters(
        resample=generation.TransformResample(
            border_mode=border_mode_from_string(border_mode),
            transform=generation.TransformMatrix(data=sum(transform, [])),
            prev_transform=generation.TransformMatrix(data=sum(prev_transform, [])) if prev_transform else None,
            depth_warp=depth_warp,
            export_mask=export_mask
        )
    )


#==============================================================================
# General utility functions
#==============================================================================

def artifact_type_to_string(artifact_type: generation.ArtifactType):
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

def create_video_from_frames(frames_path: str, mp4_path: str, fps: int=24, reverse: bool=False):
    """
    Convert a series of image frames to a video file using ffmpeg.

    :param frames_path: The path to the directory containing the image frames named frame_00000.png, frame_00001.png, etc.
    :param mp4_path: The path to save the output video file.
    :param fps: The frames per second for the output video. Default is 24.
    :param reverse: A flag to reverse the order of the frames in the output video. Default is False.
    """

    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', os.path.join(frames_path, "frame_%05d.png"),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryslow',
        mp4_path
    ]
    if reverse:
        cmd.insert(-1, '-vf')
        cmd.insert(-1, 'reverse')    

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr)

def extract_frames_from_video(video_path: str, frames_subdir: str='frames'):
    """
    Extracts all frames from a video to a subdirectory of the video's parent folder.
    :param video_path: A path to the video.
    :param frames_subdir: Name of the subdirectory to save the frames into.
    :return: The frames subdirectory path.
    """
    out_dir = os.path.join(os.path.dirname(video_path), frames_subdir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    cmd = [
        'ffmpeg',
        '-i', video_path,
        os.path.join(out_dir, "frame_%05d.png"),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr)

    return out_dir

def image_mix(img_a: Image.Image, img_b: Image.Image, ratio: Union[float, Image.Image]) -> Image.Image:
    """
    Performs a linear interpolation between two images
    :param img_a: The first image.
    :param img_b: The second image.
    :param ratio: Mix ratio or mask image.
    :return: The mixed image
    """
    if img_a.size != img_b.size:
        raise ValueError(f"img_a size {img_a.size} does not match img_b size {img_b.size}")
    
    if isinstance(ratio, Image.Image):
        if ratio.size != img_a.size:
            raise ValueError(f"mix ratio size {ratio.size} does not match img_a size {img_a.size}")
        return Image.composite(img_b, img_a, ratio)

    return Image.blend(img_a, img_b, ratio)

def image_to_jpg_bytes(image: Image.Image, quality: int=90) -> bytes:
    """
    Compresses an image to a JPEG byte array.
    :param image: The image to convert.
    :param quality: The JPEG quality to use.
    :return: The JPEG byte array.
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.getvalue()

def image_to_png_bytes(image: Image.Image) -> bytes:
    """
    Compresses an image to a PNG byte array.
    :param image: The image to convert.
    :return: The PNG byte array.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def image_to_prompt(
    image: Image.Image,
    type: generation.ArtifactType=generation.ARTIFACT_IMAGE
) -> generation.Prompt:
    """
    Create Prompt message type from an image.
    :param image: The image.
    :param type: The ArtifactType to use (ARTIFACT_IMAGE, ARTIFACT_MASK, or ARTIFACT_DEPTH).
    """
    return generation.Prompt(artifact=generation.Artifact(
        type=type, 
        binary=image_to_png_bytes(image)
    ))

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

def tensor_to_prompt(tensor: 'tensors_pb.Tensor') -> generation.Prompt:
    """
    Create Prompt message type from a tensor.
    :param tensor: The tensor.    
    """
    return generation.Prompt(artifact=generation.Artifact(
        type=generation.ARTIFACT_TENSOR, 
        tensor=tensor
    ))

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

import base64
import bisect
import cv2
import glob
import json
import logging
import math
import numpy as np
import os
import param
import random
import shutil

from collections import OrderedDict, deque
from dataclasses import dataclass, fields
from keyframed.dsl import curve_from_cn_string
from PIL import Image, ImageOps
from types import SimpleNamespace
from typing import Callable, cast, Deque, Dict, Generator, List, Optional, Tuple, Union

from stability_sdk.api import Context, generation
from stability_sdk.utils import (
    camera_pose_transform,
    color_adjust_transform,
    depth_calc_transform,
    guidance_from_string,
    image_mix,
    image_to_png_bytes,
    interpolate_mode_from_string,
    resample_transform,
    sampler_from_string,
)
import stability_sdk.matrix as matrix

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

DEFAULT_MODEL = 'stable-diffusion-xl-1024-v0-9'
TRANSLATION_SCALE = 1.0/200.0 # matches Disco and Deforum

docstring_bordermode = ( 
    "Method that will be used to fill empty regions, e.g. after a rotation transform."
    "\n\t* reflect - Mirror pixels across the image edge to fill empty regions."
    "\n\t* replicate - Use closest pixel values (default)."
    "\n\t* wrap - Treat image borders as if they were connected, i.e. use pixels from left edge to fill empty regions touching the right edge."
    "\n\t* zero - Fill empty regions with black pixels."
    "\n\t* prefill - Do simple inpainting over empty regions."
)

class BasicSettings(param.Parameterized):
    width = param.Integer(default=512, doc="Output image dimensions. Will be resized to a multiple of 64.")
    height = param.Integer(default=512, doc="Output image dimensions. Will be resized to a multiple of 64.")
    sampler = param.Selector(
        default='K_dpmpp_2m', 
        objects=[
            "DDIM", "PLMS", "K_euler", "K_euler_ancestral", "K_heun", "K_dpm_2", 
            "K_dpm_2_ancestral", "K_lms", "K_dpmpp_2m", "K_dpmpp_2s_ancestral"
        ]
    )
    model = param.Selector(
        default=DEFAULT_MODEL, 
        check_on_set=False, # allow old and new models without raising ValueError
        objects=[
            "stable-diffusion-512-v2-1", "stable-diffusion-xl-beta-v2-2-2", "stable-diffusion-xl-1024-v0-9", 
            "stable-diffusion-xl-1024-v1-0", "custom"
        ]
    )
    custom_model = param.String(default="", doc="Identifier of custom model to use.")
    seed = param.Integer(default=-1, doc="Provide a seed value for more deterministic behavior. Negative seed values will be replaced with a random seed (default).")
    cfg_scale = param.Number(default=7, softbounds=(0,20), doc="Classifier-free guidance scale. Strength of prompt influence on denoising process. `cfg_scale=0` gives unconditioned sampling.")
    clip_guidance = param.Selector(default='None', objects=["None", "Simple", "FastBlue", "FastGreen"], doc="CLIP-guidance preset.")
    init_image = param.String(default='', doc="Path to image. Height and width dimensions will be inherited from image.")
    init_sizing = param.Selector(default='stretch', objects=["cover", "stretch", "resize-canvas"])
    mask_path = param.String(default="", doc="Path to image or video mask")
    mask_invert = param.Boolean(default=False, doc="White in mask marks areas to change by default.")
    preset = param.Selector(
        default='None', 
        objects=[
            'None', '3d-model', 'analog-film', 'anime', 'cinematic', 'comic-book', 'digital-art', 
            'enhance', 'fantasy-art', 'isometric', 'line-art', 'low-poly', 'modeling-compound', 
            'neon-punk', 'origami', 'photographic', 'pixel-art',
        ]
    )

class AnimationSettings(param.Parameterized):
    animation_mode = param.Selector(default='3D warp', objects=['2D', '3D warp', '3D render', 'Video Input'])
    max_frames = param.Integer(default=72, doc="Force stop of animation job after this many frames are generated.")
    border = param.Selector(default='replicate', objects=['reflect', 'replicate', 'wrap', 'zero', 'prefill'], doc=docstring_bordermode)
    noise_add_curve = param.String(default="0:(0.02)")
    noise_scale_curve = param.String(default="0:(0.99)")
    strength_curve = param.String(default="0:(0.65)", doc="Image Strength (of init image relative to the prompt). 0 for ignore init image and attend only to prompt, 1 would return the init image unmodified")
    steps_curve = param.String(default="0:(30)", doc="Diffusion steps")
    steps_strength_adj = param.Boolean(default=False, doc="Adjusts number of diffusion steps based on current previous frame strength value.")    
    interpolate_prompts = param.Boolean(default=False, doc="Smoothly interpolate prompts between keyframes. Defaults to False")
    locked_seed = param.Boolean(default=False)

class CameraSettings(param.Parameterized):
    """
    See disco/deforum keyframing syntax, originally developed by Chigozie Nri
    General syntax: "<frameId>:(<valueAtFrame>), f2:(v2),f3:(v3)...." 
    Values between intermediate keyframes will be linearly interpolated by default to produce smooth transitions.
    For abrupt transitions, specify values at adjacent keyframes.
    """
    angle = param.String(default="0:(0)", doc="Camera rotation angle in degrees for 2D mode")
    zoom = param.String(default="0:(1)", doc="Camera zoom factor for 2D mode (<1 zooms out, >1 zooms in)")
    translation_x = param.String(default="0:(0)")
    translation_y = param.String(default="0:(0)")
    translation_z = param.String(default="0:(0)")
    rotation_x = param.String(default="0:(0)", doc="Camera rotation around X-axis in degrees for 3D modes")
    rotation_y = param.String(default="0:(0)", doc="Camera rotation around Y-axis in degrees for 3D modes")
    rotation_z = param.String(default="0:(0)", doc="Camera rotation around Z-axis in degrees for 3D modes")


class CoherenceSettings(param.Parameterized):
    diffusion_cadence_curve = param.String(default="0:(1)", doc="One greater than the number of frames between diffusion operations. A cadence of 1 performs diffusion on each frame. Values greater than one will generate frames using interpolation methods.")
    cadence_interp = param.Selector(default='mix', objects=['film', 'mix', 'rife', 'vae-lerp', 'vae-slerp'])
    cadence_spans = param.Boolean(default=False, doc="Experimental diffusion cadence mode for better outpainting")


class ColorSettings(param.Parameterized):
    color_coherence = param.Selector(default='LAB', objects=['None', 'HSV', 'LAB', 'RGB'], doc="Color space that will be used for inter-frame color adjustments.")
    brightness_curve = param.String(default="0:(1.0)")
    contrast_curve = param.String(default="0:(1.0)")
    hue_curve = param.String(default="0:(0.0)")
    saturation_curve = param.String(default="0:(1.0)")
    lightness_curve = param.String(default="0:(0.0)")
    color_match_animate = param.Boolean(default=True, doc="Animate color match between key frames.")


class DepthSettings(param.Parameterized):
    depth_model_weight = param.Number(default=0.3, softbounds=(0,1), doc="Blend factor between AdaBins and MiDaS depth models.")
    near_plane = param.Number(default=200, doc="Distance to nearest plane of camera view volume.")
    far_plane = param.Number(default=10000, doc="Distance to furthest plane of camera view volume.")
    fov_curve = param.String(default="0:(25)", doc="FOV angle of camera volume in degrees.")
    depth_blur_curve = param.String(default="0:(0.0)", doc="Blur strength of depth map.")
    depth_warp_curve = param.String(default="0:(1.0)", doc="Depth warp strength.")
    save_depth_maps = param.Boolean(default=False)
    

class Rendering3dSettings(param.Parameterized):
    camera_type = param.Selector(default='perspective', objects=['perspective', 'orthographic'])
    render_mode = param.Selector(default='mesh', objects=['mesh', 'pointcloud'], doc="Mode for image and mask rendering. 'pointcloud' is a bit faster, but 'mesh' is more stable")
    mask_power = param.Number(default=0.3, softbounds=(0, 4), doc="Raises each mask (0, 1) value to this power. The higher the value the more changes will be applied to the nearest objects")
    
class InpaintingSettings(param.Parameterized):
    use_inpainting_model = param.Boolean(default=False, doc="If True, inpainting will be performed using dedicated inpainting model. If False, inpainting will be performed with the regular model that is selected")
    inpaint_border = param.Boolean(default=False, doc="Use inpainting on top of border regions for 2D and 3D warp modes. Defaults to False")
    mask_min_value = param.String(default="0:(0.25)", doc="Mask postprocessing for non-inpainting model. Mask floor values will be clipped by this value prior to inpainting")
    mask_binarization_thr = param.Number(default=0.5, softbounds=(0,1), doc="Grayscale mask values lower than this value will be set to 0, values that are higher — to 1.")
    save_inpaint_masks = param.Boolean(default=False)

class VideoInputSettings(param.Parameterized):
    video_init_path = param.String(default="", doc="Path to video input")
    extract_nth_frame = param.Integer(default=1, bounds=(1,None), doc="Only use every Nth frame of the video")
    video_mix_in_curve = param.String(default="0:(0.02)")
    video_flow_warp = param.Boolean(default=True, doc="Whether or not to transfer the optical flow from the video to the generated animation as a warp effect.")

class VideoOutputSettings(param.Parameterized):
    fps = param.Integer(default=12, doc="Frame rate to use when generating video output.")
    reverse = param.Boolean(default=False, doc="Whether to reverse the output video or not.")

class AnimationArgs(
    BasicSettings,
    AnimationSettings,
    CameraSettings,
    CoherenceSettings,
    ColorSettings,
    DepthSettings,
    Rendering3dSettings,
    InpaintingSettings,
    VideoInputSettings,
    VideoOutputSettings
):
    """
    Aggregates parameters from the multiple settings classes.
    """

@dataclass
class FrameArgs:
    """Expansion of key framed Args to per-frame values"""
    angle: List[float]
    zoom: List[float]
    translation_x: List[float]
    translation_y: List[float]
    translation_z: List[float]
    rotation_x: List[float]
    rotation_y: List[float]
    rotation_z: List[float]
    brightness_curve: List[float]
    contrast_curve: List[float]
    hue_curve: List[float]
    saturation_curve: List[float]
    lightness_curve: List[float]
    noise_add_curve: List[float]
    noise_scale_curve: List[float]
    steps_curve: List[float]
    strength_curve: List[float]
    diffusion_cadence_curve: List[float]
    fov_curve: List[float]
    depth_blur_curve: List[float]
    depth_warp_curve: List[float]
    video_mix_in_curve: List[float]
    mask_min_value: List[float]


def args_to_dict(args):
    """
    Converts arguments object to an OrderedDict
    """
    if isinstance(args, param.Parameterized):
        return OrderedDict(args.param.values())
    elif isinstance(args, SimpleNamespace):
        return OrderedDict(vars(args))
    else:
        raise NotImplementedError(f"Unsupported arguments object type: {type(args)}")

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Convert a cv2 BGR ndarray to a PIL Image"""
    return Image.fromarray(cv2_img[:, :, ::-1])

def interpolate_frames(
    context: Context, 
    frames_path: str, 
    out_path: str, 
    interp_mode: generation.InterpolateMode, 
    interp_factor: int
) -> Generator[Image.Image, None, None]:
    """Interpolates frames in a directory using the specified interpolation mode."""
    assert interp_factor > 1, "Interpolation factor must be greater than 1"

    # gather source frames
    frame_files = glob.glob(os.path.join(frames_path, "frame_*.png"))
    frame_files.sort()

    # perform frame interpolation
    os.makedirs(out_path, exist_ok=True)
    ratios = np.linspace(0, 1, interp_factor+1)[1:-1].tolist()
    for i in range(len(frame_files) - 1):
        shutil.copy(frame_files[i], os.path.join(out_path, f"frame_{i * interp_factor:05d}.png"))
        frame1 = Image.open(frame_files[i])
        frame2 = Image.open(frame_files[i + 1])
        yield frame1
        tweens = context.interpolate([frame1, frame2], ratios, interp_mode)
        for ti, tween in enumerate(tweens):
            tween.save(os.path.join(out_path, f"frame_{i * interp_factor + ti + 1:05d}.png"))
            yield tween

    # copy final frame
    shutil.copy(frame_files[-1], os.path.join(out_path, f"frame_{(len(frame_files)-1) * interp_factor:05d}.png"))        

def mask_erode_blur(mask: Image.Image, mask_erode: int, mask_blur: int) -> Image.Image:
    mask = np.array(mask)
    if mask_erode > 0:
        ks = mask_erode*2 + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)), iterations=1)
    if mask_blur > 0:
        ks = mask_blur*2 + 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
    return Image.fromarray(mask)

def make_xform_2d(
    w: float, h: float,
    rotation_angle: float, # in radians
    scale_factor: float,
    translate_x: float,
    translate_y: float,
) -> matrix.Matrix:
    center = (w / 2, h / 2)
    pre = matrix.translation(-center[0], -center[1], 0)
    post = matrix.translation(center[0], center[1], 0)
    rotate = matrix.rotation_euler(0, 0, rotation_angle)
    scale = matrix.scale(scale_factor, scale_factor, 1)
    rotate_scale = matrix.multiply(post, matrix.multiply(rotate, matrix.multiply(scale, pre)))
    # match 3D camera translation, +X moves camera to right, +Y moves camera up
    translate = matrix.translation(-translate_x, translate_y, 0)
    return matrix.multiply(rotate_scale, translate)

def model_supports_clip_guidance(model_name: str) -> bool:
    return not model_name.startswith('stable-diffusion-xl')

def model_requires_depth(model_name: str) -> bool:
    return model_name == 'stable-diffusion-depth-v2-0'

def sampler_supports_clip_guidance(sampler_name: str) -> bool:
    supported_samplers = [
        generation.SAMPLER_K_EULER_ANCESTRAL,
        generation.SAMPLER_K_DPM_2_ANCESTRAL,
        generation.SAMPLER_K_DPMPP_2S_ANCESTRAL
    ]
    return sampler_from_string(sampler_name) in supported_samplers

def to_3x3(m: matrix.Matrix) -> matrix.Matrix:
    # convert 4x4 matrix with 2D rotation, scale, and translation to 3x3 matrix
    return [[m[0][0], m[0][1], m[0][3]],
            [m[1][0], m[1][1], m[1][3]],
            [m[3][0], m[3][1], m[3][3]]]


class Animator:
    def __init__(
        self,
        api_context: Context,
        animation_prompts: dict,
        args: Optional[AnimationArgs] = None,
        out_dir: Optional[str] = None,
        negative_prompt: str = '',
        negative_prompt_weight: float = -1.0,
        resume: bool = False
    ):
        self.api = api_context
        self.animation_prompts = animation_prompts
        self.args = args or AnimationArgs()
        self.color_match_images: Optional[Dict[int, Image.Image]] = {}
        self.diffusion_cadence_ofs: int = 0
        self.frame_args: FrameArgs
        self.inpaint_mask: Optional[Image.Image] = None
        self.key_frame_values: List[int] = []
        self.out_dir: Optional[str] = out_dir
        self.mask: Optional[Image.Image] = None
        self.mask_reader = None
        self.cadence_on: bool = False
        self.prior_frames: Deque[Image.Image] = deque([], 1)    # forward warped prior frames. stores one image with cadence off, two images otherwise
        self.prior_diffused: Deque[Image.Image] = deque([], 1)  # results of diffusion. stores one image with cadence off, two images otherwise
        self.prior_xforms: Deque[matrix.Matrix] = deque([], 1)   # accumulated transforms since last diffusion. stores one with cadence off, two otherwise
        self.negative_prompt: str = negative_prompt
        self.negative_prompt_weight: float = negative_prompt_weight
        self.start_frame_idx: int = 0
        self.video_prev_frame: Optional[Image.Image] = None
        self.video_reader: Optional[cv2.VideoCapture] = None

        # configure Api to retry on classifier obfuscations
        self.api._retry_obfuscation = True

        # two stage 1024 model requires longer timeout
        if self.args.model.startswith('stable-diffusion-xl-1024'):
            self.api._request_timeout = 120.0

        # create output directory
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        elif self.args.save_depth_maps or self.args.save_inpaint_masks:
            raise ValueError('out_dir must be specified when saving depth maps or inpaint masks')

        self.setup_animation(resume)

    def build_frame_xform(self, frame_idx) -> matrix.Matrix:
        args, frame_args = self.args, self.frame_args

        if self.args.animation_mode == '2D':
            angle = frame_args.angle[frame_idx]
            scale = frame_args.zoom[frame_idx]
            dx = frame_args.translation_x[frame_idx]
            dy = frame_args.translation_y[frame_idx]
            return make_xform_2d(args.width, args.height, math.radians(angle), scale, dx, dy)

        elif self.args.animation_mode in ('3D warp', '3D render'):
            dx = frame_args.translation_x[frame_idx]
            dy = frame_args.translation_y[frame_idx]
            dz = frame_args.translation_z[frame_idx]
            rx = frame_args.rotation_x[frame_idx]
            ry = frame_args.rotation_y[frame_idx]
            rz = frame_args.rotation_z[frame_idx]

            dx, dy, dz = -dx*TRANSLATION_SCALE, dy*TRANSLATION_SCALE, -dz*TRANSLATION_SCALE
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)

            # create xform for the current frame
            world_view = matrix.multiply(matrix.translation(dx, dy, dz), matrix.rotation_euler(rx, ry, rz))
            return world_view

        else:
            return matrix.identity

    def emit_frame(self, frame_idx: int, out_frame: Image.Image) -> Image.Image:
        if self.args.save_depth_maps:
            depth_image = self.generate_depth_image(out_frame)
            self.save_to_out_dir(frame_idx, depth_image, prefix='depth')

        if self.args.save_inpaint_masks and self.inpaint_mask is not None:
            self.save_to_out_dir(frame_idx, self.inpaint_mask, prefix='mask')

        self.save_to_out_dir(frame_idx, out_frame)
        return out_frame

    def generate_depth_image(self, image: Image.Image) -> Image.Image:
        results, _ = self.api.transform(
            [image], 
            depth_calc_transform(blend_weight=self.args.depth_model_weight)
        )
        return results[0]

    def get_animation_prompts_weights(self, frame_idx: int) -> Tuple[List[str], List[float]]:
        prev, next, tween = self.get_key_frame_tween(frame_idx)
        if prev == next or not self.args.interpolate_prompts:
            return [self.animation_prompts[prev]], [1.0]
        else:
            return [self.animation_prompts[prev], self.animation_prompts[next]], [1.0 - tween, tween]

    def get_color_match_image(self, frame_idx: int) -> Image.Image:
        if not self.args.color_match_animate:
            return self.color_match_images.get(0)

        prev, next, tween = self.get_key_frame_tween(frame_idx)

        if prev not in self.color_match_images:
            self.color_match_images[prev] = self._render_frame(prev, self.args.seed)
        prev_match = self.color_match_images[prev]
        if prev == next:
            return prev_match

        if next not in self.color_match_images:
            self.color_match_images[next] = self._render_frame(next, self.args.seed)
        next_match = self.color_match_images[next]

        # Create image combining colors from previous and next key frames without mixing
        # the RGB values. Tiles of next key frame are filled in over tiles of previous 
        # key frame. The tween value increases the subtile size on each axis so the transition
        # is non-linear - staying with previous key frame longer then quickly moving to next.
        blended = prev_match.copy()
        width, height, tile_size = blended.width, blended.height, 64
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                cut = next_match.crop((x, y, x + int(tile_size * tween), y + int(tile_size * tween)))
                blended.paste(cut, (x, y))
        return blended

    def get_key_frame_tween(self, frame_idx: int) -> Tuple[int, int, float]:
        """Returns previous and next key frames along with in between ratio"""
        keys = self.key_frame_values
        idx = bisect.bisect_right(keys, frame_idx)
        prev, next = idx - 1, idx
        if next == len(keys):
            return keys[-1], keys[-1], 1.0
        else:
            tween = (frame_idx - keys[prev]) / (keys[next] - keys[prev])
            return keys[prev], keys[next], tween

    def get_frame_filename(self, frame_idx, prefix="frame") -> Optional[str]:
        return os.path.join(self.out_dir, f"{prefix}_{frame_idx:05d}.png") if self.out_dir else None

    def image_resize(self, img: Image.Image, mode: str = 'stretch') -> Image.Image:
        width, height = img.size
        if mode == 'cover':
            scale = max(self.args.width / width, self.args.height / height)
            img = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            x = (img.width - self.args.width) // 2
            y = (img.height - self.args.height) // 2
            img = img.crop((x, y, x + self.args.width, y + self.args.height))
        elif mode == 'stretch':
            img = img.resize((self.args.width, self.args.height), resample=Image.LANCZOS)
        else:  # 'resize-canvas'
            width, height = map(lambda x: x - x % 64, (width, height))
            self.args.width, self.args.height = width, height
        return img

    def inpaint_frame(
        self,
        frame_idx: int,
        image: Image.Image,
        mask: Image.Image,
        seed: Optional[int] = None,
        mask_blur_radius: Optional[int] = 8
    ) -> Image.Image:
        args = self.args
        steps = int(self.frame_args.steps_curve[frame_idx])
        sampler = sampler_from_string(args.sampler.lower())
        guidance = guidance_from_string(args.clip_guidance)

        # fetch set of prompts and weights for this frame
        prompts, weights = self.get_animation_prompts_weights(frame_idx)
        if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
            prompts.append(self.negative_prompt)
            weights.append(-abs(self.negative_prompt_weight))

        if args.use_inpainting_model:
            binary_mask = self._postprocess_inpainting_mask(
                mask, binarize=True, blur_radius=mask_blur_radius)
            results = self.api.inpaint(
                image, binary_mask,
                prompts, weights, 
                steps=steps,
                seed=seed if seed is not None else args.seed,
                cfg_scale=args.cfg_scale,
                sampler=sampler, 
                init_strength=0.0,
                masked_area_init=generation.MASKED_AREA_INIT_ZERO,
                guidance_preset=guidance,
                preset=args.preset,
            )
        else:
            mask_min_value = self.frame_args.mask_min_value[frame_idx]
            binary_mask = self._postprocess_inpainting_mask(
                mask, binarize=True, min_val=mask_min_value, blur_radius=mask_blur_radius)
            adjusted_steps = max(5, int(steps * (1.0 - mask_min_value))) if args.steps_strength_adj else steps
            noise_scale = self.frame_args.noise_scale_curve[frame_idx]
            results = self.api.generate(
                prompts, weights, 
                args.width, args.height, 
                steps=adjusted_steps,
                seed=seed if seed is not None else args.seed,
                cfg_scale=args.cfg_scale,
                sampler=sampler,
                init_image=image,
                init_strength=mask_min_value,
                init_noise_scale=noise_scale,
                mask=binary_mask,
                masked_area_init=generation.MASKED_AREA_INIT_ORIGINAL,
                guidance_preset=guidance,
                preset=args.preset,
            )
        return results[generation.ARTIFACT_IMAGE][0]

    def load_init_image(self, fpath=None):
        if fpath is None:
            fpath =  self.args.init_image
        if not fpath:
            return

        img = self.image_resize(Image.open(fpath), self.args.init_sizing)
            
        self.prior_frames.extend([img, img])
        self.prior_diffused.extend([img, img])

    def load_mask(self):
        if not self.args.mask_path:
            return

        # try to load mask as an image
        mask = Image.open(self.args.mask_path)
        if mask is not None:
            self.set_mask(mask)

        # try to load mask as a video
        if self.mask is None:
            self.mask_reader = cv2.VideoCapture(self.args.mask_path)
            self.next_mask()

        if self.mask is None:
            raise Exception(f"Failed to read mask from {self.args.mask_path}")

    def load_video(self):
        if self.args.animation_mode != 'Video Input' or not self.args.video_init_path:
            return

        self.video_reader = cv2.VideoCapture(self.args.video_init_path)
        if self.video_reader is not None:
            success, image = self.video_reader.read()
            if not success:
                raise Exception(f"Failed to read first frame from {self.args.video_init_path}")
            self.video_prev_frame = self.image_resize(cv2_to_pil(image), 'cover')
            self.prior_frames.extend([self.video_prev_frame, self.video_prev_frame])
            self.prior_diffused.extend([self.video_prev_frame, self.video_prev_frame])

    def next_mask(self):
        if not self.mask_reader:
            return False

        for _ in range(self.args.extract_nth_frame):
            success, mask = self.mask_reader.read()
            if not success:
                return

        self.set_mask(cv2_to_pil(mask))

    def prepare_init_ops(self, init_image: Optional[Image.Image], frame_idx: int, noise_seed:int) -> List[generation.TransformParameters]:
        if init_image is None:
            return []

        args, frame_args = self.args, self.frame_args
        brightness = frame_args.brightness_curve[frame_idx]
        contrast = frame_args.contrast_curve[frame_idx]
        hue = frame_args.hue_curve[frame_idx]
        saturation = frame_args.saturation_curve[frame_idx]
        lightness = frame_args.lightness_curve[frame_idx]
        noise_amount = frame_args.noise_add_curve[frame_idx]

        color_match_image = None
        if args.color_coherence != 'None' and frame_idx > 0:
            color_match_image = self.get_color_match_image(frame_idx)

        do_color_match = args.color_coherence != 'None' and color_match_image is not None
        do_bchsl = brightness != 1.0 or contrast != 1.0 or hue != 0.0 or saturation != 1.0 or lightness != 0.0
        do_noise = noise_amount > 0.0

        init_ops: List[generation.TransformParameters] = []

        if do_color_match or do_bchsl or do_noise:
            init_ops.append(color_adjust_transform(
                brightness=brightness,
                contrast=contrast,
                hue=hue,
                saturation=saturation,
                lightness=lightness,
                match_image=color_match_image,
                match_mode=args.color_coherence,
                noise_amount=noise_amount,
                noise_seed=noise_seed
            ))

        return init_ops

    def render(self) -> Generator[Image.Image, None, None]:
        args = self.args
        seed = args.seed

        # experimental span-based outpainting mode
        if args.cadence_spans and args.animation_mode != 'Video Input':
            for idx, frame in self._spans_render():
                yield self.emit_frame(idx, frame)
            return

        for frame_idx in range(self.start_frame_idx, args.max_frames):
            # select image generation model
            self.api._generate.engine_id = args.custom_model if args.model == "custom" else args.model
            if model_requires_depth(args.model) and not self.prior_frames:
                self.api._generate.engine_id = DEFAULT_MODEL

            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_curve[frame_idx]))
            self.set_cadence_mode(enabled=(diffusion_cadence > 1))
            is_diffusion_frame = (frame_idx - self.diffusion_cadence_ofs) % diffusion_cadence == 0

            steps = int(self.frame_args.steps_curve[frame_idx])
            strength = max(0.0, self.frame_args.strength_curve[frame_idx])

            # fetch set of prompts and weights for this frame
            prompts, weights = self.get_animation_prompts_weights(frame_idx)
            if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
                prompts.append(self.negative_prompt)
                weights.append(-abs(self.negative_prompt_weight))

            
            # transform prior frames
            stashed_prior_frames = [i.copy() for i in self.prior_frames] if self.mask is not None else []
            self.inpaint_mask = None
            if args.animation_mode == '2D':
                self.inpaint_mask = self.transform_2d(frame_idx)
            elif args.animation_mode in ('3D render', '3D warp'):
                self.inpaint_mask = self.transform_3d(frame_idx)
            elif args.animation_mode == 'Video Input':
                self.inpaint_mask = self.transform_video(frame_idx)

            # apply inpainting
            # If cadence is disabled and inpainting is performed using the same model as for generation,
            # we can optimize inpaint->generate calls into a single generate call.
            if args.inpaint_border and self.inpaint_mask is not None \
                    and (self.cadence_on or args.use_inpainting_model):
                for i in range(len(self.prior_frames)):
                    # The earliest prior frame will be popped right after the generation step, so its inpainting would be redundant.
                    if self.cadence_on and is_diffusion_frame and i==0:
                        continue
                    self.prior_frames[i] = self.inpaint_frame(
                        frame_idx, self.prior_frames[i], self.inpaint_mask,
                        seed=None if args.use_inpainting_model else seed)

            # apply mask to transformed prior frames
            self.next_mask()
            if self.mask is not None:
                for i in range(len(self.prior_frames)):
                    if self.cadence_on and is_diffusion_frame and i==0:
                        continue
                    self.prior_frames[i] = image_mix(self.prior_frames[i], stashed_prior_frames[i], self.mask)

            # either run diffusion or emit an inbetween frame
            if is_diffusion_frame:
                init_image = self.prior_frames[-1] if len(self.prior_frames) and strength > 0 else None
                init_strength = strength if init_image is not None else 0.0

                # mix video frame into init image
                mix_in = self.frame_args.video_mix_in_curve[frame_idx]
                if init_image is not None and mix_in > 0 and self.video_prev_frame is not None:
                    init_image = image_mix(init_image, self.video_prev_frame, mix_in)

                # when using depth model, compute a depth init image
                init_depth = None
                if init_image is not None and model_requires_depth(args.model):
                    depth_source = self.video_prev_frame if self.video_prev_frame is not None else init_image
                    params = depth_calc_transform(blend_weight=1.0, blur_radius=0, reverse=True)
                    results, _ = self.api.transform([depth_source], params)
                    init_depth = results[0]

                # builds set of transform ops to prepare init image for generation
                init_image_ops = self.prepare_init_ops(init_image, frame_idx, seed)

                # For in-diffusion frames instead of a full run through inpainting model and then generate call,
                # inpainting can be done in a single call with non-inpainting model
                do_inpainting = not self.cadence_on and not args.use_inpainting_model \
                        and self.inpaint_mask is not None \
                        and (args.inpaint_border or args.animation_mode == '3D render')
                if do_inpainting:
                    mask_min_value = self.frame_args.mask_min_value[frame_idx]
                    init_strength = min(strength, mask_min_value) 
                    self.inpaint_mask = self._postprocess_inpainting_mask(
                        self.inpaint_mask, 
                        mask_pow=args.mask_power if args.animation_mode == '3D render' else None,
                        mask_multiplier=strength,
                        blur_radius=None,
                        min_val=mask_min_value)

                # generate the next frame
                sampler = sampler_from_string(args.sampler.lower())
                guidance = guidance_from_string(args.clip_guidance)
                noise_scale = self.frame_args.noise_scale_curve[frame_idx]
                adjusted_steps = int(max(5, steps*(1.0-init_strength))) if args.steps_strength_adj else int(steps)
                generate_request = self.api.generate(
                    prompts, weights, 
                    args.width, args.height, 
                    steps=adjusted_steps,
                    seed=seed,
                    cfg_scale=args.cfg_scale,
                    sampler=sampler, 
                    init_image=init_image if init_image_ops is None else None, 
                    init_strength=init_strength,
                    init_noise_scale=noise_scale, 
                    init_depth=init_depth,
                    mask = self.inpaint_mask if do_inpainting else self.mask,
                    masked_area_init=generation.MASKED_AREA_INIT_ORIGINAL,
                    guidance_preset=guidance,
                    preset=args.preset,
                    return_request=True
                )
                image = self.api.transform_and_generate(init_image, init_image_ops, generate_request)

                if args.color_coherence != 'None' and frame_idx == 0:
                    self.color_match_images[0] = image
                if not len(self.prior_frames):
                    self.prior_frames.append(image)
                    self.prior_diffused.append(image)
                    self.prior_xforms.append(matrix.identity)

                self.prior_frames.append(image)
                self.prior_diffused.append(image)
                self.prior_xforms.append(matrix.identity)
                self.diffusion_cadence_ofs = frame_idx
                out_frame = image if not self.cadence_on else self.prior_frames[0]
            else:
                assert self.cadence_on
                # smoothly blend between prior frames
                tween = ((frame_idx - self.diffusion_cadence_ofs) % diffusion_cadence) / float(diffusion_cadence)
                out_frame = self.api.interpolate(
                    [self.prior_frames[0], self.prior_frames[1]],
                    [tween],
                    interpolate_mode_from_string(args.cadence_interp)
                )[0]

            # save and return final frame
            yield self.emit_frame(frame_idx, out_frame)

            if not args.locked_seed:
                seed += 1

    def save_settings(self, filename: str):
        settings_filepath = os.path.join(self.out_dir, filename) if self.out_dir else filename
        with open(settings_filepath, "w", encoding="utf-8") as f:
            save_dict = args_to_dict(self.args)
            for k in ['angle', 'zoom', 'translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y', 'rotation_z']:
                save_dict.move_to_end(k, last=True)
            save_dict['animation_prompts'] = self.animation_prompts
            save_dict['negative_prompt'] = self.negative_prompt
            save_dict['negative_prompt_weight'] = self.negative_prompt_weight
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

    def save_to_out_dir(self, frame_idx: int, image: Image.Image, prefix: str = "frame"):
        if self.out_dir is not None:
            image.save(self.get_frame_filename(frame_idx, prefix=prefix))

    def set_mask(self, mask: Image.Image):
        self.mask = mask.convert('L').resize((self.args.width, self.args.height), resample=Image.LANCZOS)

        # this is intentionally flipped because we want white in the mask to represent
        # areas that should change which is opposite from the backend which treats
        # the mask as per pixel offset in the schedule starting value
        if not self.args.mask_invert:
            self.mask = ImageOps.invert(self.mask)

    def set_cadence_mode(self, enabled: bool):
        def set_queue_size(prior_queue: deque, prev_length: int, new_length: int) -> deque:
            assert new_length in (1, 2)
            if new_length == prev_length:
                return prior_queue
            new_queue: deque = deque([], new_length)
            if len(prior_queue) > 0:
                if new_length == 2 and prev_length == 1:
                    new_queue.extend([prior_queue[0], prior_queue[0]])
                elif new_length == 1 and prev_length == 2:
                    new_queue.append(prior_queue[-1])        
            return new_queue
        
        if enabled == self.cadence_on:
            return
        elif enabled:
            self.prior_frames = set_queue_size(self.prior_frames, 1, 2)
            self.prior_diffused = set_queue_size(self.prior_diffused, 1, 2)
            self.prior_xforms = set_queue_size(self.prior_xforms, 1, 2)
        else:
            self.prior_frames = set_queue_size(self.prior_frames, 2, 1)
            self.prior_diffused = set_queue_size(self.prior_diffused, 2, 1)
            self.prior_xforms = set_queue_size(self.prior_xforms, 2, 1)
        self.cadence_on = enabled

    def setup_animation(self, resume):
        args = self.args

        # change request for random seed into explicit value so it is saved to settings
        if args.seed <= 0:
            args.seed = random.randint(0, 2**32 - 1)

        # select image generation model
        self.api._generate.engine_id = args.custom_model if args.model == "custom" else args.model

        # validate dimensions
        if args.width % 64 != 0 or args.height % 64 != 0:
            args.width, args.height = map(lambda x: x - x % 64, (args.width, args.height))
            logger.warning(f"Adjusted dimensions to {args.width}x{args.height} to be multiples of 64.")

        # validate border settings
        if args.border == 'wrap' and args.animation_mode != '2D':
            args.border = 'reflect'
            logger.warning(f"Border 'wrap' is only supported in 2D mode, switching to '{args.border}'.")
        if args.border == 'prefill' and args.animation_mode in ('2D', '3D warp') and not args.inpaint_border:
            args.border = 'reflect'
            logger.warning(f"Border 'prefill' is only supported when 'inpaint_border' is enabled, switching to '{args.border}'.")

        # validate clip guidance setting against selected model and sampler
        if args.clip_guidance.lower() != 'none':
            if not (model_supports_clip_guidance(args.model) and sampler_supports_clip_guidance(args.sampler)):
                unsupported = args.model if not model_supports_clip_guidance(args.model) else args.sampler
                logger.warning(f"CLIP guidance is not supported by {unsupported}, disabling guidance.")
                args.clip_guidance = 'None'

        # expand key frame strings to per frame series
        frame_args_dict = {f.name: curve_from_cn_string(getattr(args, f.name)) for f in fields(FrameArgs)}
        self.frame_args = FrameArgs(**frame_args_dict)        

        # prepare sorted list of key frames
        self.key_frame_values = sorted(list(self.animation_prompts.keys()))
        if self.key_frame_values[0] != 0:
            raise ValueError("First keyframe must be 0")
        if len(self.key_frame_values) != len(set(self.key_frame_values)):
            raise ValueError("Duplicate keyframes are not allowed!")

        diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_curve[self.start_frame_idx]))
        # initialize accumulated transforms
        self.set_cadence_mode(enabled=(diffusion_cadence > 1))
        self.prior_xforms.extend([matrix.identity, matrix.identity])

        # prepare inputs
        self.load_mask()
        self.load_video()
        self.load_init_image()

        # handle resuming animation from last frames of a previous run
        if resume:
            if not self.out_dir:
                raise ValueError("Cannot resume animation without out_dir specified")
            frames = [f for f in os.listdir(self.out_dir) if f.endswith(".png") and f.startswith("frame_")]
            self.start_frame_idx = len(frames)
            self.diffusion_cadence_ofs = self.start_frame_idx
            if self.start_frame_idx > 2:
                prev = Image.open(self.get_frame_filename(self.start_frame_idx-2))
                next = Image.open(self.get_frame_filename(self.start_frame_idx-1))
                self.prior_frames.extend([prev, next])
                self.prior_diffused.extend([prev, next])
            elif self.start_frame_idx > 1 and not self.cadence_on:
                prev = Image.open(self.get_frame_filename(self.start_frame_idx-1))
                self.prior_frames.append(prev)
                self.prior_diffused.append(prev)

    def transform_2d(self, frame_idx) -> Optional[Image.Image]:
        if not len(self.prior_frames):
            return None

        # create xform for the current frame
        xform = self.build_frame_xform(frame_idx)

        # check if we can skip transform request
        if np.allclose(xform, matrix.identity):
            return None

        args = self.args
        if not args.inpaint_border:
            # apply xform to prior frames running xforms
            for i in range(len(self.prior_xforms)):
                self.prior_xforms[i] = matrix.multiply(xform, self.prior_xforms[i])

            # warp prior diffused frames by accumulated xforms
            for i in range(len(self.prior_diffused)):
                params = resample_transform(args.border, to_3x3(self.prior_xforms[i]), export_mask=args.inpaint_border)
                xformed, mask = self.api.transform([self.prior_diffused[i]], params)
                self.prior_frames[i] = xformed[0]
        else:
            params = resample_transform(args.border, to_3x3(xform), export_mask=args.inpaint_border)
            transformed_prior_frames, mask = self.api.transform(self.prior_frames, params)
            self.prior_frames.extend(transformed_prior_frames)

        return mask[0] if isinstance(mask, list) else mask

    def transform_3d(self, frame_idx) -> Optional[Image.Image]:
        if not len(self.prior_frames):
            return None

        args, frame_args = self.args, self.frame_args
        near, far = args.near_plane, args.far_plane
        fov = frame_args.fov_curve[frame_idx]
        depth_blur = int(frame_args.depth_blur_curve[frame_idx])
        depth_warp = frame_args.depth_warp_curve[frame_idx]
        
        depth_calc = depth_calc_transform(args.depth_model_weight, depth_blur)

        # create xform for the current frame
        world_view = self.build_frame_xform(frame_idx)
        projection = matrix.projection_fov(math.radians(fov), 1.0, near, far)

        if False:
            # currently disabled. for 3D mode transform accumulation needs additional 
            # depth map changes to work properly without swimming artifacts

            # apply world_view xform to prior frames running xforms
            for i in range(len(self.prior_xforms)):
                self.prior_xforms[i] = matrix.multiply(world_view, self.prior_xforms[i])

            # warp prior diffused frames by accumulated xforms
            for i in range(len(self.prior_diffused)):
                wvp = matrix.multiply(projection, self.prior_xforms[i])
                resample = resample_transform(args.border, wvp, projection, depth_warp=depth_warp, export_mask=args.inpaint_border)
                xformed, mask = self.api.transform_3d([self.prior_diffused[i]], depth_calc, resample)
                self.prior_frames[i] = xformed[0]
        else:
            if args.animation_mode == '3D warp':
                wvp = matrix.multiply(projection, world_view)
                transform_op = resample_transform(args.border, wvp, projection, depth_warp=depth_warp, export_mask=args.inpaint_border)
            else:
                transform_op = camera_pose_transform(
                    world_view, near, far, fov, 
                    args.camera_type,
                    render_mode=args.render_mode,
                    do_prefill=not args.use_inpainting_model)
            transformed_prior_frames, mask = self.api.transform_3d(self.prior_frames, depth_calc, transform_op)
            self.prior_frames.extend(transformed_prior_frames)
            return mask[0] if isinstance(mask, list) else mask

    def transform_video(self, frame_idx) -> Optional[Image.Image]:
        assert self.video_reader is not None
        if not len(self.prior_frames):
            return None

        args = self.args
        for _ in range(args.extract_nth_frame):
            success, video_next_frame = self.video_reader.read()
            video_next_frame = cv2_to_pil(video_next_frame)
        if success:
            video_next_frame = self.image_resize(video_next_frame, 'cover')
            mask = None
            if args.video_flow_warp and video_next_frame is not None:
                # warp_flow is in `extras` and will change in the future
                prev_b64 = base64.b64encode(image_to_png_bytes(self.video_prev_frame)).decode('utf-8')
                next_b64 = base64.b64encode(image_to_png_bytes(video_next_frame)).decode('utf-8')
                extras = { "warp_flow": { "prev_frame": prev_b64, "next_frame": next_b64, "export_mask": args.inpaint_border } }
                transformed_prior_frames, masks = self.api.transform(self.prior_frames, generation.TransformParameters(), extras=extras)
                if masks is not None:
                    mask = masks[0]
                self.prior_frames.extend(transformed_prior_frames)
            self.video_prev_frame = video_next_frame
            return mask
        return None

    def _postprocess_inpainting_mask(
        self,
        mask: Union[Image.Image, np.ndarray],
        mask_pow: Optional[float] = None,
        mask_multiplier: Optional[float] = None,
        binarize: bool = False,
        blur_radius: Optional[int] = None,
        min_val: Optional[float] = None
    ) -> Image.Image:
        # Being applied in 3D render mode. Camera pose transform operation returns a mask which pixel values encode
        # how much signal from the previous frame is present there. But a mapping from the signal presence values
        # to the optimal per-pixel init strength is unknown, and roughly guessed as a per-pixel power function.
        # Leaving mask_pow=1 results in near objects changing to a greater extent than a natural emergence of fine details when approaching an object.
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        if mask_pow is not None:
            mask = (np.power(mask / 255., mask_pow) * 255).astype(np.uint8)
        if mask_multiplier is not None:
            mask = (mask * mask_multiplier).astype(np.uint8)
        if binarize:
            mask = np.where(mask > self.args.mask_binarization_thr * 255, 255, 0).astype(np.uint8)
        if blur_radius:
            kernel_size = blur_radius*2+1
            mask = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8))
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        if min_val is not None:
            mask = mask.clip(255 * min_val, 255).astype(np.uint8)
        return Image.fromarray(mask)
    
    def _render_frame(
        self, 
        frame_idx: int, 
        seed: int, 
        init: Optional[Image.Image]=None, 
        mask: Optional[Image.Image]=None, 
        strength: Optional[float]=None
    ) -> Image.Image:
        args = self.args
        steps = int(self.frame_args.steps_curve[frame_idx])
        strength = strength if strength is not None else max(0.0, self.frame_args.strength_curve[frame_idx])
        adjusted_steps = int(max(5, steps*(1.0-strength))) if args.steps_strength_adj else int(steps)

        # fetch set of prompts and weights for this frame
        prompts, weights = self.get_animation_prompts_weights(frame_idx)
        if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
            prompts.append(self.negative_prompt)
            weights.append(-abs(self.negative_prompt_weight))

        init_ops = self.prepare_init_ops(init, frame_idx, seed)

        sampler = sampler_from_string(args.sampler.lower())
        guidance = guidance_from_string(args.clip_guidance)
        generate_request = self.api.generate(
            prompts, weights, 
            args.width, args.height, 
            steps = adjusted_steps,
            seed = seed,
            cfg_scale = args.cfg_scale,
            sampler = sampler, 
            init_image = init if init_ops is None else None, 
            init_strength = strength if init is not None else 0.0,
            init_noise_scale = self.frame_args.noise_scale_curve[frame_idx], 
            mask = mask if mask is not None else self.mask,
            masked_area_init = generation.MASKED_AREA_INIT_ORIGINAL,
            guidance_preset = guidance,
            preset = args.preset,
            return_request = True
        )

        result_image = self.api.transform_and_generate(init, init_ops, generate_request)

        if args.color_coherence != 'None' and frame_idx == 0:
            self.color_match_images[0] = result_image

        return result_image

    def _span_render(self, start: int, end: int, prev_frame: Image.Image, next_seed: Callable[[], int]) -> Generator[Tuple[int, Image.Image], None, None]:
        args = self.args

        def apply_xform(frame: Image.Image, xform: matrix.Matrix, frame_idx: int) -> Tuple[Image.Image, Image.Image]:
            args, frame_args = self.args, self.frame_args
            if args.animation_mode == '2D':
                xform = to_3x3(xform)
                frames, masks = self.api.transform([frame], resample_transform(args.border, xform, export_mask=True))
            else:
                fov = frame_args.fov_curve[frame_idx]
                depth_blur = int(frame_args.depth_blur_curve[frame_idx])
                depth_warp = frame_args.depth_warp_curve[frame_idx]
                projection = matrix.projection_fov(math.radians(fov), 1.0, args.near_plane, args.far_plane)                
                wvp = matrix.multiply(projection, xform)
                depth_calc = depth_calc_transform(args.depth_model_weight, depth_blur)
                resample = resample_transform(args.border, wvp, projection, depth_warp=depth_warp, export_mask=True)
                frames, masks = self.api.transform_3d([frame], depth_calc, resample)
            masks = cast(List[Image.Image], masks)
            return frames[0], masks[0]

        # transform the previous frame forward
        accum_xform = matrix.identity
        forward_frames, forward_masks = [], []
        for frame_idx in range(start, end):
            accum_xform = matrix.multiply(self.build_frame_xform(frame_idx), accum_xform)
            frame, mask = apply_xform(prev_frame, accum_xform, frame_idx)
            forward_frames.append(frame)
            forward_masks.append(mask)

        # inpaint the final frame
        if not np.all(forward_masks[-1]):
            forward_frames[-1] = self.inpaint_frame(
                end-1, forward_frames[-1], forward_masks[-1], 
                mask_blur_radius=0, seed=next_seed())

        # run diffusion on top of the final result to allow content to evolve over time
        strength = max(0.0, self.frame_args.strength_curve[end-1])
        if strength < 1.0:
            final_frame = self._render_frame(end-1, next_seed(), forward_frames[-1])
        else:
            final_frame = forward_frames[-1]

        # go backwards through the frames in the span        
        backward_frames, backward_masks = [final_frame], [Image.new('L', forward_masks[-1].size, 255)]
        accum_xform = matrix.identity
        for frame_idx in range(end-2, start-1, -1):
            frame_xform = self.build_frame_xform(frame_idx+1)
            inv_xform = np.linalg.inv(frame_xform).tolist()
            accum_xform = matrix.multiply(inv_xform, accum_xform)
            xformed, mask = apply_xform(backward_frames[-1], accum_xform, frame_idx)
            backward_frames.insert(0, xformed)
            backward_masks.insert(0, mask)

        # inpaint the backwards frame
        if not np.all(backward_masks[0]):
            backward_frames[0] = self.inpaint_frame(
                start, backward_frames[0], backward_masks[0], 
                mask_blur_radius=0, seed=next_seed())

        # yield the final frames blending from forward to backward
        for idx, (frame_fwd, frame_bwd) in enumerate(zip(forward_frames, backward_frames)):
            t = (idx) / max(1, end-start-1)
            fwd_fill = image_mix(frame_bwd, frame_fwd, mask_erode_blur(forward_masks[idx], 8, 8))
            bwd_fill = image_mix(frame_fwd, frame_bwd, mask_erode_blur(backward_masks[idx], 8, 8))
            blended = self.api.interpolate(
                [fwd_fill, bwd_fill], 
                [t], 
                interpolate_mode_from_string(args.cadence_interp)
            )[0]
            yield start+idx, blended

    def _spans_render(self) -> Generator[Tuple[int, Image.Image], None, None]:
        frame_idx = self.start_frame_idx
        seed = self.args.seed
        def next_seed() -> int:
            nonlocal seed
            if not self.args.locked_seed:
                seed += 1
            return seed

        prev_frame = self._render_frame(frame_idx, seed, None)
        yield frame_idx, prev_frame

        while frame_idx < self.args.max_frames:
            # determine how many frames the span will process together
            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_curve[frame_idx]))
            if frame_idx + diffusion_cadence > self.args.max_frames:
                diffusion_cadence = self.args.max_frames - frame_idx

            # render all frames in the span
            for idx, frame in self._span_render(frame_idx, frame_idx + diffusion_cadence, prev_frame, next_seed):
                yield idx, frame
                prev_frame = frame
                next_seed()

            frame_idx += diffusion_cadence

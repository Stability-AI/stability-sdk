import base64
import bisect
import cv2
import json
import logging
import math
import numpy as np
import os
import param
import random

from collections import OrderedDict, deque
from PIL import Image
from types import SimpleNamespace
from typing import Deque, Generator, List, Optional, Tuple, Union

from stability_sdk.client import (
    Api,
    generation,
)
from stability_sdk.utils import (
    blend_op,
    color_adjust_op,
    cv2_to_pil,
    depthcalc_op,
    guidance_from_string,
    image_mix,
    image_to_png_bytes,
    interp_mode_from_str,
    key_frame_inbetweens,
    key_frame_parse,
    resample_op,
    camera_pose_op,
    sampler_from_string,
)
import stability_sdk.matrix as matrix

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

DEFAULT_MODEL = 'stable-diffusion-v1-5'
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
    sampler = param.ObjectSelector(default='K_euler_ancestral', objects=["DDIM", "PLMS", "K_euler", "K_euler_ancestral", "K_heun", "K_dpm_2", "K_dpm_2_ancestral", "K_lms", "K_dpmpp_2m", "K_dpmpp_2s_ancestral"])
    model = param.ObjectSelector(default=DEFAULT_MODEL, objects=["stable-diffusion-v1-5", "stable-diffusion-512-v2-1", "stable-diffusion-768-v2-1", "stable-diffusion-depth-v2-0", "custom"])
    custom_model = param.String(default="", doc="Identifier of custom model to use.")
    seed = param.Integer(default=-1, doc="Provide a seed value for more deterministic behavior. Negative seed values will be replaced with a random seed (default).")
    cfg_scale = param.Number(default=7, softbounds=(0,20), doc="Classifier-free guidance scale. Strength of prompt influence on denoising process. `cfg_scale=0` gives unconditioned sampling.")
    clip_guidance = param.ObjectSelector(default='FastBlue', objects=["None", "Simple", "FastBlue", "FastGreen"], doc="CLIP-guidance preset.")
    init_image = param.String(default='', doc="Path to image. Height and width dimensions will be inherited from image.")
    init_sizing = param.ObjectSelector(default='stretch', objects=["cover", "stretch", "resize-canvas"])
    mask_path = param.String(default="", doc="Path to image or video mask")
    mask_invert = param.Boolean(default=False, doc="White in mask marks areas to change by default.")

class AnimationSettings(param.Parameterized):
    animation_mode = param.ObjectSelector(default='3D warp', objects=['2D', '3D warp', '3D render', 'Video Input'])
    max_frames = param.Integer(default=72, doc="Force stop of animation job after this many frames are generated.")
    border = param.ObjectSelector(default='replicate', objects=['reflect', 'replicate', 'wrap', 'zero', 'prefill'], doc=docstring_bordermode)
    noise_add_curve = param.String(default="0:(0.02)")
    noise_scale_curve = param.String(default="0:(1.02)")
    strength_curve = param.String(default="0:(0.65)", doc="Image Strength (of init image relative to the prompt). 0 for ignore init image and attend only to prompt, 1 would return the init image unmodified")
    steps_curve = param.String(default="0:(50)", doc="Diffusion steps")
    steps_strength_adj = param.Boolean(default=True, doc="Adjusts number of diffusion steps based on current previous frame strength value.")    
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
    cadence_interp = param.ObjectSelector(default='mix', objects=['mix', 'rife', 'vae-lerp', 'vae-slerp'])
    cadence_spans = param.Boolean(default=False, doc="Experimental diffusion cadence mode for better outpainting")


class ColorSettings(param.Parameterized):
    color_coherence = param.ObjectSelector(default='LAB', objects=['None', 'HSV', 'LAB', 'RGB'], doc="Color space that will be used for inter-frame color adjustments.")
    brightness_curve = param.String(default="0:(1.0)")
    contrast_curve = param.String(default="0:(1.0)")
    hue_curve = param.String(default="0:(0.0)")
    saturation_curve = param.String(default="0:(1.0)")
    lightness_curve = param.String(default="0:(0.0)")


class DepthSettings(param.Parameterized):
    depth_model_weight = param.Number(default=0.3, softbounds=(0,1), doc="Blend factor between AdaBins and MiDaS depth models.")
    near_plane = param.Number(default=200, doc="Distance to nearest plane of camera view volume.")
    far_plane = param.Number(default=10000, doc="Distance to furthest plane of camera view volume.")
    fov_curve = param.String(default="0:(25)", doc="FOV angle of camera volume in degrees.")
    depth_blur_curve = param.String(default="0:(0.0)", doc="Blur strength of depth map.")
    depth_warp_curve = param.String(default="0:(1.0)", doc="Depth warp strength.")
    save_depth_maps = param.Boolean(default=False)
    

class Rendering3dSettings(param.Parameterized):
    camera_type = param.ObjectSelector(default='perspective', objects=['perspective', 'orthographic'])
    image_render_method = param.ObjectSelector(default='mesh', objects=['pointcloud', 'mesh'])
    # Mask render method is selected based on a type of model (inpainting/non-inpainting) used for inpainting for current frame
    image_render_points_per_pixel = param.Integer(default=8)
    image_render_point_radius = param.Number(default=0.006)
    image_max_mesh_edge = param.Number(default=0.1)
    mask_render_points_per_pixel = param.Integer(default=4)
    mask_render_point_radius = param.Number(default=0.0045)
    mask_max_mesh_edge = param.Number(default=0.04)

class InpaintingSettings(param.Parameterized):
    non_inpainting_model_for_diffusion_frames = param.Boolean(default=False, doc="If True, for each diffusion frame, inpainting will be conducted using regular non-inpainting model to optimize number of generations.")
    inpaint_border = param.Boolean(default=False, doc="Use inpainting on top of border regions for 2D and 3D warp modes. Defaults to False")
    do_mask_fixup = param.Boolean(default=True, doc="Enforce pixels outside of inpainting region to be equal to original frame")
    mask_min_value = param.String(default="0:(0.1)", doc="Mask postprocessing for non-inpainting model. Mask floor values will be clipped by this value prior to inpainting")
    save_inpaint_masks = param.Boolean(default=False)

class VideoInputSettings(param.Parameterized):
    video_init_path = param.String(default="", doc="Path to video input")
    extract_nth_frame = param.Integer(default=1, bounds=(1,None), doc="Only use every Nth frame of the video")
    video_mix_in_curve = param.String(default="0:(0.02)")
    video_flow_warp = param.Boolean(default=True, doc="Whether or not to transfer the optical flow from the video to the generated animation as a warp effect.")

class VideoOutputSettings(param.Parameterized):
    fps = param.Integer(default=24, doc="Frame rate to use when generating video output.")
    reverse = param.Boolean(default=False, doc="Whether to reverse the output video or not.")
    vr_mode = param.Boolean(default=False, doc="Outputs side by side views for each eye using depth warp.")
    vr_eye_angle = param.Number(default=0.5, softbounds=(0,1), doc="Y-axis rotation of the eyes towards the center.")
    vr_eye_dist = param.Number(default=5.0, softbounds=(0,1), doc="Interpupillary distance (between the eyes)")
    vr_projection = param.Number(default=-0.4, softbounds=(-1,1), doc="Spherical projection of the video.")

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
    pass


def args2dict(args):
    """
    Converts arguments object to an OrderedDict
    """
    f = None
    if isinstance(args, param.Parameterized):
        f = args2dict_param
    if isinstance(args, SimpleNamespace):
        f = args2dict_simplenamespace
    if f is None:
        raise NotImplementedError(f"Unsupported arguments object type: {type(args)}")
    return f(args)

def args2dict_simplenamespace(args):
    return OrderedDict(vars(args))

def args2dict_param(args):
    return OrderedDict(args.param.values())

def mask_erode_blur(mask: np.ndarray, mask_erode: int, mask_blur: int) -> np.ndarray:
    if mask_erode > 0:
        ks = mask_erode*2 + 1
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)), iterations=1)
    if mask_blur > 0:
        ks = mask_blur*2 + 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
    return mask

def make_xform_2d(
    w: float, h: float,
    rotate: float, # in radians
    scale: float,
    translate_x: float,
    translate_y: float,
) -> matrix.Matrix:
    center = (w / 2, h / 2)
    pre = matrix.translation(-center[0], -center[1], 0)
    post = matrix.translation(center[0], center[1], 0)
    rotate = matrix.rotation_euler(0, 0, rotate)
    scale = matrix.scale(scale, scale, 1)
    rotate_scale = matrix.multiply(post, matrix.multiply(rotate, matrix.multiply(scale, pre)))
    translate = matrix.translation(translate_x, translate_y, 0)
    return matrix.multiply(rotate_scale, translate)

def model_requires_depth(model_name: str) -> bool:
    return model_name == 'stable-diffusion-depth-v2-0'

def to_3x3(m: matrix.Matrix) -> matrix.Matrix:
    # convert 4x4 matrix with 2D rotation, scale, and translation to 3x3 matrix
    return [[m[0][0], m[0][1], m[0][3]],
            [m[1][0], m[1][1], m[1][3]],
            [m[3][0], m[3][1], m[3][3]]]


class Animator:
    def __init__(
        self,
        api: Api,
        animation_prompts,
        args=None,
        out_dir='.',
        negative_prompt='',
        negative_prompt_weight=-1.0,
        resume: bool = False
    ):
        self.api = api
        self.animation_prompts = animation_prompts
        self.args = args
        self.color_match_image: Optional[np.ndarray] = None
        self.diffusion_cadence_ofs: int = 0
        self.frame_args = None
        self.keyframe_values: List[int] = None
        self.out_dir: str = out_dir
        self.mask: Optional[np.ndarray] = None
        self.mask_reader = None
        self.cadence_on = False
        self.prior_frames: Deque[np.ndarray] = deque([], 1)    # forward warped prior frames. stores one image with cadence off, two images otherwise
        self.prior_diffused: Deque[np.ndarray] = deque([], 1)  # results of diffusion. stores one image with cadence off, two images otherwise
        self.prior_xforms: Deque[matrix.Matrix] = deque([], 1)   # accumulated transforms since last diffusion. stores one with cadence off, two otherwise
        self.negative_prompt: str = negative_prompt
        self.negative_prompt_weight: float = negative_prompt_weight
        self.start_frame_idx: int = 0
        self.video_prev_frame: Optional[np.ndarray] = None
        self.video_reader = None

        # configure Api to retry on classifier obfuscations
        self.api._retry_obfuscation = True

        self.setup_animation(resume)

    def build_frame_xform(self, frame_idx) -> matrix.Matrix:
        args, frame_args = self.args, self.frame_args

        if self.args.animation_mode == '2D':
            angle = frame_args.angle_series[frame_idx]
            scale = frame_args.zoom_series[frame_idx]
            dx = frame_args.translation_x_series[frame_idx]
            dy = frame_args.translation_y_series[frame_idx]
            return make_xform_2d(args.width, args.height, math.radians(angle), scale, dx, dy)

        elif self.args.animation_mode in ('3D warp', '3D render'):
            dx = frame_args.translation_x_series[frame_idx]
            dy = frame_args.translation_y_series[frame_idx]
            dz = frame_args.translation_z_series[frame_idx]
            rx = frame_args.rotation_x_series[frame_idx]
            ry = frame_args.rotation_y_series[frame_idx]
            rz = frame_args.rotation_z_series[frame_idx]

            dx, dy, dz = -dx*TRANSLATION_SCALE, dy*TRANSLATION_SCALE, -dz*TRANSLATION_SCALE
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)

            # create xform for the current frame
            world_view = matrix.multiply(matrix.translation(dx, dy, dz), matrix.rotation_euler(rx, ry, rz))
            return world_view

        else:
            return matrix.identity

    def emit_frame(self, frame_idx: int, out_frame: np.ndarray) -> Image.Image:
        if self.args.save_depth_maps:
            depth_image = self.generate_depth_image(out_frame)
            cv2.imwrite(self.get_frame_filename(frame_idx, prefix='depth'), depth_image)

        if self.args.save_inpaint_masks and self.inpaint_mask is not None:
            cv2.imwrite(self.get_frame_filename(frame_idx, prefix='mask'), self.inpaint_mask)

        if self.args.vr_mode:
            stereo_frame = self.render_stereo_eye_views(frame_idx, out_frame)
            cv2.imwrite(self.get_frame_filename(frame_idx), stereo_frame)
            return cv2_to_pil(stereo_frame)
        else:
            cv2.imwrite(self.get_frame_filename(frame_idx), out_frame)
            return cv2_to_pil(out_frame)

    def generate_depth_image(self, image: np.ndarray) -> np.ndarray:
        results, _ = self.api.transform(
            [image], 
            depthcalc_op(blend_weight=self.args.depth_model_weight)
        )
        return results[0]

    def get_animation_prompts_weights(self, frame_idx: int) -> Tuple[List[str], List[float]]:
        keys = self.key_frame_values
        idx = bisect.bisect_right(keys, frame_idx)
        prev, next = idx - 1, idx
        if not self.args.interpolate_prompts:
            return [self.animation_prompts[keys[min(len(keys)-1, prev)]]], [1.0]
        elif next == len(keys):
            return [self.animation_prompts[keys[-1]]], [1.0]
        else:
            tween = (frame_idx - keys[prev]) / (keys[next] - keys[prev])
            return [self.animation_prompts[keys[prev]], self.animation_prompts[keys[next]]], [1.0 - tween, tween]

    def get_frame_filename(self, frame_idx, prefix="frame"):
        return os.path.join(self.out_dir, f"{prefix}_{frame_idx:05d}.png")

    def image_resize(self, img: np.ndarray, mode: str='stretch') -> np.ndarray:
        height, width, _ = img.shape
        if mode == 'cover':
            scale = max(self.args.width / width, self.args.height / height)
            img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
            x = (img.shape[1] - self.args.width) // 2
            y = (img.shape[0] - self.args.height) // 2
            img = img[y:y+self.args.height, x:x+self.args.width]
        elif mode == 'stretch':
            img = cv2.resize(img, (self.args.width, self.args.height), interpolation=cv2.INTER_LANCZOS4)
        else: # 'resize-canvas'
            width, height = map(lambda x: x - x % 64, (width, height))
            self.args.width, self.args.height = width, height
        return img

    def inpaint_frame(self, frame_idx: int, image: np.ndarray, mask: np.ndarray,
                      use_inpaint_model: bool=True, mask_fixup: Optional[bool]=None) -> np.ndarray:
        args = self.args
        steps = int(self.frame_args.steps_series[frame_idx])
        strength = max(0.0, self.frame_args.strength_series[frame_idx])
        adjusted_steps = int(max(5, steps*(1.0-strength))) if args.steps_strength_adj else int(steps)
        sampler = sampler_from_string(args.sampler.lower())
        guidance = guidance_from_string(args.clip_guidance)

        # fetch set of prompts and weights for this frame
        prompts, weights = self.get_animation_prompts_weights(frame_idx)
        if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
            prompts.append(self.negative_prompt)
            weights.append(-abs(self.negative_prompt_weight))

        if use_inpaint_model:
            results = self.api.inpaint(
                image, mask,
                prompts, weights, 
                steps=adjusted_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                sampler=sampler, 
                init_strength=0.0,
                masked_area_init=generation.MASKED_AREA_INIT_ZERO,
                mask_fixup=mask_fixup if mask_fixup is not None else False,
                guidance_preset=guidance,
            )
        else:
            results = self.api.generate(
                prompts, weights, 
                args.width, args.height, 
                steps=adjusted_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                sampler=sampler, 
                init_image=image, 
                init_strength=strength if image is not None else 0.0,
                mask=mask,
                masked_area_init=generation.MASKED_AREA_INIT_ORIGINAL,
                mask_fixup=mask_fixup if mask_fixup is not None else True,
                guidance_preset=guidance,
            )
        return results[generation.ARTIFACT_IMAGE][0]

    def save_settings(self, filename: str):
        settings_filepath = os.path.join(self.out_dir, filename)
        with open(settings_filepath, "w+", encoding="utf-8") as f:
            save_dict = args2dict(self.args)
            for k in ['angle', 'zoom', 'translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y', 'rotation_z']:
                save_dict.move_to_end(k, last=True)
            save_dict['animation_prompts'] = self.animation_prompts
            save_dict['negative_prompt'] = self.negative_prompt
            save_dict['negative_prompt_weight'] = self.negative_prompt_weight
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

    def setup_animation(self, resume):
        args = self.args

        # change request for random seed into explicit value so it is saved to settings
        if args.seed <= 0:
            args.seed = random.randint(0, 2**32 - 1)

        def curve_to_series(curve: str) -> List[float]:
            return key_frame_inbetweens(key_frame_parse(curve), args.max_frames)    

        # expand key frame strings to per frame series
        self.frame_args = SimpleNamespace(**dict(
            angle_series = curve_to_series(args.angle),
            zoom_series = curve_to_series(args.zoom),
            translation_x_series = curve_to_series(args.translation_x),
            translation_y_series = curve_to_series(args.translation_y),
            translation_z_series = curve_to_series(args.translation_z),
            rotation_x_series = curve_to_series(args.rotation_x),
            rotation_y_series = curve_to_series(args.rotation_y),
            rotation_z_series = curve_to_series(args.rotation_z),
            brightness_series = curve_to_series(args.brightness_curve),
            contrast_series = curve_to_series(args.contrast_curve),
            hue_series = curve_to_series(args.hue_curve),
            saturation_series = curve_to_series(args.saturation_curve),
            lightness_series = curve_to_series(args.lightness_curve),
            noise_add_series = curve_to_series(args.noise_add_curve),
            noise_scale_series = curve_to_series(args.noise_scale_curve),
            steps_series = curve_to_series(args.steps_curve),
            strength_series = curve_to_series(args.strength_curve),
            diffusion_cadence_series = curve_to_series(args.diffusion_cadence_curve),
            fov_series = curve_to_series(args.fov_curve),
            depth_blur_series = curve_to_series(args.depth_blur_curve),
            depth_warp_series = curve_to_series(args.depth_warp_curve),
            video_mix_in_series = curve_to_series(args.video_mix_in_curve),
            mask_min_value = curve_to_series(args.mask_min_value),
        ))

        # prepare sorted list of key frames
        self.key_frame_values = sorted(list(self.animation_prompts.keys()))
        if self.key_frame_values[0] != 0:
            raise ValueError("First keyframe must be 0")
        if len(self.key_frame_values) != len(set(self.key_frame_values)):
            raise ValueError("Duplicate keyframes are not allowed!")

        diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_series[self.start_frame_idx]))
        # initialize accumulated transforms
        self.set_cadence_mode(enabled=(diffusion_cadence > 1))
        self.prior_xforms.extend([matrix.identity, matrix.identity])

        if args.animation_mode=="3D render":
            args.near_plane = -1  # Needed for mesh rendering

        # prepare inputs
        self.load_mask()
        self.load_video()
        self.load_init_image()

        # handle resuming animation from last frames of a previous run
        if resume:
            frames = [f for f in os.listdir(self.out_dir) if f.endswith(".png") and f.startswith("frame_")]
            self.start_frame_idx = len(frames)
            self.diffusion_cadence_ofs = self.start_frame_idx
            if self.start_frame_idx > 2:
                prev = cv2.imread(self.get_frame_filename(self.start_frame_idx-2))
                next = cv2.imread(self.get_frame_filename(self.start_frame_idx-1))
                self.prior_frames.extend([prev, next])
                self.prior_diffused.extend([prev, next])
            elif self.start_frame_idx > 1 and not self.cadence_on:
                prev = cv2.imread(self.get_frame_filename(self.start_frame_idx-1))
                self.prior_frames.append(prev)
                self.prior_diffused.append(prev)

    def load_init_image(self, fpath=None):
        if fpath is None:
            fpath =  self.args.init_image
        if not fpath:
            return

        img = self.image_resize(cv2.imread(fpath), self.args.init_sizing)
            
        self.prior_frames.extend([img, img])
        self.prior_diffused.extend([img, img])

    def load_mask(self):
        if not self.args.mask_path:
            return

        # try to load mask as an image
        mask = cv2.imread(self.args.mask_path)
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
            self.video_prev_frame = self.image_resize(image, 'cover')
            self.prior_frames.extend([self.video_prev_frame, self.video_prev_frame])
            self.prior_diffused.extend([self.video_prev_frame, self.video_prev_frame])

    def next_mask(self):
        if not self.mask_reader:
            return False

        for _ in range(self.args.extract_nth_frame):
            success, mask = self.mask_reader.read()
            if not success:
                return

        self.set_mask(mask)

    def prepare_init_ops(self, init_image: Optional[np.ndarray], frame_idx: int, noise_seed:int) -> List[generation.TransformParameters]:
        if init_image is None:
            return []

        args, frame_args = self.args, self.frame_args
        brightness = frame_args.brightness_series[frame_idx]
        contrast = frame_args.contrast_series[frame_idx]
        hue = frame_args.hue_series[frame_idx]
        saturation = frame_args.saturation_series[frame_idx]
        lightness = frame_args.lightness_series[frame_idx]
        noise_amount = frame_args.noise_add_series[frame_idx]

        do_color_match = args.color_coherence != 'None' and self.color_match_image is not None
        do_bchsl = brightness != 1.0 or contrast != 1.0 or hue != 0.0 or saturation != 1.0 or lightness != 0.0
        do_noise = noise_amount > 0.0

        init_ops: List[generation.TransformParameters] = []

        if do_color_match or do_bchsl or do_noise:
            init_ops.append(color_adjust_op(
                brightness=brightness,
                contrast=contrast,
                hue=hue,
                saturation=saturation,
                lightness=lightness,
                match_image=self.color_match_image,
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

            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_series[frame_idx]))
            self.set_cadence_mode(enabled=(diffusion_cadence > 1))
            is_diffusion_frame = (frame_idx - self.diffusion_cadence_ofs) % diffusion_cadence == 0

            steps = int(self.frame_args.steps_series[frame_idx])
            strength = max(0.0, self.frame_args.strength_series[frame_idx])

            # fetch set of prompts and weights for this frame
            prompts, weights = self.get_animation_prompts_weights(frame_idx)
            if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
                prompts.append(self.negative_prompt)
                weights.append(-abs(self.negative_prompt_weight))

            inpaint_with_non_inpainting_model = is_diffusion_frame and args.non_inpainting_model_for_diffusion_frames
            if args.animation_mode == '3D render':
                if inpaint_with_non_inpainting_model:
                    # For non-inpainting model, the pointcloud-rendered grayscale mask is better suited.
                    args.mask_render_method = "pointcloud"
                else:
                    # For inpainting model, the mesh-rendered binary mask is better suited.
                    args.mask_render_method = "mesh"
            
            # transform prior frames
            stashed_prior_frames = [i.copy() for i in self.prior_frames] if self.mask is not None else None
            self.inpaint_mask = None
            if args.animation_mode == '2D':
                self.inpaint_mask = self.transform_2d(frame_idx)
            elif args.animation_mode in ('3D render', '3D warp'):
                self.inpaint_mask = self.transform_3d(frame_idx)
            elif args.animation_mode == 'Video Input':
                self.inpaint_mask = self.transform_video(frame_idx)

            # apply inpainting
            if args.inpaint_border and self.inpaint_mask is not None \
                    and not (args.non_inpainting_model_for_diffusion_frames and not self.cadence_on):
                for i in range(len(self.prior_frames)):
                    # The latest prior frame will be popped right after the generation step, so this inpainting call for it would be redundant.
                    if self.cadence_on and is_diffusion_frame and i==0:
                        continue
                    self.prior_frames[i] = self.inpaint_frame(
                        frame_idx, self.prior_frames[i], self.inpaint_mask,
                        mask_fixup=args.do_mask_fixup,
                        use_inpaint_model=True)

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

                # mix video frame into init image
                mix_in = self.frame_args.video_mix_in_series[frame_idx]
                if init_image is not None and mix_in > 0 and self.video_prev_frame is not None:
                    init_image = image_mix(init_image, self.video_prev_frame, mix_in)

                # when using depth model, compute a depth init image
                init_depth = None
                if init_image is not None and model_requires_depth(args.model):
                    depth_source = self.video_prev_frame if self.video_prev_frame is not None else init_image
                    params = depthcalc_op(blend_weight=1.0, blur_radius=0, reverse=True)
                    results, _ = self.api.transform([depth_source], params)
                    init_depth = results[0]

                # builds set of transform ops to prepare init image for generation
                init_image_ops = self.prepare_init_ops(init_image, frame_idx, seed)

                # For in-diffusion frames instead of a full run through inpainting model and then generate call,
                # inpainting can be done in a single call with non-inpainting model
                do_inpainting = args.non_inpainting_model_for_diffusion_frames \
                        and self.inpaint_mask is not None \
                        and (args.inpaint_border or args.animation_mode == '3D render')
                start_diffusion_from = min(strength, self.frame_args.mask_min_value[frame_idx] if do_inpainting else 1.0)
                if do_inpainting:
                    self.inpaint_mask = self._postprocess_inpainting_mask(self.inpaint_mask, frame_idx, max_val=strength)

                # generate the next frame
                sampler = sampler_from_string(args.sampler.lower())
                guidance = guidance_from_string(args.clip_guidance)
                noise_scale = self.frame_args.noise_scale_series[frame_idx]
                adjusted_steps = int(max(5, steps*(1.0-start_diffusion_from))) if args.steps_strength_adj else int(steps)
                init_strength = (strength if not do_inpainting else start_diffusion_from) if init_image is not None else 0.0
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
                    mask_fixup=args.do_mask_fixup,
                    guidance_preset=guidance,
                    return_request=True
                )
                image = self.api.transform_and_generate(init_image, init_image_ops, generate_request)

                if self.color_match_image is None and args.color_coherence != 'None':
                    self.color_match_image = image
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
                    interp_mode_from_str(args.cadence_interp)
                )[0]

            # save and return final frame
            yield self.emit_frame(frame_idx, out_frame)

            if not args.locked_seed:
                seed += 1

    def render_stereo_eye_views(self, frame_idx, frame: np.ndarray) -> np.ndarray:
        args, frame_args = self.args, self.frame_args
        fov = frame_args.fov_series[frame_idx]
        projection = matrix.projection_fov(math.radians(fov), 1.0, args.near_plane, args.far_plane)

        # VR spherical projection is experimental development feature and may change or be removed
        extras = { "spherical_proj": args.vr_projection }

        eye_images = []
        for eye_idx in range(2):
            theta = args.vr_eye_angle * (math.pi / 180)
            ray_origin = math.cos(theta) * args.vr_eye_dist / 2 * (-1.0 if eye_idx == 0 else 1.0)
            ray_rotation = theta if eye_idx == 0 else -theta                               
            world_view = matrix.multiply(matrix.translation(-(ray_origin) * TRANSLATION_SCALE, 0, 0), 
                                         matrix.rotation_euler(0, math.radians(ray_rotation), 0))
            wvp = matrix.multiply(projection, world_view)
            depth_calc = depthcalc_op(args.depth_model_weight)
            resample = resample_op(args.border, wvp, projection, depth_warp=1.0, export_mask=False)
            results, _ = self.api.transform_resample_3d([frame], depth_calc, resample, extras=extras)
            eye_images.append(results[0])
        
        return np.concatenate(eye_images, axis=1)

    def set_mask(self, mask: np.ndarray):
        self.mask = cv2.resize(mask, (self.args.width, self.args.height), interpolation=cv2.INTER_LANCZOS4)
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        # this is intentionally flipped because we want white in the mask to represent
        # areas that should change which is opposite from the backend which treats
        # the mask as per pixel offset in the schedule starting value
        if not self.args.mask_invert:
            self.mask = 255 - self.mask

    def set_cadence_mode(self, enabled: bool):
        def set_queue_size(prior_queue: deque, prev_length: int, new_length: int) -> deque:
            assert new_length in (1, 2)
            if new_length == prev_length:
                return prior_queue
            new_queue = deque([], new_length)
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

    def transform_2d(self, frame_idx) -> Optional[np.ndarray]:
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
                params = resample_op(args.border, to_3x3(self.prior_xforms[i]), export_mask=args.inpaint_border)
                xformed, mask = self.api.transform([self.prior_diffused[i]], params)
                self.prior_frames[i] = xformed[0]
        else:
            params = resample_op(args.border, to_3x3(xform), export_mask=args.inpaint_border)
            transformed_prior_frames, mask = self.api.transform(self.prior_frames, params)
            self.prior_frames.extend(transformed_prior_frames)

        return mask[0] if isinstance(mask, list) else mask

    def transform_3d(self, frame_idx) -> Optional[np.ndarray]:
        if not len(self.prior_frames):
            return None

        args, frame_args = self.args, self.frame_args
        near, far = args.near_plane, args.far_plane
        fov = frame_args.fov_series[frame_idx]
        depth_blur = int(frame_args.depth_blur_series[frame_idx])
        depth_warp = frame_args.depth_warp_series[frame_idx]
        
        depth_calc = depthcalc_op(args.depth_model_weight, depth_blur)

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
                resample = resample_op(args.border, wvp, projection, depth_warp=depth_warp, export_mask=args.inpaint_border)
                xformed, mask = self.api.transform_3d([self.prior_diffused[i]], depth_calc, resample)
                self.prior_frames[i] = xformed[0]
        else:
            if args.animation_mode == '3D warp':
                wvp = matrix.multiply(projection, world_view)
                transform_op = resample_op(args.border, wvp, projection, depth_warp=depth_warp, export_mask=args.inpaint_border)
            else:
                transform_op = camera_pose_op(
                    world_view, near, far, fov, 
                    args.camera_type,
                    args.image_render_method,
                    args.image_render_point_radius if args.image_render_method=="pointcloud" else None,
                    args.image_render_points_per_pixel if args.image_render_method=="pointcloud" else None,
                    args.image_max_mesh_edge if args.image_render_method=="mesh" else None,
                    args.mask_render_method,
                    args.mask_render_point_radius if args.mask_render_method=="pointcloud" else None,
                    args.mask_render_points_per_pixel if args.mask_render_method=="pointcloud" else None,
                    args.mask_max_mesh_edge if args.mask_render_method=="mesh" else None,
                    True)
            transformed_prior_frames, mask = self.api.transform_3d(self.prior_frames, depth_calc, transform_op)
            self.prior_frames.extend(transformed_prior_frames)
            return mask[0] if isinstance(mask, list) else mask

    def transform_video(self, frame_idx) -> Optional[np.ndarray]:
        if not len(self.prior_frames):
            return None

        op = None
        args = self.args
        for _ in range(args.extract_nth_frame):
            success, video_next_frame = self.video_reader.read()
        if success:
            video_next_frame = self.image_resize(video_next_frame, 'cover')
            mask = None
            if args.video_flow_warp and video_next_frame is not None:
                # warp_flow is in `extras` and will change in the future
                prev_b64 = base64.b64encode(image_to_png_bytes(self.video_prev_frame)).decode('utf-8')
                next_b64 = base64.b64encode(image_to_png_bytes(video_next_frame)).decode('utf-8')
                extras = { "warp_flow": { "prev_frame": prev_b64, "next_frame": next_b64} }
                transformed_prior_frames, mask = self.api.transform(self.prior_frames, None, extras=extras)
                self.prior_frames.extend(transformed_prior_frames)
            self.video_prev_frame = video_next_frame
            self.color_match_image = video_next_frame
            return mask
        return None

    def _postprocess_inpainting_mask(self, mask, frame_idx, blur_radius=5, min_val=None, max_val=None):
        mask = cv2.erode(mask, np.ones((blur_radius, blur_radius), np.uint8))
        mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        mask_min_value = self.frame_args.mask_min_value[frame_idx] if min_val is None else min_val
        mask_max_value = 1.0 if max_val is None else max_val
        return mask.clip(255 * mask_min_value, 255 * mask_max_value).astype(np.uint8)
    
    def _span_render_frame(
        self, 
        frame_idx: int, 
        seed: int, 
        init: Optional[np.ndarray], 
        mask: Optional[np.ndarray]=None, 
        strength: Optional[float]=None
    ) -> np.ndarray:
        args = self.args
        steps = int(self.frame_args.steps_series[frame_idx])
        strength = strength if strength is not None else max(0.0, self.frame_args.strength_series[frame_idx])
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
            init_noise_scale = self.frame_args.noise_scale_series[frame_idx], 
            mask = mask if mask is not None else self.mask,
            masked_area_init = generation.MASKED_AREA_INIT_ORIGINAL,
            mask_fixup = True,
            guidance_preset = guidance,
            return_request = True
        )

        result_image = self.api.transform_and_generate(init, init_ops, generate_request)

        if self.color_match_image is None and args.color_coherence != 'None':
            self.color_match_image = result_image

        return result_image

    def _span_render(self, start: int, end: int, seed: int, prev_frame: Optional[np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
        args = self.args

        def apply_xform(frame: np.ndarray, xform: matrix.Matrix, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
            args, frame_args = self.args, self.frame_args
            if args.animation_mode == '2D':
                xform = to_3x3(xform)
                frames, masks = self.api.transform([frame], resample_op(args.border, xform, export_mask=True))
            else:
                fov = frame_args.fov_series[frame_idx]
                depth_blur = int(frame_args.depth_blur_series[frame_idx])
                depth_warp = frame_args.depth_warp_series[frame_idx]
                projection = matrix.projection_fov(math.radians(fov), 1.0, args.near_plane, args.far_plane)                
                wvp = matrix.multiply(projection, xform)
                depth_calc = depthcalc_op(args.depth_model_weight, depth_blur)
                resample = resample_op(args.border, wvp, projection, depth_warp=depth_warp, export_mask=True)
                frames, masks = self.api.transform_resample_3d([frame], depth_calc, resample)
            return frames[0], masks[0]

        if prev_frame is None:
            prev_frame = self._span_render_frame(start, seed, None)

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
            forward_frames[-1] = self.inpaint_frame(end-1, forward_frames[-1], forward_masks[-1])

        # run diffusion on top of the final result to allow content to evolve over time
        strength = max(0.0, self.frame_args.strength_series[end-1])
        if strength < 1.0:
            final_frame = self._span_render_frame(end-1, seed, forward_frames[-1])
        else:
            final_frame = forward_frames[-1]

        # go backwards through the frames in the span        
        backward_frames, backward_masks = [final_frame], [np.full_like(forward_masks[-1], 255)]
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
            backward_frames[0] = self.inpaint_frame(start, backward_frames[0], backward_masks[0])

        # yield the final frames blending from forward to backward
        for idx, (frame_fwd, frame_bwd) in enumerate(zip(forward_frames, backward_frames)):
            t = (idx) / max(1, end-start-1)
            fwd_fill = image_mix(frame_bwd, frame_fwd, mask_erode_blur(forward_masks[idx], 8, 8))
            bwd_fill = image_mix(frame_fwd, frame_bwd, mask_erode_blur(backward_masks[idx], 8, 8))
            blended = self.api.interpolate([fwd_fill, bwd_fill], [t], interp_mode_from_str(args.cadence_interp))[0]
            yield start+idx, blended

    def _spans_render(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = self.start_frame_idx
        seed = self.args.seed
        prev_frame: np.ndarray = None
        while frame_idx < self.args.max_frames:
            # determine how many frames the span will process together
            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_series[frame_idx]))
            if frame_idx + diffusion_cadence > self.args.max_frames:
                diffusion_cadence = self.args.max_frames - frame_idx

            # render all frames in the span
            for idx, frame in self._span_render(frame_idx, frame_idx + diffusion_cadence, seed, prev_frame):
                yield idx, frame
                prev_frame = frame
                if not self.args.locked_seed:
                    seed += 1

            frame_idx += diffusion_cadence

import bisect
import cv2
import json
import logging
import numpy as np
import os
import param
import random

from collections import OrderedDict
from PIL import Image
from types import SimpleNamespace
from typing import Generator, List, Tuple, Union

from stability_sdk.client import (
    image_gen,
    image_inpaint,
    generation,
)

from stability_sdk.utils import (
    sampler_from_string,
    key_frame_inbetweens,
    key_frame_parse,
    guidance_from_string,
    image_mix,
    image_xform,
    warp2d_op,
    warp3d_op,
    colormatch_op,
    depthcalc_op,
    warpflow_op,
    blend_op,
    contrast_op,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


docstring_bordermode = ( 
    "Method that will be used to fill empty regions, e.g. after a rotation transform."
    "\n\t* reflect - Mirror pixels across the image edge to fill empty regions."
    "\n\t* replicate - Use closest pixel values (default)."
    "\n\t* wrap - Treat image borders as if they were connected, i.e. use pixels from left edge to fill empty regions touching the right edge."
    "\n\t* zero - Fill empty regions with black pixels."
)


from keyframed import Keyframed, to_keyframed
from dataclasses import dataclass

@dataclass
class Prompt:
    prompt: Union[str,Image.Image]
    weight_curve: Keyframed


@dataclass
class Prompts:
    prompts: List[Prompt]

    def __getitem__(self, k):
        prompts, weights = [], []
        for p in self.prompts:
            p_ = p.prompt
            w_ = p.weight_curve[k]
            if w_ != 0:
              prompts.append(p_)
              weights.append(w_)
        return prompts, weights


# to do: these defaults and bounds should be configured in a language agnostic way so they can be 
# shared across client libraries, front end, etc.
# https://param.holoviz.org/user_guide/index.html
# TO DO: "prompt" argument has a bit of a logical collision with animation prompts
class BasicSettings(param.Parameterized):
    #prompt  = param.String(default="A beautiful painting of yosemite national park, by Neil Gaiman", doc="A string")
    height  = param.Integer(default=512, doc="Output image dimensions. Will be resized to a multiple of 64.")
    width   = param.Integer(default=512, doc="Output image dimensions. Will be resized to a multiple of 64.")
    sampler = param.ObjectSelector(default='K_euler_ancestral', objects=["DDIM", "PLMS", "K_euler", "K_euler_ancestral", "K_heun", "K_dpm_2", "K_dpm_2_ancestral", "K_lms"])
    seed    = param.Integer(default=-1, doc="Provide a seed value for more deterministic behavior. Negative seed values will be replaced with a random seed (default).")
    cfg_scale = param.Number(default=7, softbounds=(0,20), doc="Classifier-free guidance scale. Strength of prompt influence on denoising process. `cfg_scale=0` gives unconditioned sampling.")
    clip_guidance = param.ObjectSelector(default='FastBlue', objects=["None", "Simple", "FastBlue", "FastGreen"], doc="CLIP-guidance preset.")
    init_image = param.String(default='', doc="Path to image. Height and width dimensions will be inherited from image.")
    ####
    # missing param: n_samples = param.Integer(1, bounds=(1,9))

class AnimationSettings(param.Parameterized):
    animation_mode = param.ObjectSelector(default='3D', objects=['2D', '3D', 'Video Input'])
    max_frames = param.Integer(default=60, doc="Force stop of animation job after this many frames are generated.")
    border = param.ObjectSelector(default='replicate', objects=['reflect', 'replicate', 'wrap', 'zero'], doc=docstring_bordermode)
    inpaint_border = param.Boolean(default=False, doc="Use inpainting on top of border regions. Defaults to False")
    interpolate_prompts = param.Boolean(default=False, doc="Smoothly interpolate prompts between keyframes. Defaults to False")
    locked_seed = param.Boolean(default=False)


# TO DO: ability to specify backfill/interpolation method for each parameter
# TO DO: ability to provide a function that returns a parameter value given some frame index
# TO DO: inherit from param.String to add validation to these things
# TO DO: should defaults be noop or opinionated? Maybe a separate object for opinionated defaults?
class KeyframedSettings(param.Parameterized):
    """
    See disco/deforum keyframing syntax, originally developed by Chigozie Nri
    General syntax: "<frameId>:(<valueAtFrame>), f2:(v2),f3:(v3)...." 
    Values between intermediate keyframes will be linearly interpolated by default to produce smooth transitions.
    For abrupt transitions, specify values at adjacent keyframes.
    """
    angle = param.String(default="0:(1)")
    zoom = param.String(default="0:(1)")
    translation_x = param.String(default="0:(0)")
    translation_y = param.String(default="0:(0)")
    translation_z = param.String(default="0:(1)")
    rotation_x = param.String(default="0:(0)", doc="Euler angle in radians")
    rotation_y = param.String(default="0:(0)", doc="Euler angle in radians")
    rotation_z = param.String(default="0:(0)", doc="Euler angle in radians")
    brightness_curve = param.String(default="0:(1.0)")
    contrast_curve = param.String(default="0:(1.0)")
    noise_curve = param.String(default="0:(0.0)")
    noise_scale_curve = param.String(default="0:(1.02)")
    steps_curve = param.String(default="0:(50)", doc="Diffusion steps")
    strength_curve = param.String(default="0:(0.65)", doc="Image Strength (of init image relative to the prompt). 0 for ignore init image and attend only to prompt, 1 would return the init image unmodified")


# should diffusion cadence be moved up to the keyframed settings?
# if not, maybe stuff like steps and strength should be moved elsewhere?
class CoherenceSettings(param.Parameterized):
    color_coherence = param.ObjectSelector(default='LAB', objects=['None', 'HSV', 'LAB', 'RGB'], doc="Color space that will be used for inter-frame color adjustments.")
    diffusion_cadence_curve = param.String(default="0:(4)", doc="One greater than the number of frames between diffusion operations. A cadence of 1 performs diffusion on each frame. Values greater than one will generate frames using interpolation methods.")


# TO DO: change to a generic `depth_weight` rather than specifying model name in the parameter
class DepthwarpSettings(param.Parameterized):
    #use_depth_warping = True #@param {type:"boolean"}
    midas_weight = param.Number(default=0.3, softbounds=(0,1), doc="Strength of depth model influence.")
    near_plane = param.Number(default=200, doc="Distance to nearest plane of camera view volume.")
    far_plane = param.Number(default=10000, doc="Distance to furthest plane of camera view volume.")
    fov_curve = param.String(default="0:(25)", doc="FOV angle of camera volume in degrees.")
    save_depth_maps = param.Boolean(default=False)


class VideoInputSettings(param.Parameterized):
    video_init_path = param.String(default="", doc="Path to video input")
    extract_nth_frame = param.Integer(default=1, bounds=(1,None), doc="Only use every Nth frame of the video")
    video_mix_in_curve = param.String(default="0:(0.02)")
    video_flow_warp = param.Boolean(default=True, doc="Whether or not to transfer the optical flow from the video to the generated animation as a warp effect.")


class AnimationArgs(
    BasicSettings,
    AnimationSettings,
    KeyframedSettings,
    CoherenceSettings,
    DepthwarpSettings,
    VideoInputSettings,
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

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


from pathlib import Path

class Animator:
    def __init__(
        self,
        stub: 'generation_grpc.GenerationServiceStub',
        animation_prompts: Keyframed,
        init_images: Keyframed=None,
        args=None,
        out_dir='.',
        #####
        # we shouldn't be treating these special. to do: more generic prompt input
        negative_prompt='',
        negative_prompt_weight=-1.0,
        #####
        resume: bool = False,
        transform_engine_id='transform-server-v1',
        inpaint_engine_id='stable-diffusion-v1-5',
        generate_engine_id='stable-diffusion-v1-5',
    ):
        self.animation_prompts = animation_prompts
        if init_images:
            self.init_images = init_images
        self.args = args
        self.color_match_image: np.ndarray = None
        self.diffusion_cadence_ofs: int = 0
        self.keyframe_values: List[int] = None
        self.out_dir: str = out_dir
        self.prior_frames: List[np.ndarray] = []
        self.negative_prompt: str = negative_prompt
        self.negative_prompt_weight: float = negative_prompt_weight
        self.start_frame_idx: int = 0
        self.video_prev_frame: np.ndarray = None
        self.video_reader = None

        self.stub = stub
        self.transform_engine_id = transform_engine_id
        self.inpaint_engine_id = inpaint_engine_id
        self.generate_engine_id = generate_engine_id

        self.setup_animation(resume)

    def get_animation_prompts_weights(self, frame_idx: int) -> Tuple[List[str], List[float]]:
        if isinstance(self.animation_prompts, Prompts):
            # patch in init_image, cause this isn't hacky at all
            if hasattr(self, 'init_images'):
                ims, wts = self.init_images[frame_idx]
                init_path = Path(ims[0])
                if init_path.exists():
                    self.prepare_init_image(fpath=init_path)
            return self.animation_prompts[frame_idx]
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

    def get_frame_filename(self, frame_idx, depth=False):
        prefix = "depth" if depth else "frame"
        return os.path.join(self.out_dir, f"{prefix}_{frame_idx:05d}.png")

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

    def prepare_init_image(self, fpath=None):
        if fpath is None:
            #fpath = self.args.get('init_image')
            fpath =  self.args.init_image
        if not fpath:
            return
        img = Image.open(fpath)
        self.args.height, self.args.width = img.size[0], img.size[1]
        p = self.prior_frames
        self.prior_frames = [img, img] + p 

    def setup_animation(self, resume):
        args = self.args

        # change request for random seed into explicit value so it is saved to settings
        if args.seed <= 0:
            args.seed = random.randint(0, 2**32 - 1)

        def curve_to_series(curve: str) -> List[float]:
            try:
                from keyframed import Keyframed
                print("using fancy keyframes")
                if isinstance(curve, Keyframed):
                    k = curve
                elif isinstance(curve, str):
                    k = Keyframed.from_string(curve)
                elif isinstance(curve, dict):
                    k = Keyframed(curve)
                if args.max_frames:
                    k.set_length(args.max_frames)
                return k
            except ImportError:
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
            noise_series = curve_to_series(args.noise_curve),
            noise_scale_series = curve_to_series(args.noise_scale_curve),
            steps_series = curve_to_series(args.steps_curve),
            strength_series = curve_to_series(args.strength_curve),
            diffusion_cadence_series = curve_to_series(args.diffusion_cadence_curve),
            fov_series = curve_to_series(args.fov_curve),
            video_mix_in_series = curve_to_series(args.video_mix_in_curve),
        ))

        if not isinstance(self.animation_prompts, Prompts):
            # to do: move this
            # prepare sorted list of key frames
            self.key_frame_values = sorted(list(self.animation_prompts.keys()))
            if self.key_frame_values[0] != 0:
                raise ValueError("First keyframe must be 0")
            if len(self.key_frame_values) != len(set(self.key_frame_values)):
                raise ValueError("Duplicate keyframes are not allowed!")

        # prepare video input
        video_in = args.video_init_path if args.animation_mode == 'Video Input' else None
        if video_in:
            self.load_video(video_in)
        
        # what it says
        self.prepare_init_image()

        # handle resuming animation from last frames of a previous run
        if resume:
            frames = [f for f in os.listdir(self.out_dir) if f.endswith(".png") and f.startswith("frame_")]
            self.start_frame_idx = len(frames)
            self.diffusion_cadence_ofs = self.start_frame_idx
            if self.start_frame_idx > 2:
                self.prior_frames = [
                    cv2.imread(self.get_frame_filename(self.start_frame_idx-2)),
                    cv2.imread(self.get_frame_filename(self.start_frame_idx-1))
                ]

    def load_video(self, video_in):
        self.video_reader = cv2.VideoCapture(video_in)
        if self.video_reader is not None:
            success, image = self.video_reader.read()
            if not success:
                raise Exception(f"Failed to read first frame from {video_in}")
            self.video_prev_frame = cv2.resize(image, (self.args.width, self.args.height), interpolation=cv2.INTER_LANCZOS4)
            self.prior_frames = [self.video_prev_frame, self.video_prev_frame]

    def build_prior_frame_transforms(self, frame_idx) -> List[generation.TransformOperation]:
        if not len(self.prior_frames):
            return []

        args = self.args
        frame_args = self.frame_args
        ops = []

        if args.save_depth_maps or args.animation_mode == '3D':
            ops.append(depthcalc_op(
                blend_weight=args.midas_weight,
                export=args.save_depth_maps,
            ))

        if args.animation_mode == '2D':
            ops.append(warp2d_op(
                border_mode=args.border,
                rotate=frame_args.angle_series[frame_idx], 
                scale=frame_args.zoom_series[frame_idx], 
                translate_x=frame_args.translation_x_series[frame_idx], 
                translate_y=frame_args.translation_y_series[frame_idx], 
            ))
        elif args.animation_mode == '3D':
            ops.append(warp3d_op(
                border_mode = args.border,
                translate_x = frame_args.translation_x_series[frame_idx],
                translate_y = frame_args.translation_y_series[frame_idx],
                translate_z = frame_args.translation_z_series[frame_idx],
                rotate_x = frame_args.rotation_x_series[frame_idx],
                rotate_y = frame_args.rotation_y_series[frame_idx],
                rotate_z = frame_args.rotation_z_series[frame_idx],
                near_plane = args.near_plane,
                far_plane = args.far_plane,
                fov = frame_args.fov_series[frame_idx],
            ))

        elif args.animation_mode == 'Video Input':
            for _ in range(args.extract_nth_frame):
                success, video_next_frame = self.video_reader.read()
            if success:
                video_next_frame = cv2.resize(
                    video_next_frame, 
                    (args.width, args.height), 
                    interpolation=cv2.INTER_LANCZOS4
                )
                if args.video_flow_warp:
                    ops.append(warpflow_op(
                        prev_frame=self.video_prev_frame,
                        next_frame=video_next_frame,
                    ))
                self.video_prev_frame = video_next_frame
                self.color_match_image = video_next_frame

        return ops

    def render(self) -> Generator[Image.Image, None, None]:
        args = self.args
        seed = args.seed

        for frame_idx in range(self.start_frame_idx, args.max_frames):

            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_series[frame_idx]))
            steps = int(self.frame_args.steps_series[frame_idx])

            # fetch set of prompts and weights for this frame
            prompts, weights = self.get_animation_prompts_weights(frame_idx)
            if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
                prompts.append(self.negative_prompt)
                weights.append(-abs(self.negative_prompt_weight))

            # transform prior frames
            ops = self.build_prior_frame_transforms(frame_idx)
            if len(ops):
                self.prior_frames, mask = image_xform(self.stub, self.prior_frames, ops, self.transform_engine_id)

                if args.save_depth_maps:
                    depth_map = self.prior_frames.pop(0)
                    cv2.imwrite(self.get_frame_filename(frame_idx, depth=True), depth_map)

                if args.inpaint_border and mask is not None:
                    for i in range(len(self.prior_frames)):
                        self.prior_frames[i] = image_inpaint(
                            stub=self.stub,
                            image=self.prior_frames[i],
                            mask=mask,
                            prompts=prompts,
                            weights=weights,
                            steps=steps//2,
                            seed=seed,
                            cfg_scale=args.cfg_scale,
                            #blur_ks=...,
                            engine_id=self.inpaint_engine_id,
                        )

            # either run diffusion or emit an inbetween frame
            if (frame_idx - self.diffusion_cadence_ofs) % diffusion_cadence == 0:
                strength = self.frame_args.strength_series[frame_idx]

                # apply additional noising and color matching to previous frame to use as init
                init_image = self.prior_frames[-1] if len(self.prior_frames) and strength > 0 else None
                if init_image is not None:
                    noise = self.frame_args.noise_series[frame_idx]
                    brightness = self.frame_args.brightness_series[frame_idx]
                    contrast = self.frame_args.contrast_series[frame_idx]
                    mix_in = self.frame_args.video_mix_in_series[frame_idx]

                    init_ops = []
                    if args.color_coherence != 'None' and self.color_match_image is not None:                    
                        init_ops.append(colormatch_op(
                            palette_image=self.color_match_image,
                            color_mode=args.color_coherence,
                        ))
                    if mix_in > 0 and self.video_prev_frame is not None:
                        init_ops.append(blend_op(
                            amount=mix_in, 
                            target=self.video_prev_frame
                        ))
                    if brightness != 1.0 or contrast != 1.0:
                        init_ops.append(contrast_op(
                            brightness=brightness,
                            contrast=contrast,
                        ))
                    if noise > 0:
                        init_ops.append(generation.TransformOperation(
                            add_noise=generation.TransformAddNoise(amount=noise, seed=seed)
                        ))
                    if len(init_ops):
                        init_image = image_xform(self.stub, [init_image], init_ops, self.transform_engine_id)[0][0]

                # generate the next frame
                sampler = sampler_from_string(args.sampler.lower())
                guidance = guidance_from_string(args.clip_guidance)
                noise_scale = self.frame_args.noise_scale_series[frame_idx]
                image = image_gen(
                    self.stub, 
                    args.width,
                    args.height, 
                    prompts,
                    weights, 
                    steps,
                    seed,
                    args.cfg_scale,
                    sampler, 
                    init_image,
                    strength,
                    init_noise_scale=noise_scale, 
                    guidance_preset=guidance,
                )

                if self.color_match_image is None and args.color_coherence != 'None':
                    self.color_match_image = image
                if not len(self.prior_frames):
                    self.prior_frames = [image, image]
                                
                cv2.imwrite(self.get_frame_filename(frame_idx), self.prior_frames[1])
                yield cv2_to_pil(self.prior_frames[1])
                self.prior_frames[0] = self.prior_frames[1]
                self.prior_frames[1] = image
                self.diffusion_cadence_ofs = frame_idx
            else:
                # smoothly blend between prior frames
                tween = ((frame_idx - self.diffusion_cadence_ofs) % diffusion_cadence) / float(diffusion_cadence)
                t = image_mix(self.prior_frames[0], self.prior_frames[1], tween)
                cv2.imwrite(self.get_frame_filename(frame_idx), t)
                yield cv2_to_pil(t)

            if not args.locked_seed:
                seed += 1

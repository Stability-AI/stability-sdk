import bisect
import cv2
import datetime
import json
import logging
import os
import random

from base64 import b64encode
from collections import OrderedDict
from IPython import display # this should be conditional on notebook env
import numpy as np
import param
import pandas as pd
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from typing import List, Tuple

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


def display_frame(image: np.ndarray):
    display.clear_output(wait=True)
    display.display(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


docstring_bordermode = ( 
    "Method that will be used to fill empty regions, e.g. after a rotation transform."
    "\n\t* reflect - Mirror pixels across the image edge to fill empty regions."
    "\n\t* replicate - Use closest pixel values (default)."
    "\n\t* wrap - Treat image borders as if they were connected, i.e. use pixels from left edge to fill empty regions touching the right edge."
    "\n\t* zero - Fill empty regions with black pixels."
)

# to do: these defaults and bounds should be configured in a language agnostic way so they can be 
# shared across client libraries, front end, etc.
# https://param.holoviz.org/user_guide/index.html
# TO DO: "prompt" argument has a bit of a logical collision with animation prompts
class BasicSettings(param.Parameterized):
    #prompt  = param.String(default="A beautiful painting of yosemite national park, by Neil Gaiman", doc="A string")
    height  = param.Integer(default=512, bounds=(256, 1024), doc="Output image dimensions. Will be resized to a multiple of 64.")
    width   = param.Integer(default=512, bounds=(256, 1024), doc="Output image dimensions. Will be resized to a multiple of 64.")
    sampler = param.ObjectSelector(default='K_euler_ancestral', objects=["DDIM", "PLMS", "K_euler", "K_euler_ancestral", "K_heun", "K_dpm_2", "K_dpm_2_ancestral", "K_lms"])
    seed    = param.Integer(default=-1, doc="Provide a seed value for more deterministic behavior. Negative seed values will be replaced with a random seed (default).")
    cfg_scale = param.Number(default=7, softbounds=(0,20), doc="Classifier-free guidance scale. Strength of prompt influence on denoising process. `cfg_scale=0` gives unconditioned sampling.")
    clip_guidance = param.ObjectSelector(default='FastBlue', objects=["None", "Simple", "FastBlue", "FastGreen"], doc="CLIP-guidance preset.")
    ####
    # missing param: n_samples = param.Integer(1, bounds=(1,9))

class AnimationSettings(param.Parameterized):
    animation_mode = param.ObjectSelector(default='3D', objects=['2D', '3D', 'Video Input'])
    max_frames = param.Integer(default=60, doc="Force stop of animation job after this many frames are generated.")
    border = param.ObjectSelector(default='replicate', objects=['reflect', 'replicate', 'wrap', 'zero'], doc=docstring_bordermode)
    # this should really be a border mode
    inpaint_border = param.Boolean(default=False, doc="Use diffusion inpainting to backfill empty border regions. Overrides `border`, defaults to False")
    interpolate_prompts = param.Boolean(default=False)
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
    # do these camera parameters need to be integer valued?
    near_plane = param.Number(default=200, doc="Distance to nearest plane of camera view volume.")
    far_plane = param.Number(default=10000, doc="Distance to furthest plane of camera view volume.")
    fov_curve = param.String(default="0:(25)", doc="FOV angle of camera volume in degrees.")
    save_depth_maps = param.Boolean(default=False)


class VideoInputSettings(param.Parameterized):
    video_init_path = param.String(default="./video_in.mp4", doc="Path to video input")
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


class Animator:
    def __init__(
        self,
        animation_prompts,
        args=None,
        out_dir='.',
        #####
        # we shouldn't be treating these special. to do: more generic prompt input
        negative_prompt='',
        negative_prompt_weight=0.0,
        #####
        transform_engine_id='transform-server-v1',
        inpaint_engine_id='stable-diffusion-v1-5',
        generate_engine_id='stable-diffusion-v1-5',
    ):
        self.animation_prompts = animation_prompts
        self.args = args
        self.out_dir = out_dir
        self.negative_prompt = negative_prompt
        self.negative_prompt_weight = negative_prompt_weight
        self.transform_engine_id = transform_engine_id
        self.inpaint_engine_id = inpaint_engine_id
        self.generate_engine_id = generate_engine_id

        self.video_prev_frame = None
        self.setup_animation()

    def get_animation_prompts_weights(
        self,
        frame_idx: int,
        key_frame_values: List[int],
        interp: bool
    ) -> Tuple[List[str], List[float]]:
        idx = bisect.bisect_right(key_frame_values, frame_idx)
        prev, next = idx - 1, idx
        if not interp:
            return [self.animation_prompts[key_frame_values[min(len(key_frame_values)-1, prev)]]], [1.0]
        elif next == len(key_frame_values):
            return [self.animation_prompts[key_frame_values[-1]]], [1.0]
        else:
            tween = (frame_idx - key_frame_values[prev]) / (key_frame_values[next] - key_frame_values[prev])
            return [self.animation_prompts[key_frame_values[prev]], self.animation_prompts[key_frame_values[next]]], [1.0 - tween, tween]


    def save_settings(self):
            # save settings for the animation
            timestring = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            settings_filename = os.path.join(self.out_dir, f"{timestring}_settings.txt")
            with open(settings_filename, "w+", encoding="utf-8") as f:
                save_dict = args2dict(self.args)
                for k in ['angle', 'zoom', 'translation_x', 'translation_y', 'translation_z', 'rotation_x', 'rotation_y', 'rotation_z']:
                    save_dict.move_to_end(k, last=True)
                save_dict['animation_prompts'] = self.animation_prompts
                save_dict['negative_prompt'] = self.negative_prompt
                save_dict['negative_prompt_weight'] = self.negative_prompt_weight
                json.dump(save_dict, f, ensure_ascii=False, indent=4)

    def setup_animation(self):
        args = self.args

        # change request for random seed into explicit value so it is saved to settings
        if args.seed <= 0:
            args.seed = random.randint(0, 2**32 - 1)

        def curve_to_series(curve: str) -> List[float]:
            return key_frame_inbetweens(key_frame_parse(curve), args.max_frames)    

        self.frame_args = SimpleNamespace(**dict(
            angle_series = curve_to_series(args.angle)
            ,zoom_series = curve_to_series(args.zoom)
            ,translation_x_series = curve_to_series(args.translation_x)
            ,translation_y_series = curve_to_series(args.translation_y)
            ,translation_z_series = curve_to_series(args.translation_z)
            ,rotation_x_series = curve_to_series(args.rotation_x)
            ,rotation_y_series = curve_to_series(args.rotation_y)
            ,rotation_z_series = curve_to_series(args.rotation_z)
            ,brightness_series = curve_to_series(args.brightness_curve)
            ,contrast_series = curve_to_series(args.contrast_curve)
            ,noise_series = curve_to_series(args.noise_curve)
            ,noise_scale_series = curve_to_series(args.noise_scale_curve)
            ,steps_series = curve_to_series(args.steps_curve)
            ,strength_series = curve_to_series(args.strength_curve)
            ,diffusion_cadence_series = curve_to_series(args.diffusion_cadence_curve)
            ,fov_series = curve_to_series(args.fov_curve)
            ,video_mix_in_series = curve_to_series(args.video_mix_in_curve)
        ))

        # prepare sorted list of key frames
        key_frame_values = sorted(list(self.animation_prompts.keys()))
        if key_frame_values[0] != 0:
            raise ValueError("First keyframe must be 0")
        if len(key_frame_values) != len(set(key_frame_values)):
            raise ValueError("Duplicate keyframes are not allowed!")
        self.keyframe_values = key_frame_values

        self.prior_frames = []
        self.video_reader = None
        self.video_prev_frame = None
        video_in = args.video_init_path if args.animation_mode == 'Video Input' else None
        if video_in:
            self.load_video(video_in)

    def load_video(self, video_in):
        # diffusion performed every N frames. two prior diffused frames
        # are transformed and blended between to produce each output frame
        #diffusion_cadence_ofs = 0

        # load input video
        self.video_reader = cv2.VideoCapture(video_in)
        #video_extract_nth = args.extract_nth_frame
        if self.video_reader is not None:
            success, image = self.video_reader.read()
            if not success:
                raise Exception(f"Failed to read first frame from {video_in}")
            video_prev_frame = cv2.resize(image, (self.args.W, self.args.H), interpolation=cv2.INTER_LANCZOS4)
            self.prior_frames = [video_prev_frame, video_prev_frame]
            self.video_prev_frame = video_prev_frame

    def build_prior_frame_transforms(
        self,
        frame_idx,
        color_match_image,
    ):
        args = self.args
        frame_args = self.frame_args
        ops = []
        if args.save_depth_maps or args.animation_mode == '3D':
            op=depthcalc_op(
                blend_weight=args.midas_weight,
                export=args.save_depth_maps,
                )
            ops.append(op)

        if args.animation_mode == '2D':
            op = warp2d_op(
                border_mode=args.border,
                rotate=frame_args.angle_series[frame_idx], 
                scale=frame_args.zoom_series[frame_idx], 
                translate_x=frame_args.translation_x_series[frame_idx], 
                translate_y=frame_args.translation_y_series[frame_idx], 
            )
            ops.append(op)
        elif args.animation_mode == '3D':
            op = warp3d_op(
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
                )
            ops.append(op)

        elif args.animation_mode == 'Video Input':
            video_extract_nth = args.extract_nth_frame
            for i in range(video_extract_nth):
                success, video_next_frame = self.video_reader.read()
            if success:
                video_next_frame = cv2.resize(video_next_frame, (args.W, args.H), interpolation=cv2.INTER_LANCZOS4)
                if args.video_flow_warp:
                    op = warpflow_op(
                        prev_frame=self.video_prev_frame,
                        next_frame=video_next_frame,
                    )
                    ops.append(op)
                self.video_prev_frame = video_next_frame
                color_match_image = video_next_frame

        return ops, color_match_image

    def render_animation(
        self,
        stub,
        args=None,
        out_dir=None
    ):

        if not args:
            args = self.args
        if not out_dir:
            out_dir = self.out_dir
        key_frame_values = self.keyframe_values
        seed = args.seed
        color_match_image = None # optional target for color matching
        inpaint_mask = None      # optional mask of revealed areas
        diffusion_cadence_ofs = 0 # diffusion performed every N frames.

        #video_reader = self.video_reader
        prior_frames = self.prior_frames
        
        # to facilitate resuming, we need some sort of self.generate_frame(frame_idx) function that gets looped over here
        for frame_idx in tqdm(range(args.max_frames)):


            diffusion_cadence = max(1, int(self.frame_args.diffusion_cadence_series[frame_idx]))
            steps = int(self.frame_args.steps_series[frame_idx])

            # fetch set of prompts and weights for this frame
            prompts, weights = self.get_animation_prompts_weights(frame_idx, key_frame_values, interp=args.interpolate_prompts)
            if len(self.negative_prompt) and self.negative_prompt_weight != 0.0:
                prompts.append(self.negative_prompt)
                weights.append(-abs(self.negative_prompt_weight))

            ops=[]
            if len(prior_frames):
                (ops, color_match_image) = self.build_prior_frame_transforms(
                    frame_idx=frame_idx,
                    color_match_image=color_match_image,
                )

            if len(ops):
                prior_frames, mask = image_xform(stub, prior_frames, ops, self.transform_engine_id)
                inpaint_mask = mask if args.inpaint_border else None

                depth_map = prior_frames.pop(0) if len(prior_frames) == 3 else None
                if depth_map is not None and args.save_depth_maps:
                    cv2.imwrite(os.path.join(out_dir, f"depth_{frame_idx:05d}.png"), depth_map)

                if inpaint_mask is not None:
                    for i in range(len(prior_frames)):
                        prior_frames[i] = image_inpaint(
                            stub=stub,
                            image=prior_frames[i],
                            mask=inpaint_mask,
                            prompts=prompts,
                            weights=weights,
                            steps=steps//2,
                            seed=seed,
                            cfg_scale=args.cfg_scale,
                            #blur_ks=...,
                            engine_id=self.inpaint_engine_id,
                        )
                    inpaint_mask = None

            # either run diffusion or emit an inbetween frame
            if (frame_idx-diffusion_cadence_ofs) % diffusion_cadence == 0: # DMARX: i don't thinks subtracting the 'offset' is needed here, it's a modulo already

                # didn't we already do this in self.build_prior_frame_transforms ?
                # TO DO: add a color match op after each inpaint step.
                # maybe this just needs to be a subprocedure of image_inpaint?
                if inpaint_mask is not None:
                    prior_frames[-1] = image_inpaint(
                        stub=stub,
                        image=prior_frames[-1],
                        mask=inpaint_mask,
                        prompts=prompts,
                        weights=weights,
                        steps=steps//2,
                        seed=seed,
                        cfg_scale=args.cfg_scale,
                        #blur_ks=...,
                        engine_id=self.inpaint_engine_id,
                    )
                    inpaint_mask = None
                strength = self.frame_args.strength_series[frame_idx]

                # apply additional noising and color matching to previous frame to use as init
                init_image = prior_frames[-1] if len(prior_frames) and strength > 0 else None
                if init_image is not None:
                    noise = self.frame_args.noise_series[frame_idx]
                    brightness = self.frame_args.brightness_series[frame_idx]
                    contrast = self.frame_args.contrast_series[frame_idx]
                    mix_in = self.frame_args.video_mix_in_series[frame_idx]

                    ops = [] # if we previously populated ops before, looks like we're going to overwrite it here. is that on purpose? guessing it's not... I think maybe this should have a different name to distinguish it as init_image specific ops.
                    if args.color_coherence != 'None' and color_match_image is not None:                    
                        op = colormatch_op(
                            palette_image=color_match_image,
                            color_mode=args.color_coherence,
                        )
                        ops.append(op)
                    if mix_in > 0 and self.video_prev_frame is not None:
                        op = blend_op(
                            amount=mix_in, 
                            target=self.video_prev_frame)
                        ops.append(op)
                    if brightness != 1.0 or contrast != 1.0:
                        op=contrast_op(
                            brightness=brightness,
                            contrast=contrast,
                        )
                        ops.append(op)
                    if noise > 0:
                        ops.append(generation.TransformOperation(add_noise=generation.TransformAddNoise(amount=noise, seed=seed)))
                    if len(ops):
                        init_image = image_xform(stub, [init_image], ops, self.transform_engine_id)[0][0]

                # generate the next frame
                sampler = sampler_from_string(args.sampler.lower())
                guidance = guidance_from_string(args.clip_guidance)
                noise_scale = self.frame_args.noise_scale_series[frame_idx]
                image = image_gen(
                    stub, 
                    args.W,
                    args.H, 
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

                if color_match_image is None:
                    color_match_image = image
                if not len(prior_frames):
                    prior_frames = [image, image]
                
                cv2.imwrite(os.path.join(out_dir, f'frame_{frame_idx:05}.png'), prior_frames[1])
                display_frame(prior_frames[1])
                prior_frames[0] = prior_frames[1]
                prior_frames[1] = image
                diffusion_cadence_ofs = frame_idx
            else:
                # smoothly blend between prior frames
                tween = ((frame_idx-diffusion_cadence_ofs) % diffusion_cadence) / float(diffusion_cadence)
                t = image_mix(prior_frames[0], prior_frames[1], tween)
                cv2.imwrite(os.path.join(out_dir, f'frame_{frame_idx:05}.png'), t)
                display_frame(t)

            if not args.locked_seed:
                seed += 1
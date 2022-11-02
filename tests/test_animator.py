from stability_sdk.animation import Animator, AnimationArgs
from types import SimpleNamespace

import pytest

#def test_init_args():
#    AnimationArgs()

def test_init_animator(default_anim_args):
    Animator(
        args=default_anim_args,
        #out_dir='.',
        #animation_prompts=None,
        #negative_prompt=None,
        #negative_prompt_weight=None
    )

def test_save_settings(default_anim_args):
    artist=Animator(args=default_anim_args)
    artist.save_settings()

def test_get_weights(default_anim_args):
    artist = Animator(args=default_anim_args)
    artist.get_animation_prompts_weights(
        frame_idx=0,
        key_frame_values=[0],
        interp=False
    )

def test_load_video(default_anim_args, vidpath):
    artist = Animator(args=default_anim_args)
    artist.load_video(video_in=vidpath)
    assert len(artist.prior_frames) > 0
    assert artist.video_prev_frame is not None

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D'])
def test_build_prior_txs(default_anim_args, vidpath, animation_mode):
    default_anim_args.animation_mode=animation_mode
    artist = Animator(args=default_anim_args)
    artist.load_video(video_in=vidpath) # just to populate prior frames
    outv = artist.build_prior_frame_transforms(frame_idx=0, color_match_image=artist.prior_frames[0])
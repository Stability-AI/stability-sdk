from stability_sdk.animation import Animator, AnimationArgs
from types import SimpleNamespace

from stability_sdk import client

import grpc
import pytest

#def test_init_args():
#    AnimationArgs()

animation_prompts={0:"foo bar"}

def test_init_animator(default_anim_args):
    Animator(
        args=default_anim_args,
        #out_dir='.',
        animation_prompts=animation_prompts,
        #negative_prompt=None,
        #negative_prompt_weight=None
    )

def test_init_animator_prompts_notoptional(default_anim_args):
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'animation_prompts'"):
        Animator(
            args=default_anim_args,
        )

def test_save_settings(default_anim_args):
    artist=Animator(args=default_anim_args, animation_prompts=animation_prompts)
    artist.save_settings()

def test_get_weights(default_anim_args):
    artist = Animator(args=default_anim_args, animation_prompts=animation_prompts)
    artist.get_animation_prompts_weights(
        frame_idx=0,
        key_frame_values=[0],
        interp=False
    )

def test_load_video(default_anim_args, vidpath):
    artist = Animator(args=default_anim_args, animation_prompts=animation_prompts)
    artist.load_video(video_in=vidpath)
    assert len(artist.prior_frames) > 0
    assert artist.video_prev_frame is not None
    assert all([v is not None for v in artist.prior_frames]) 

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D'])
def test_build_prior_txs(default_anim_args, vidpath, animation_mode):
    default_anim_args.animation_mode=animation_mode
    artist = Animator(args=default_anim_args, animation_prompts=animation_prompts)
    artist.load_video(video_in=vidpath) # just to populate prior frames
    outv = artist.build_prior_frame_transforms(frame_idx=0, color_match_image=artist.prior_frames[0])

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D'])
def test_render(default_anim_args, animation_mode, grpc_addr, grpc_server, vidpath):
    default_anim_args.animation_mode=animation_mode
    artist = Animator(args=default_anim_args, animation_prompts=animation_prompts)
    stub = client.open_channel(grpc_addr[0])
    if animation_mode == 'Video Input':
        artist.load_video(video_in=vidpath)
        print(len(artist.prior_frames))
        print([type(p) for p in artist.prior_frames])
    # to do: better mocking
    with pytest.raises(grpc._channel._MultiThreadedRendezvous):
        artist.render_animation(stub=stub)
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
        stub=None,
        args=default_anim_args,
        animation_prompts=animation_prompts,
        #out_dir='.',
        #negative_prompt=None,
        #negative_prompt_weight=None
    )

def test_init_animator_prompts_notoptional(default_anim_args):
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'animation_prompts'"):
        Animator(
            stub=None,
            args=default_anim_args,
        )

def test_save_settings(default_anim_args):
    artist=Animator(stub=None, args=default_anim_args, animation_prompts=animation_prompts)
    artist.save_settings("settings.txt")

def test_get_weights(default_anim_args):
    artist = Animator(stub=None, args=default_anim_args, animation_prompts=animation_prompts)
    artist.get_animation_prompts_weights(frame_idx=0)

def test_load_video(default_anim_args, vidpath):
    artist = Animator(stub=None, args=default_anim_args, animation_prompts=animation_prompts)
    artist.load_video(video_in=vidpath)
    assert len(artist.prior_frames) > 0
    assert artist.video_prev_frame is not None
    assert all([v is not None for v in artist.prior_frames]) 

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D'])
def test_build_prior_txs(default_anim_args, vidpath, animation_mode):
    default_anim_args.animation_mode=animation_mode
    default_anim_args.video_init_path = vidpath
    artist = Animator(stub=None, args=default_anim_args, animation_prompts=animation_prompts)
    artist.load_video(video_in=vidpath) # just to populate prior frames
    ops = artist.build_prior_frame_transforms(frame_idx=0)

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D'])
def test_render(default_anim_args, animation_mode, grpc_addr, grpc_server, vidpath):
    default_anim_args.animation_mode=animation_mode
    default_anim_args.video_init_path = vidpath
    stub = client.open_channel(grpc_addr[0])
    artist = Animator(stub=stub, args=default_anim_args, animation_prompts=animation_prompts)
    if animation_mode == 'Video Input':
        artist.load_video(video_in=vidpath)
        print(len(artist.prior_frames))
        print([type(p) for p in artist.prior_frames])
    # to do: better mocking
    image = artist.render()
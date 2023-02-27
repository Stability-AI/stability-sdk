from pathlib import Path
from types import SimpleNamespace

from stability_sdk import client
from stability_sdk.animation import Animator, AnimationArgs

import grpc
import pytest

from .test_client import MockStub

animation_prompts={0:"foo bar"}

def test_init_animator():
    Animator(
        api=client.Api(stub=MockStub()),
        args=AnimationArgs(),
        animation_prompts=animation_prompts,
        #out_dir='.',
        #negative_prompt=None,
        #negative_prompt_weight=None
    )

def test_init_animator_prompts_notoptional():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'animation_prompts'"):
        Animator(
            api=client.Api(stub=MockStub()),
            args=AnimationArgs(),
        )

def test_save_settings():
    artist=Animator(api=client.Api(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    artist.save_settings("settings.txt")

def test_get_weights():
    artist = Animator(api=client.Api(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    artist.get_animation_prompts_weights(frame_idx=0)

def test_load_video(vidpath):
    args = AnimationArgs()
    args.animation_mode = 'Video Input'
    args.video_init_path = vidpath
    artist = Animator(api=client.Api(stub=MockStub()), args=args, animation_prompts=animation_prompts)
    assert len(artist.prior_frames) > 0
    assert artist.video_prev_frame is not None
    assert all([v is not None for v in artist.prior_frames]) 

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D warp','3D render'])
def test_render(animation_mode, grpc_addr, grpc_server, vidpath):
    args = AnimationArgs()
    args.animation_mode = animation_mode
    args.video_init_path = vidpath
    channel = client.open_channel(grpc_addr[0])
    artist = Animator(api=client.Api(channel=channel), args=args, animation_prompts=animation_prompts)
    if animation_mode == 'Video Input':
        print(len(artist.prior_frames))
        print([type(p) for p in artist.prior_frames])
    # to do: better mocking
    image = artist.render()


def test_init_image_none():
    artist = Animator(api=client.Api(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    assert len(artist.prior_frames) == 0

def test_init_image_from_args(impath):
    args = AnimationArgs()
    args.init_image = impath
    artist = Animator(api=client.Api(stub=MockStub()), args=args, animation_prompts=animation_prompts)
    assert len(artist.prior_frames) == 1

def test_init_image_from_input(impath):
    artist = Animator(api=client.Api(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    print(artist.prior_frames)
    assert len(artist.prior_frames) == 0
    artist.load_init_image()
    assert len(artist.prior_frames) == 0
    assert Path(impath).exists()
    artist.load_init_image(impath)
    assert len(artist.prior_frames) == 1
    artist.set_cadence_mode(True)
    assert len(artist.prior_frames) == 2
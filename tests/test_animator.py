import pytest

from pathlib import Path

from stability_sdk.animation import Animator, AnimationArgs
from stability_sdk.api import Context

from .test_api import MockStub

animation_prompts={0:"foo bar"}

def test_init_animator():
    Animator(
        Context(stub=MockStub()),
        args=AnimationArgs(),
        animation_prompts=animation_prompts,
    )

def test_init_animator_prompts_notoptional():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'animation_prompts'"):
        Animator(
            Context(stub=MockStub()),
            args=AnimationArgs(),
        )

def test_save_settings():
    animator = Animator(Context(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    animator.save_settings("settings.txt")

def test_get_weights():
    animator = Animator(Context(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    animator.get_animation_prompts_weights(frame_idx=0)

def test_load_video(vidpath):
    args = AnimationArgs()
    args.animation_mode = 'Video Input'
    args.video_init_path = vidpath
    animator = Animator(Context(stub=MockStub()), args=args, animation_prompts=animation_prompts)
    assert len(animator.prior_frames) > 0
    assert animator.video_prev_frame is not None
    assert all([v is not None for v in animator.prior_frames]) 

@pytest.mark.parametrize('animation_mode', ['Video Input','2D','3D warp','3D render'])
def test_render(animation_mode, vidpath):
    args = AnimationArgs()
    args.animation_mode = animation_mode
    args.video_init_path = vidpath
    animator = Animator(Context(stub=MockStub()), args=args, animation_prompts=animation_prompts)
    if animation_mode == 'Video Input':
        print(len(animator.prior_frames))
        print([type(p) for p in animator.prior_frames])
    _ = animator.render()

def test_init_image_none():
    animator = Animator(Context(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    assert len(animator.prior_frames) == 0

def test_init_image_from_args(impath):
    args = AnimationArgs()
    args.init_image = impath
    animator = Animator(Context(stub=MockStub()), args=args, animation_prompts=animation_prompts)
    assert len(animator.prior_frames) == 1

def test_init_image_from_input(impath):
    animator = Animator(Context(stub=MockStub()), args=AnimationArgs(), animation_prompts=animation_prompts)
    print(animator.prior_frames)
    assert len(animator.prior_frames) == 0
    animator.load_init_image()
    assert len(animator.prior_frames) == 0
    assert Path(impath).exists()
    animator.load_init_image(impath)
    assert len(animator.prior_frames) == 1
    animator.set_cadence_mode(True)
    assert len(animator.prior_frames) == 2
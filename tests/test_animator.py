from stability_sdk.animation import Animator, AnimationArgs
from types import SimpleNamespace

def test_init():
    Animator(
        #args=SimpleNamespace(),
        #out_dir='.',
        #animation_prompts=None,
        #negative_prompt=None,
        #negative_prompt_weight=None
    )

def test_init_args():
    AnimationArgs()
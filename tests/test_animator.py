from stability_sdk.animation import Animator, AnimationArgs
from types import SimpleNamespace

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
    Animator(args=default_anim_args).save_settings()
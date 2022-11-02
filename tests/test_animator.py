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
    artist=Animator(args=default_anim_args)
    artist.save_settings()

def test_get_weights(default_anim_args):
    artist = Animator(args=default_anim_args)
    artist.get_animation_prompts_weights(
        frame_idx=0,
        key_frame_values=[0],
        interp=False
    )
from typing import ByteString

import pytest

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk.utils import (
    _2d_only_modes,
    BORDER_MODES_2D,
    BORDER_MODES_3D,
    COLOR_SPACES,
    GUIDANCE_PRESETS,
    SAMPLERS,
    border_mode_from_str_2d,
    border_mode_from_str_3d,
    color_match_from_string,
    sampler_from_string,
    guidance_from_string,
    truncate_fit,
    ########
    image_mix,
    image_to_jpg_bytes,
    image_to_png_bytes,
    image_to_prompt,
    image_xform,
    #########
    key_frame_inbetweens,
    key_frame_parse,
    #########
    warp2d_op,
    warp3d_op,
    colormatch_op,
    depthcalc_op,
    warpflow_op,
)

@pytest.mark.parametrize("border", BORDER_MODES_2D.keys())
def test_border_mode_from_str_2d_valid(border):
    border_mode_from_str_2d(s=border)
    assert True

def test_border_mode_from_str_2d_invalid():
    with pytest.raises(ValueError, match="invalid 2d border mode"):
        border_mode_from_str_2d(s='not a real border mode')


@pytest.mark.parametrize("border", BORDER_MODES_3D.keys())
def test_border_mode_from_str_3d_valid(border):
    border_mode_from_str_3d(s=border)
    assert True

def test_border_mode_from_str_3d_invalid():
    with pytest.raises(ValueError, match="invalid 3d border mode"):
        border_mode_from_str_3d(s='not a real border mode')


@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
def test_sampler_from_str_valid(sampler_name):
    sampler_from_string(s=sampler_name)
    assert True

def test_sampler_from_str_invalid():
    with pytest.raises(ValueError, match="invalid sampler"):
        sampler_from_string(s='not a real sampler')

@pytest.mark.parametrize("preset_name", GUIDANCE_PRESETS.keys())
def test_guidance_from_string_valid(preset_name):
    guidance_from_string(s=preset_name)
    assert True

def test_guidance_from_string_invalid():
    with pytest.raises(ValueError, match="invalid guidance preset"):
        guidance_from_string(s='not a real preset')

@pytest.mark.parametrize("color_space_name", COLOR_SPACES.keys())
def test_color_match_from_string_valid(color_space_name):
    color_match_from_string(s=color_space_name)
    assert True

def test_color_match_from_string_invalid():
    with pytest.raises(ValueError, match="invalid color space"):
        color_match_from_string(s='not a real colorspace')


####################################
# to do: pytest.mark.paramaterized #

def test_truncate_fit0():
    outv = truncate_fit(
        prefix='foo_', 
        prompt='bar', 
        ext='.baz', 
        ts=12345678,
        idx=0, 
        max=99)
    assert outv == 'foo_bar_12345678_0.baz'
 
def test_truncate_fit1():
    outv = truncate_fit(
        prefix='foo_', 
        prompt='bar', 
        ext='.baz', 
        ts=12345678,
        idx=0, 
        max=22)
    assert outv == 'foo_ba_12345678_0.baz'
 
####################3

# to do: this should fail for lerp values outside [0,1]
def test_image_mix(np_image):
    outv = image_mix(
        img_a=np_image,
        img_b=np_image,
        tween=0.5
    )
    assert isinstance(outv, type(np_image))
    assert outv.shape == np_image.shape

def test_image_to_jpg_bytes(np_image):
    outv=image_to_jpg_bytes(image=np_image)
    assert isinstance(outv, ByteString)

def test_image_to_png_bytes(np_image):
    outv=image_to_png_bytes(image=np_image)
    assert isinstance(outv, ByteString)

def test_image_to_prompt(np_image):
    outv = image_to_prompt(np_image)
    assert isinstance(outv, generation.Prompt)
    assert outv.artifact.type == generation.ARTIFACT_IMAGE

def test_image_to_prompt_mask(np_image):
    outv = image_to_prompt(np_image, is_mask=True)
    assert isinstance(outv, generation.Prompt)
    assert outv.artifact.type == generation.ARTIFACT_MASK

########################################

    # warp2d_op,
    # warp3d_op,
    # colormatch_op,
    # depthcalc_op,
    # warpflow_op,

# should a null transform op even return a transform op?
# would probably be better if this actually returned None or an empty list or 
# some sort of NOOP
@pytest.mark.parametrize("border_mode", BORDER_MODES_2D.keys())
def test_warp2d_op_valid(border_mode):
    op = warp2d_op(
        border_mode = border_mode_from_str_2d(border_mode),
        rotate = 0,
        scale = 0,
        translate_x = 0,
        translate_y = 0,
    )
    assert isinstance(op, generation.TransformOperation)

@pytest.mark.parametrize("border_mode", ['not a border mode'])
def test_warp2d_op_invalid(border_mode):
    op = warp2d_op(
        border_mode = border_mode,
        rotate = 0,
        scale = 0,
        translate_x = 0,
        translate_y = 0,
    )
    assert isinstance(op, generation.TransformOperation)

@pytest.mark.parametrize("border_mode", BORDER_MODES_3D.keys())
def test_warp3d_op_valid(border_mode):
    op = warp3d_op(
        border_mode=border_mode,
        translate_x=0,
        translate_y=0,
        translate_z=0,
        rotate_x=0,
        rotate_y=0,
        rotate_z=0,
        near_plane=0,
        far_plane=0,
        fov=0, 
    )
    assert isinstance(op, generation.TransformOperation)
import pytest

from typing import ByteString

from stability_sdk.utils import (
    COLOR_SPACES,
    GUIDANCE_PRESETS,
    SAMPLERS,
    get_sampler_from_str,
    guidance_from_string,
    truncate_fit,
    ########
    image_mix,
    image_to_jpg_bytes,
    image_to_png_bytes,
    image_to_prompt,
    image_to_prompt_mask,
    image_xform,
    #########
    key_frame_inbetweens,
    key_frame_parse,
    #########
    warp2d_op,
    warp3d_op,
)


@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
def test_get_sampler_from_str_valid(sampler_name):
    get_sampler_from_str(s=sampler_name)
    assert True

def test_get_sampler_from_str_invalid():
    with pytest.raises(ValueError, match="invalid sampler"):
        get_sampler_from_str(s='not a real sampler')



@pytest.mark.parametrize("preset_name", GUIDANCE_PRESETS.keys())
def test_guidance_from_string_valid(preset_name):
    guidance_from_string(s=preset_name)
    assert True

def test_guidance_from_string_invalid():
    with pytest.raises(ValueError, match="invalid guidance preset"):
        guidance_from_string(s='not a real preset')

        
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


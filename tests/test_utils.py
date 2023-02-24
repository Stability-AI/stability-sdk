import numpy as np
import pytest

from typing import ByteString

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.matrix as matrix
from stability_sdk.utils import (
    BORDER_MODES,
    COLOR_SPACES,
    GUIDANCE_PRESETS,
    SAMPLERS,
    artifact_type_to_str,
    border_mode_from_str,
    color_match_from_string,
    get_sampler_from_str,
    guidance_from_string,
    sampler_from_string,
    truncate_fit,
    ########
    image_mix,
    image_to_jpg_bytes,
    image_to_png_bytes,
    image_to_prompt,
    #########
    key_frame_inbetweens,
    key_frame_parse,
    #########
    blend_op,
    color_adjust_op,
    depthcalc_op,
    resample_op
)

@pytest.mark.parametrize("border", BORDER_MODES.keys())
def test_border_mode_from_str_2d_valid(border):
    border_mode_from_str(s=border)
    assert True

def test_border_mode_from_str_2d_invalid():
    with pytest.raises(ValueError, match="invalid border mode"):
        border_mode_from_str(s='not a real border mode')

@pytest.mark.parametrize("artifact_type", generation.ArtifactType.values())
def test_artifact_type_to_str_valid(artifact_type):
    type_str = artifact_type_to_str(artifact_type)
    assert type_str == generation.ArtifactType.Name(artifact_type)

def test_artifact_type_to_str_invalid():
    type_str = artifact_type_to_str(-1)
    assert type_str == 'ARTIFACT_UNRECOGNIZED'

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

def test_image_mix(np_image):
    outv = image_mix(
        img_a=np_image,
        img_b=np_image,
        ratio=0.5
    )
    assert isinstance(outv, type(np_image))
    assert outv.shape == np_image.shape

def test_image_mix_per_channel(np_image):
    per_channel_ratios = np.full(np_image.shape, 0.5)
    outv = image_mix(
        img_a=np_image,
        img_b=np_image,
        ratio=per_channel_ratios
    )
    assert isinstance(outv, type(np_image))
    assert outv.shape == np_image.shape

def test_image_mix_per_pixel(np_image):
    per_pixel_ratios = np.full(np_image.shape[:2], 0.5)
    outv = image_mix(
        img_a=np_image,
        img_b=np_image,
        ratio=per_pixel_ratios
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
    outv = image_to_prompt(np_image, type=generation.ARTIFACT_MASK)
    assert isinstance(outv, generation.Prompt)
    assert outv.artifact.type == generation.ARTIFACT_MASK

########################################

@pytest.mark.parametrize("color_mode", COLOR_SPACES.keys())
def test_colormatch_op_valid(np_image, color_mode):
    op = color_adjust_op(
        match_image=np_image,
        match_mode=color_mode
    )
    assert isinstance(op, generation.TransformParameters)

def test_colormatch_op_invalid(np_image):
    with pytest.raises(ValueError, match="invalid color space"):
        _ = color_adjust_op(
            match_image=np_image,
            match_mode="not a real color mode",
        )

@pytest.mark.parametrize("border_mode", BORDER_MODES.keys())
def test_resample_op_valid(border_mode):
    op = resample_op(
        border_mode=border_mode, 
        transform=matrix.identity, 
        prev_transform=matrix.identity, 
        depth_warp=1.0, 
        export_mask=False
    )
    assert isinstance(op, generation.TransformParameters)

@pytest.mark.parametrize("border_mode", ['not a border mode'])
def test_resample_op_invalid(border_mode):
    with pytest.raises(ValueError, match="invalid border mode"):
        _ = resample_op(
            border_mode=border_mode, 
            transform=matrix.identity, 
            prev_transform=matrix.identity, 
            depth_warp=1.0, 
            export_mask=False
        )

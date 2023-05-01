import pytest

from PIL import Image
from typing import ByteString

import stability_sdk.matrix as matrix
from stability_sdk.api import generation
from stability_sdk.utils import (
    BORDER_MODES,
    COLOR_SPACES,
    GUIDANCE_PRESETS,
    SAMPLERS,
    artifact_type_to_string,
    border_mode_from_string,
    color_match_from_string,
    guidance_from_string,
    sampler_from_string,
    truncate_fit,
    ########
    image_mix,
    image_to_jpg_bytes,
    image_to_png_bytes,
    image_to_prompt,
    #########
    color_adjust_op,
    depthcalc_op,
    resample_op
)

@pytest.mark.parametrize("border", BORDER_MODES.keys())
def test_border_mode_from_str_2d_valid(border):
    border_mode_from_string(s=border)
    assert True

def test_border_mode_from_str_2d_invalid():
    with pytest.raises(ValueError, match="invalid border mode"):
        border_mode_from_string(s='not a real border mode')

@pytest.mark.parametrize("artifact_type", generation.ArtifactType.values())
def test_artifact_type_to_str_valid(artifact_type):
    type_str = artifact_type_to_string(artifact_type)
    assert type_str == generation.ArtifactType.Name(artifact_type)

def test_artifact_type_to_str_invalid():
    type_str = artifact_type_to_string(-1)
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

def test_image_mix(pil_image):
    result = image_mix(img_a=pil_image, img_b=pil_image, ratio=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == pil_image.size

def test_image_to_jpg_bytes(pil_image):
    result = image_to_jpg_bytes(pil_image)
    assert isinstance(result, ByteString)

def test_image_to_png_bytes(pil_image):
    result = image_to_png_bytes(image=pil_image)
    assert isinstance(result, ByteString)

def test_image_to_prompt(pil_image):
    result = image_to_prompt(pil_image)
    assert isinstance(result, generation.Prompt)
    assert result.artifact.type == generation.ARTIFACT_IMAGE

def test_image_to_prompt_mask(pil_image):
    result = image_to_prompt(pil_image, type=generation.ARTIFACT_MASK)
    assert isinstance(result, generation.Prompt)
    assert result.artifact.type == generation.ARTIFACT_MASK

########################################

@pytest.mark.parametrize("color_mode", COLOR_SPACES.keys())
def test_colormatch_op_valid(pil_image, color_mode):
    op = color_adjust_op(
        match_image=pil_image,
        match_mode=color_mode
    )
    assert isinstance(op, generation.TransformParameters)

def test_colormatch_op_invalid(pil_image):
    with pytest.raises(ValueError, match="invalid color space"):
        _ = color_adjust_op(
            match_image=pil_image,
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

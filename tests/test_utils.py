import pytest

from PIL import Image
from typing import ByteString

import stability_sdk.matrix as matrix
from stability_sdk.api import generation
from stability_sdk.utils import (
    BORDER_MODES,
    COLOR_MATCH_MODES,
    GUIDANCE_PRESETS,
    SAMPLERS,
    artifact_type_to_string,
    border_mode_from_string,
    color_adjust_transform,
    color_match_from_string,
    depth_calc_transform,
    expand_subprompts,
    guidance_from_string,
    image_mix,
    image_to_jpg_bytes,
    image_to_png_bytes,
    image_to_prompt,
    parse_weighted_subprompts,
    resample_transform,
    sampler_from_string,
    truncate_fit,
)

@pytest.mark.parametrize("artifact_type", generation.ArtifactType.values())
def test_artifact_type_to_str_valid(artifact_type):
    type_str = artifact_type_to_string(artifact_type)
    assert type_str == generation.ArtifactType.Name(artifact_type)

def test_artifact_type_to_str_invalid():
    type_str = artifact_type_to_string(-1)
    assert type_str == 'ARTIFACT_UNRECOGNIZED'

@pytest.mark.parametrize("border", BORDER_MODES.keys())
def test_border_mode_from_str_2d_valid(border):
    border_mode_from_string(s=border)
    assert True

def test_border_mode_from_str_2d_invalid():
    with pytest.raises(ValueError, match="invalid border mode"):
        border_mode_from_string(s='not a real border mode')

@pytest.mark.parametrize("color_match_mode", COLOR_MATCH_MODES.keys())
def test_color_match_from_string_valid(color_match_mode):
    color_match_from_string(s=color_match_mode)
    assert True

def test_color_match_from_string_invalid():
    with pytest.raises(ValueError, match="invalid color match"):
        color_match_from_string(s='not a real color match mode')

def test_expand_subprompts():
    assert expand_subprompts("single prompt") == (["single prompt"], [1.0])
    assert expand_subprompts(["one","two"]) == (["one", "two"], [1.0, 1.0])
    assert expand_subprompts(["single prompt"], [1.0]) == (["single prompt"], [1.0])
    assert expand_subprompts(["one", "two"], [1.0, 1.0]) == (["one", "two"], [1.0, 1.0])
    assert expand_subprompts(["one:0.1|two:0.2"], [1.0]) == (["one", "two"], [0.1, 0.2])
    assert expand_subprompts(["one:0.1|two:0.2", "three"], [0.5, 1.0]) == (["one", "two", "three"], [0.05, 0.1, 1.0])
    with pytest.raises(ValueError, match="Length of prompts and weights must be the same"):
        expand_subprompts(["one","two"], [1.0])

@pytest.mark.parametrize("preset_name", GUIDANCE_PRESETS.keys())
def test_guidance_from_string_valid(preset_name):
    guidance_from_string(s=preset_name)
    assert True

def test_guidance_from_string_invalid():
    with pytest.raises(ValueError, match="invalid guidance preset"):
        guidance_from_string(s='not a real preset')

def test_parse_weighted_prompts():
    assert parse_weighted_subprompts("single prompt") == (["single prompt"], [1.0])
    assert parse_weighted_subprompts("one|two") == (["one", "two"], [1.0, 1.0])
    assert parse_weighted_subprompts("  whitespace  |    removed") == (["whitespace", "removed"], [1.0, 1.0])
    assert parse_weighted_subprompts("single with weight :0.5") == (["single with weight"], [0.5])
    assert parse_weighted_subprompts("nospace:.1") == (["nospace"], [0.1])
    assert parse_weighted_subprompts("two:2|weighted:3") == (["two", "weighted"], [2,3])
    assert parse_weighted_subprompts("one:.1|two|three:3.3") == (["one", "two", "three"], [.1,1,3.3])

@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
def test_sampler_from_str_valid(sampler_name):
    sampler_from_string(s=sampler_name)
    assert True

def test_sampler_from_str_invalid():
    with pytest.raises(ValueError, match="invalid sampler"):
        sampler_from_string(s='not a real sampler')

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


#==============================================================================
# Image functions
#==============================================================================

def test_image_mix(pil_image):
    result = image_mix(img_a=pil_image, img_b=pil_image, ratio=0.5)
    assert isinstance(result, Image.Image)
    assert result.size == pil_image.size

def test_image_mix_mask(pil_image):
    result = image_mix(img_a=pil_image, img_b=pil_image, ratio=pil_image.convert('L'))
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


#==============================================================================
# Transform functions
#==============================================================================

@pytest.mark.parametrize("color_mode", COLOR_MATCH_MODES.keys())
def test_colormatch_valid(pil_image, color_mode):
    op = color_adjust_transform(
        match_image=pil_image,
        match_mode=color_mode
    )
    assert isinstance(op, generation.TransformParameters)

def test_colormatch_invalid(pil_image):
    with pytest.raises(ValueError, match="invalid color match"):
        _ = color_adjust_transform(
            match_image=pil_image,
            match_mode="not a real color match mode",
        )

@pytest.mark.parametrize("border_mode", BORDER_MODES.keys())
def test_resample_valid(border_mode):
    op = resample_transform(
        border_mode=border_mode, 
        transform=matrix.identity, 
        prev_transform=matrix.identity, 
        depth_warp=1.0, 
        export_mask=False
    )
    assert isinstance(op, generation.TransformParameters)

@pytest.mark.parametrize("border_mode", ['not a border mode'])
def test_resample_invalid(border_mode):
    with pytest.raises(ValueError, match="invalid border mode"):
        _ = resample_transform(
            border_mode=border_mode, 
            transform=matrix.identity, 
            prev_transform=matrix.identity, 
            depth_warp=1.0, 
            export_mask=False
        )

import pytest
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from stability_sdk.utils import (
    SAMPLERS,
    artifact_type_to_str,
    get_sampler_from_str,
    truncate_fit,
)

@pytest.mark.parametrize("artifact_type", generation.ArtifactType.values())
def test_artifact_type_to_str_valid(artifact_type):
    type_str = artifact_type_to_str(artifact_type)
    assert type_str == generation.ArtifactType.Name(artifact_type)

def test_artifact_type_to_str_invalid():
    type_str = artifact_type_to_str(-1)
    assert type_str == 'ARTIFACT_UNRECOGNIZED'

@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
def test_get_sampler_from_str_valid(sampler_name):
    get_sampler_from_str(s=sampler_name)
    assert True

def test_get_sampler_from_str_invalid():
    with pytest.raises(ValueError, match="unknown sampler"):
        get_sampler_from_str(s='not a real sampler')


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
 

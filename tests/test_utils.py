import pytest

from stability_sdk.utils import (
    SAMPLERS,
    get_sampler_from_str,
    truncate_fit,
)


@pytest.mark.parametrize("sampler_name", SAMPLERS.keys())
def test_get_sampler_from_str_valid(sampler_name):
    get_sampler_from_str(s=sampler_name)
    assert True

def test_get_sampler_from_str_invalid():
    try:
        get_sampler_from_str(s='not a real sampler')
        assert False
    except ValueError:
        assert True

def test_truncate_fit():
    truncate_fit(
        prefix='foo', 
        prompt='bar', 
        ext='.baz', 
        ts=0,
        idx=0, 
        max=99)
    assert True
 

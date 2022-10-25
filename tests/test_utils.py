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
 

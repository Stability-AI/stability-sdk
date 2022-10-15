import pytest
from PIL import Image

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

def test_client_import():
    from stability_sdk import client
    assert True

def test_StabilityInference_init():
    class_instance = client.StabilityInference(key='thisIsNotARealKey')
    assert True

def test_StabilityInference_init_nokey_error():
    try:
        class_instance = client.StabilityInference()
        assert False
    except ValueError:
        assert True

def test_StabilityInference_init_nokey_insecure_host():
    class_instance = client.StabilityInference(host='foo.bar.baz')
    assert True

@pytest.mark.parametrize("sampler_name", client.algorithms.keys())
def test_get_sampler_from_str_valid(sampler_name):
    client.get_sampler_from_str(s=sampler_name)
    assert True

def test_get_sampler_from_str_invalid():
    try:
        client.get_sampler_from_str(s='not a real sampler')
        assert False
    except ValueError:
        assert True

def test_truncate_fit():
    client.truncate_fit(
        prefix='foo', 
        prompt='bar', 
        ext='.baz', 
        ts=0,
        idx=0, 
        max=99)
    assert True

def test_image_to_prompt():
    im = Image.new('RGB',(1,1))
    prompt = client.image_to_prompt(im, init=False, mask=False)
    assert isinstance(prompt, generation.Prompt)

def test_image_to_prompt_init():
    im = Image.new('RGB',(1,1))
    prompt = client.image_to_prompt(im, init=True, mask=False)
    assert isinstance(prompt, generation.Prompt)

def test_image_to_prompt_mask():
    im = Image.new('RGB',(1,1))
    prompt = client.image_to_prompt(im, init=False, mask=True)
    assert isinstance(prompt, generation.Prompt)

def test_image_to_prompt_init_mask():
    im = Image.new('RGB',(1,1))
    try:
        prompt = client.image_to_prompt(im, init=True, mask=True)
        assert False
    except ValueError:
        assert True

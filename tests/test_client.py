from PIL import Image
from typing import Generator

from stability_sdk import client
from stability_sdk.api import generation


def test_StabilityInference_init():
    _ = client.StabilityInference(key='thisIsNotARealKey')
    assert True

def test_StabilityInference_init_nokey_insecure_host():
    _ = client.StabilityInference(host='foo.bar.baz')
    assert True

def test_image_to_prompt_init():
    im = Image.new('RGB', (1,1))
    prompt = client.image_to_prompt(im)
    assert isinstance(prompt, generation.Prompt)

def test_image_to_prompt_mask():
    im = Image.new('RGB', (1,1))
    prompt = client.image_to_prompt(im, type=generation.ARTIFACT_MASK)
    assert isinstance(prompt, generation.Prompt)

def test_server_mocking(grpc_server, grpc_addr):
    class_instance = client.StabilityInference(host=grpc_addr[0])
    response = class_instance.generate(prompt="foo bar")
    print(response)
    # might need this link later:
    # - https://stackoverflow.com/questions/54541338/calling-function-that-yields-from-a-pytest-fixture
    assert isinstance(response, Generator)

def test_upscale(grpc_server, grpc_addr):
    class_instance = client.StabilityInference(host=grpc_addr[0])
    im = Image.new('RGB',(1,1))
    response = class_instance.upscale(init_image=im)
    print(response)
    # might need this link later:
    # - https://stackoverflow.com/questions/54541338/calling-function-that-yields-from-a-pytest-fixture
    assert isinstance(response, Generator)


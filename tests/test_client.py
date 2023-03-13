import pytest
from PIL import Image

from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

import grpc

# feel like we should be using this, not sure how/where
import grpc_testing

from typing import Generator

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


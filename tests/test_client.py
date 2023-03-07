import base64
import cv2
import numpy as np
import pytest
from PIL import Image

from stability_sdk import client
from stability_sdk.utils import image_to_png_bytes
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

import grpc

# feel like we should be using this, not sure how/where
import grpc_testing

from typing import Generator

class MockStub:
    def __init__(self):
        pass

    def ChainGenerate(self, chain: generation.ChainRequest, **kwargs) -> Generator[generation.Answer, None, None]:
        pass

    def Generate(self, request: generation.Request, **kwargs) -> Generator[generation.Answer, None, None]:
        if request.HasField("image"):
            width, height = request.image.width, request.image.height
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            binary = image_to_png_bytes(image)            
            artifact = generation.Artifact(
                id=0,
                type=generation.ARTIFACT_IMAGE, 
                mime="image/png",
                binary=binary,
                size=len(binary)
            )
            yield generation.Answer(artifacts=[artifact])
        elif request.HasField("interpolate"):
            assert len(request.prompt) == 2
            assert request.prompt[0].artifact.type == generation.ARTIFACT_IMAGE
            assert request.prompt[1].artifact.type == generation.ARTIFACT_IMAGE
            image_a = cv2.imdecode(np.frombuffer(request.prompt[0].artifact.binary, np.uint8), cv2.IMREAD_COLOR)
            image_b = cv2.imdecode(np.frombuffer(request.prompt[1].artifact.binary, np.uint8), cv2.IMREAD_COLOR)
            assert image_a.shape == image_b.shape
            width, height = image_a.shape[1], image_a.shape[0]

            for idx, _ in enumerate(request.interpolate.ratios):
                image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                binary = image_to_png_bytes(image)                
                artifact = generation.Artifact(
                    id=idx,
                    type=generation.ARTIFACT_IMAGE, 
                    mime="image/png",
                    binary=binary,
                    index=idx,
                    size=len(binary)
                )
                yield generation.Answer(artifacts=[artifact])


def test_api_generate():
    api = client.Api(stub=MockStub())
    width, height = 512, 768
    results = api.generate(prompts=["foo bar"], weights=[1.0], width=width, height=height)
    assert isinstance(results, dict)
    assert generation.ARTIFACT_IMAGE in results
    assert len(results[generation.ARTIFACT_IMAGE]) == 1
    image = results[generation.ARTIFACT_IMAGE][0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (height, width, 3)

def test_api_generate_mse_loss():
    api = client.Api(stub=MockStub())
    width, height = 512, 768
    mse_loss_im = Image.new('RGB', (1,1))
    mse_loss_im_b64 = base64.b64encode(image_to_png_bytes(mse_loss_im)).decode('utf-8')
    extras = { "mse_loss": { "mse_image": mse_loss_im_b64,
                             "adj_mse_scale": 0.1 } }
    result = api.generate(prompts=["foo bar"], weights=[1.0], width=width, height=height, extras=extras)
    assert isinstance(result[1][0], np.ndarray)
    assert result[1][0].shape == (height, width, 3)

def test_api_inpaint():
    api = client.Api(stub=MockStub())
    width, height = 512, 768
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    mask = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    results = api.inpaint(image, mask, prompts=["foo bar"], weights=[1.0])
    assert generation.ARTIFACT_IMAGE in results
    assert len(results[generation.ARTIFACT_IMAGE]) == 1
    image = results[generation.ARTIFACT_IMAGE][0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (height, width, 3)

def test_api_interpolate():
    api = client.Api(stub=MockStub())
    width, height = 512, 768
    image_a = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    image_b = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    results = api.interpolate([image_a, image_b], [0.3, 0.5, 0.6])
    assert len(results) == 3

def test_api_transform():
    api = client.Api(stub=MockStub())
    # TODO: implement transform tests after API finalized

def test_client_import():
    from stability_sdk import client
    assert True

def test_StabilityInference_init():
    class_instance = client.StabilityInference(key='thisIsNotARealKey')
    assert True

def test_StabilityInference_init_nokey_insecure_host():
    class_instance = client.StabilityInference(host='foo.bar.baz')
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

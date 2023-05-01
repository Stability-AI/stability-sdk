import io
import numpy as np
from PIL import Image
from typing import Generator

from stability_sdk import client
from stability_sdk.api import Context, generation
from stability_sdk import utils



class MockStub:
    def __init__(self):
        pass

    def ChainGenerate(self, chain: generation.ChainRequest, **kwargs) -> Generator[generation.Answer, None, None]:
        pass

    def Generate(self, request: generation.Request, **kwargs) -> Generator[generation.Answer, None, None]:
        if request.HasField("image"):
            width, height = request.image.width, request.image.height
            image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
            binary = utils.image_to_png_bytes(image)            
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
            image_a = Image.open(io.BytesIO(request.prompt[0].artifact.binary))
            image_b = Image.open(io.BytesIO(request.prompt[1].artifact.binary))
            assert image_a.size == image_b.size
            width, height = image_a.size

            for idx, _ in enumerate(request.interpolate.ratios):
                binary = utils.image_to_png_bytes(image_a)
                artifact = generation.Artifact(
                    id=idx,
                    type=generation.ARTIFACT_IMAGE, 
                    mime="image/png",
                    binary=binary,
                    index=idx,
                    size=len(binary)
                )
                yield generation.Answer(artifacts=[artifact])
        elif request.HasField("transform"):
            assert len(request.prompt) >= 1
            assert request.prompt[0].artifact.type == generation.ARTIFACT_IMAGE
            for idx, prompt in enumerate(request.prompt):
                if prompt.artifact.type == generation.ARTIFACT_IMAGE:
                    artifact = generation.Artifact(
                        id=idx,
                        type=generation.ARTIFACT_IMAGE, 
                        mime="image/png",
                        binary=prompt.artifact.binary,
                        index=idx,
                        size=len(prompt.artifact.binary)
                    )
                    yield generation.Answer(artifacts=[artifact])


def test_api_generate():
    api = Context(stub=MockStub())
    width, height = 512, 768
    results = api.generate(prompts=["foo bar"], weights=[1.0], width=width, height=height)
    assert isinstance(results, dict)
    assert generation.ARTIFACT_IMAGE in results
    assert len(results[generation.ARTIFACT_IMAGE]) == 1
    image = results[generation.ARTIFACT_IMAGE][0]
    assert isinstance(image, Image.Image)
    assert image.size == (width, height)

def test_api_inpaint():
    api = Context(stub=MockStub())
    width, height = 512, 768
    image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    mask = Image.fromarray(np.random.randint(0, 255, (height, width), dtype=np.uint8))
    results = api.inpaint(image, mask, prompts=["foo bar"], weights=[1.0])
    assert generation.ARTIFACT_IMAGE in results
    assert len(results[generation.ARTIFACT_IMAGE]) == 1
    image = results[generation.ARTIFACT_IMAGE][0]
    assert isinstance(image, Image.Image)
    assert image.size == (width, height)

def test_api_interpolate():
    api = Context(stub=MockStub())
    width, height = 512, 768
    image_a = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    image_b = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    results = api.interpolate([image_a, image_b], [0.3, 0.5, 0.6])
    assert len(results) == 3

def test_api_transform():
    api = Context(stub=MockStub())
    width, height = 512, 768
    image = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    images, masks = api.transform([image], utils.color_adjust_op())
    assert len(images) == 1 and not masks
    assert isinstance(images[0], Image.Image)
    images, masks = api.transform([image, image], utils.color_adjust_op())
    assert len(images) == 2 and not masks

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


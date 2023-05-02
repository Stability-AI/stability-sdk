import io
import numpy as np
from PIL import Image
from typing import Generator

import stability_sdk.matrix as matrix
from stability_sdk import utils
from stability_sdk.api import Context, generation

def _artifact_from_image(image: Image.Image) -> generation.Artifact:
    binary = utils.image_to_png_bytes(image)            
    return generation.Artifact(
        type=generation.ARTIFACT_IMAGE, 
        mime="image/png",
        binary=binary,
        size=len(binary)
    )

def _rand_image(width: int=512, height: int=512) -> Image.Image:
    return Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))

class MockStub:
    def __init__(self):
        pass

    def ChainGenerate(self, chain: generation.ChainRequest, **kwargs) -> Generator[generation.Answer, None, None]:
        # Not a full implementation of chaining, but enough to test current api.Context layer
        artifacts = []
        for stage in chain.stage:
            stage.request.MergeFrom(generation.Request(prompt=[generation.Prompt(artifact=a) for a in artifacts]))
            artifacts = []
            for answer in self.Generate(stage.request):
                artifacts.extend(answer.artifacts)
        for artifact in artifacts:
            yield generation.Answer(artifacts=[artifact])        

    def Generate(self, request: generation.Request, **kwargs) -> Generator[generation.Answer, None, None]:
        if request.HasField("image"):
            image = _rand_image(request.image.width or 512, request.image.height or 512)
            yield generation.Answer(artifacts=[_artifact_from_image(image)])

        elif request.HasField("interpolate"):
            assert len(request.prompt) == 2
            assert request.prompt[0].artifact.type == generation.ARTIFACT_IMAGE
            assert request.prompt[1].artifact.type == generation.ARTIFACT_IMAGE
            image_a = Image.open(io.BytesIO(request.prompt[0].artifact.binary))
            image_b = Image.open(io.BytesIO(request.prompt[1].artifact.binary))
            assert image_a.size == image_b.size
            for ratio in request.interpolate.ratios:
                tween = utils.image_mix(image_a, image_b, ratio)
                yield generation.Answer(artifacts=[_artifact_from_image(tween)])

        elif request.HasField("transform"):
            assert len(request.prompt) >= 1

            has_depth_input, has_tensor_input = False, False
            for prompt in request.prompt:
                if prompt.artifact.type == generation.ARTIFACT_DEPTH:
                    has_depth_input = True
                elif prompt.artifact.type == generation.ARTIFACT_TENSOR:
                    has_tensor_input = True

            # 3D resample and camera pose require a depth or depth tensor artifact
            if request.transform.HasField("resample") and len(request.transform.resample.transform.data) == 16:
                assert has_depth_input or has_tensor_input
            if request.transform.HasField("camera_pose"):
                assert has_depth_input or has_tensor_input

            export_mask = request.transform.HasField("camera_pose")
            if request.transform.HasField("resample"):
                if request.transform.resample.HasField("export_mask"):
                    export_mask = request.transform.resample.export_mask                       

            for prompt in request.prompt:
                if prompt.artifact.type == generation.ARTIFACT_IMAGE:
                    image = Image.open(io.BytesIO(prompt.artifact.binary))
                    artifact = _artifact_from_image(image)
                    if request.transform.HasField("depth_calc"):
                        if request.requested_type == generation.ARTIFACT_TENSOR:                        
                            artifact.type = generation.ARTIFACT_TENSOR
                        else:
                            artifact.type = generation.ARTIFACT_DEPTH                            
                    yield generation.Answer(artifacts=[artifact])

                    if export_mask:
                        mask = _rand_image(image.width, image.height).convert("L")
                        artifact = _artifact_from_image(mask)
                        artifact.type = generation.ARTIFACT_MASK
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
    image = _rand_image(width, height)
    mask = _rand_image(width, height).convert("L")
    results = api.inpaint(image, mask, prompts=["foo bar"], weights=[1.0])
    assert generation.ARTIFACT_IMAGE in results
    assert len(results[generation.ARTIFACT_IMAGE]) == 1
    image = results[generation.ARTIFACT_IMAGE][0]
    assert isinstance(image, Image.Image)
    assert image.size == (width, height)

def test_api_interpolate():
    api = Context(stub=MockStub())
    width, height = 512, 768
    image_a = _rand_image(width, height)
    image_b = _rand_image(width, height)
    results = api.interpolate([image_a, image_b], [0.3, 0.5, 0.6])
    assert len(results) == 3
    for image in results:
        assert isinstance(image, Image.Image)
        assert image.size == (width, height)

def test_api_transform_and_generate():
    api = Context(stub=MockStub())
    width, height = 512, 704
    init_image = _rand_image(width, height)
    generate_request = api.generate(["a cute cat"], [1], width=width, height=height, 
                                    init_strength=0.65, return_request=True)
    assert isinstance(generate_request, generation.Request)
    image = api.transform_and_generate(init_image, [utils.color_adjust_transform()], generate_request)
    assert isinstance(image, Image.Image)
    assert image.size == (width, height)

def test_api_transform_camera_pose():
    api = Context(stub=MockStub())
    image = _rand_image()
    xform = matrix.identity
    pose = utils.camera_pose_transform(
        xform, 0.1, 100.0, 75.0,
        camera_type='perspective',
        render_mode='mesh',
        do_prefill=True
    )
    images, masks = api.transform_3d([image], utils.depth_calc_transform(blend_weight=1.0), pose)
    assert len(images) == 1 and len(masks) == 1
    assert isinstance(images[0], Image.Image)
    assert isinstance(masks[0], Image.Image)

def test_api_transform_color_adjust():
    api = Context(stub=MockStub())
    image = _rand_image()
    images, masks = api.transform([image], utils.color_adjust_transform())
    assert len(images) == 1 and not masks
    assert isinstance(images[0], Image.Image)
    images, masks = api.transform([image, image], utils.color_adjust_transform())
    assert len(images) == 2 and not masks

def test_api_transform_resample_3d():
    api = Context(stub=MockStub())
    image = _rand_image()
    xform = matrix.identity
    resample = utils.resample_transform('replicate', xform, xform, export_mask=True)
    images, masks = api.transform_3d([image], utils.depth_calc_transform(blend_weight=0.5), resample)
    assert len(images) == 1 and len(masks) == 1
    assert isinstance(images[0], Image.Image)
    assert isinstance(masks[0], Image.Image)

def test_api_upscale():
    api = Context(stub=MockStub())
    result = api.upscale(_rand_image())
    assert isinstance(result, Image.Image)

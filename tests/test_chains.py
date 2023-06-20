import io
import os
import warnings
import uuid

import contextlib
import grpc
from PIL import Image
import pytest
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from typing import Dict, Generator, List

from .test_api import MockStub

if 'STABILITY_HOST' in os.environ and \
    'STABILITY_KEY' in os.environ:
    stability_api = client.StabilityInference(
        host=os.environ['STABILITY_HOST'],
        key=os.environ['STABILITY_KEY'], 
        verbose=True,
    )
    stub = stability_api.stub

    scope_raises_error_with_init_image = pytest.raises(grpc._channel._MultiThreadedRendezvous, match="Face enhancer cannot be used with an init image")
else:
    # Use a mock stub if no stability host is provided
    stub = MockStub()
    # Mock stub does not raise an error with init image (this behavior comes from the middleware)
    scope_raises_error_with_init_image = contextlib.nullcontext()

img_file = 'tests/assets/4166726513_giant__rainbow_sequoia__tree_by_hayao_miyazaki___earth_tones__a_row_of_western_cedar_nurse_trees_che.png'
img_in = Image.open(img_file)

generate_engine = 'stable-diffusion-512-v2-1'
enhance_engine = 'face-enhance-v1'
upscale_engine = 'esrgan-v1-x2plus'

# request directly to enhance engine
def enhance(
            stub,
            enhance_engine,
            init_image: Image.Image,
    ) -> Generator[generation.Answer, None, None]:
        
    prompts = [client.image_to_prompt(init_image)]

    image_parameters=generation.ImageParameters()

    request_id = str(uuid.uuid4())

    rq = generation.Request(
        engine_id=enhance_engine,
        request_id=request_id,
        prompt=prompts,
        image=image_parameters
    )

    for answer in stub.Generate(rq):
        yield answer


def generate_request(
        generate_engine: str,
        prompt_str: str,
        seeds: List[int] = None,
        init_image: Image.Image = None,
        **kwargs
    ) -> generation.Request:
        
        prompt = [generation.Prompt(text=prompt_str)]
        if init_image is not None:
            prompt = prompt + [client.image_to_prompt(init_image)]

        if seeds is not None:
            image_parameters = generation.ImageParameters(seed=seeds)
        else:
            image_parameters = generation.ImageParameters()

        request_id = str(uuid.uuid4())

        rq = generation.Request(
            engine_id=generate_engine,
            request_id=request_id,
            prompt=prompt,
            image=image_parameters,
            **kwargs
        )
        
        return rq

def enhance_request(
        enhance_engine:str,
        init_image: str = None,
        **kwargs
    ) -> generation.Request:
        prompt = []
        if init_image is not None:
            prompt = prompt + [client.image_to_prompt(init_image)]

        image_parameters = generation.ImageParameters()

        request_id = str(uuid.uuid4())

        rq = generation.Request(
            engine_id=enhance_engine,
            request_id=request_id,
            image=image_parameters,
            prompt=prompt,
            **kwargs
        )
        
        return rq

def ChainGenerate(self,
                    chain_rq: generation.ChainRequest, 
                    **kwargs) -> Generator[generation.Answer, None, None]:
    artifacts = []
    for answer in stub.ChainGenerate(chain_rq):
        artifacts.extend(answer.artifacts)
    for artifact in artifacts:
        yield generation.Answer(artifacts=[artifact])   

def test_no_chains_face_enhancer():
    """

    Face enhancer directly on an image

    This should FAIL

    """
    with scope_raises_error_with_init_image:

        answers = enhance(
            stub=stub,
            enhance_engine=enhance_engine,
            init_image=img_in
        )
        
        # iterating over the generator produces the api response
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    warnings.warn(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again.")
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))


def test_generate_face_enhancer():
    """

    Generate an image ( no init image ) and then send to face_enhancer.

    This should pass

    """
    rq_1 = generate_request(
        generate_engine=generate_engine,
        prompt_str="close-up face portrait of an amazing vibrant scifi astronaut",
        requested_type=generation.ARTIFACT_IMAGE,
    )
    rq_2 = enhance_request(
        enhance_engine=enhance_engine,
        requested_type=generation.ARTIFACT_IMAGE
    )

    stages = []
    stages.append(generation.Stage(
        id=str(0),
        request=rq_1, 
        on_status=[generation.OnStatus(
            action=[
                generation.STAGE_ACTION_PASS,
                ],
            target=str(1)
        )]
    ))
    stages.append(generation.Stage(
        id=str(1),
        request=rq_2, 
        on_status=[generation.OnStatus(
            action=[generation.STAGE_ACTION_RETURN], 
            target=None
        )]
    ))
    chain_rq = generation.ChainRequest(request_id="enhancer_chain", stage=stages)
    response = stub.ChainGenerate(chain_rq, wait_for_ready=True)
    for result in response:
        for artifact in result.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))

def test_chain_generate_face_enhancer_with_init_image():
    """

    Generate an image using an init image and then send to face_enhancer.

    This should FAIL

    """
    with scope_raises_error_with_init_image:
        rq_1 = generate_request(
            generate_engine=generate_engine,
            prompt_str="close-up face portrait of an amazing vibrant scifi astronaut",
            requested_type=generation.ARTIFACT_IMAGE,
            seeds=[1],
            init_image=img_in,
        )
        rq_2 = enhance_request(
            enhance_engine=enhance_engine,
            requested_type=generation.ARTIFACT_IMAGE,
        )

        stages = []
        stages.append(generation.Stage(
            id=str(0),
            request=rq_1, 
            on_status=[generation.OnStatus(
                action=[
                    generation.STAGE_ACTION_RETURN,
                    generation.STAGE_ACTION_PASS,
                    ],
                target=str(1)
            )]
        ))
        stages.append(generation.Stage(
            id=str(1),
            request=rq_2, 
            on_status=[generation.OnStatus(
                action=[generation.STAGE_ACTION_RETURN], 
                target=None
            )]
        ))
        chain_rq = generation.ChainRequest(request_id="enhancer_chain", stage=stages)
        response = stub.ChainGenerate(chain_rq, wait_for_ready=True)
        for result in response:
            for artifact in result.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))


def test_chain_generate_init_image_x2_face_enhancer():
    """

    Generate an image with init image, 
    generate again with the previous stage init image, 
    and then send to face_enhancer.

    This should FAIL

    """

    with scope_raises_error_with_init_image:
        rq_0 = generate_request(
            generate_engine=generate_engine,
            prompt_str="close-up face portrait of an amazing vibrant scifi astronaut",
            requested_type=generation.ARTIFACT_IMAGE,
            seeds=[1],
            init_image=img_in,
        )
        rq_1 = generate_request(
            generate_engine=generate_engine,
            prompt_str="portrait in the forest",
            requested_type=generation.ARTIFACT_IMAGE,
            seeds=[1],
        )
        rq_2 = enhance_request(
            enhance_engine=enhance_engine,
            requested_type=generation.ARTIFACT_IMAGE,
        )

        stages = []
        stages.append(generation.Stage(
            id=str(0),
            request=rq_0, 
            on_status=[generation.OnStatus(
                action=[
                    generation.STAGE_ACTION_RETURN,
                    generation.STAGE_ACTION_PASS,
                    ],
                target=str(1)
            )]
        ))
        stages.append(generation.Stage(
            id=str(1),
            request=rq_1, 
            on_status=[generation.OnStatus(
                action=[
                    generation.STAGE_ACTION_RETURN,
                    generation.STAGE_ACTION_PASS,
                    ],
                target=str(2)
            )]
        ))
        stages.append(generation.Stage(
            id=str(2),
            request=rq_2, 
            on_status=[generation.OnStatus(
                action=[generation.STAGE_ACTION_RETURN], 
                target=None
            )]
        ))
        chain_rq = generation.ChainRequest(request_id="enhancer_chain4", stage=stages)
        response = stub.ChainGenerate(chain_rq, wait_for_ready=True)
        for result in response:
            for artifact in result.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))


def test_chain_face_enhancer_alone():
    """

    Face enhancer alone in a chain

    This should FAIL

    """
    with scope_raises_error_with_init_image:
        rq_0 = enhance_request(
            enhance_engine=enhance_engine,
            requested_type=generation.ARTIFACT_IMAGE,
            init_image=img_in,
        )

        stages = []
        # enhance
        stages.append(generation.Stage(
            id=str(0),
            request=rq_0, 
            on_status=[generation.OnStatus(
                action=[generation.STAGE_ACTION_RETURN], 
                target=None
            )]
        ))
        chain_rq = generation.ChainRequest(request_id="enhancer_chain", stage=stages)
        response = stub.ChainGenerate(chain_rq, wait_for_ready=True)
        for result in response:
            for artifact in result.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))

def test_chain_upscale_with_init_image_then_face_enhancer():
    """

    Upscale with an init image, 
    and then send to face enhancer.

    This should FAIL

    """

    with scope_raises_error_with_init_image:
        rq_1 = enhance_request(
            enhance_engine=upscale_engine,
            requested_type=generation.ARTIFACT_IMAGE,
            init_image=img_in,
        )
        rq_2 = enhance_request(
            enhance_engine=enhance_engine,
            requested_type=generation.ARTIFACT_IMAGE
        )

        stages = []
        # upscale
        stages.append(generation.Stage(
            id=str(1),
            request=rq_1, 
            on_status=[generation.OnStatus(
                action=[
                    generation.STAGE_ACTION_RETURN,
                    generation.STAGE_ACTION_PASS,
                    ],
                target=str(2)
            )]
        ))
        # enhance
        stages.append(generation.Stage(
            id=str(2),
            request=rq_2, 
            on_status=[generation.OnStatus(
                action=[generation.STAGE_ACTION_RETURN], 
                target=None
            )]
        ))
        chain_rq = generation.ChainRequest(request_id="enhancer_chain", stage=stages)
        response = stub.ChainGenerate(chain_rq, wait_for_ready=True)
        for result in response:
            for artifact in result.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))

#TODO test recursive chains when they are enabled in the middleware
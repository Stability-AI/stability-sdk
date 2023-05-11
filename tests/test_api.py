import pytest
import time
import base64

from stability_sdk.api import CreateRequest, CreateResponse, GenerationResponse
from stability_api.platform.generation.v1.generation_pb2 import Answer, Artifact
import stability_api.platform.generation.v1.generation_pb2 as generation

def test_text_to_image():
    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
                {"text": "A photo of a dog sitting on a couch.", "weight": 0.5},
                {"text": "Green.", "weight": -0.5},
            ],
            "height": 512,
            "width": 512,
            "cfg_scale": 7.0,
            "samples": 1,
            "steps": 50,
            "sampler": "DDIM",
            "seed": 1,
            "style_preset": "neon-punk",
            "extras": {"$IPC": {"test": "0"}},
        }
    )

    prompts = request.prompt
    assert len(prompts) == 3
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].text == "A photo of a dog sitting on a couch."
    assert prompts[1].parameters.weight == 0.5
    assert prompts[2].text == "Green."
    assert prompts[2].parameters.weight == -0.5

    image = request.image
    assert image.height == 512
    assert image.width == 512
    assert image.steps == 50
    assert image.seed == [1]


def test_image_to_image_with_strength():
    image_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")
    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
            ],
            "init_image": image_base64,
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.5,
        }
    )

    prompts = request.prompt
    assert len(prompts) == 2
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].artifact is not None
    assert prompts[1].artifact.binary is not None

    # todo - check step schedule


def test_image_to_image_with_schedule():
    image_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")
    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
            ],
            "init_image": image_base64,
            "init_image_mode": "STEP_SCHEDULE",
            "step_schedule_start": 0.5,
            "step_schedule_end": 0.75,
        }
    )

    prompts = request.prompt
    assert len(prompts) == 2
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].artifact is not None
    assert prompts[1].artifact.binary is not None

    # todo - check step schedule


def test_image_to_image_with_init_image_alpha():
    image_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")

    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
            ],
            "init_image": image_base64,
            "mask_source": "INIT_IMAGE_ALPHA",
        }
    )

    prompts = request.prompt
    assert len(prompts) == 3
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].artifact is not None
    assert prompts[1].artifact.binary is not None
    assert prompts[2].artifact is not None
    assert prompts[2].artifact.binary is not None


def test_image_to_image_with_mask_image_white():
    image_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")
    # todo make a real mask so we can test image transforms
    mask_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")

    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
            ],
            "init_image": image_base64,
            "mask_source": "MASK_IMAGE_WHITE",
            "mask_image": mask_base64,
        }
    )

    prompts = request.prompt
    assert len(prompts) == 3
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].artifact is not None
    assert prompts[1].artifact.binary is not None
    assert prompts[2].artifact is not None
    assert prompts[2].artifact.binary is not None


def test_image_to_image_with_mask_image_black():
    image_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")
    # todo make a real mask so we can test image transforms
    mask_base64 = base64.b64encode(
        open("tests/resources/beach.png", "rb").read()
    ).decode("utf-8")

    request = CreateRequest(
        {
            "text_prompts": [
                {"text": "A photo of a cat sitting on a couch."},
            ],
            "init_image": image_base64,
            "mask_source": "MASK_IMAGE_BLACK",
            "mask_image": mask_base64,
        }
    )

    prompts = request.prompt
    assert len(prompts) == 3
    assert prompts[0].text == "A photo of a cat sitting on a couch."
    assert prompts[0].parameters.weight == 1.0
    assert prompts[1].artifact is not None
    assert prompts[1].artifact.binary is not None
    assert prompts[2].artifact is not None
    assert prompts[2].artifact.binary is not None

def test_generation_response_success():
    test_result = {'result': 'success', 'artifacts': [{'base64': 'blahblah', 'finishReason': 'SUCCESS', 'seed': 1}]}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'success'
    assert response.artifacts[0].finishReason == 'SUCCESS'
    assert response.artifacts[0].seed == 1
    assert response.artifacts[0].base64 == 'blahblah'

def test_generation_response_error():
    test_result = {'result': 'error', 'error': {'id': 'blahblah', 'message': 'blahblah', 'name': 'blahblah'}}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'error'
    assert response.error.id == 'blahblah'
    assert response.error.message == 'blahblah'
    assert response.error.name == 'blahblah'
    
def test_generation_response_v1_error():
    test_result = {'result': 'error', 'id': 'blahblah', 'message': 'blahblah', 'name': 'blahblah'}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'error'
    assert response.error.id == 'blahblah'
    assert response.error.message == 'blahblah'
    assert response.error.name == 'blahblah'
    
def test_create_response_filter():
    error_artifact = Artifact(
        type=generation.ARTIFACT_TEXT,
        mime="text/plain",
        text="Prompt is invalid.",
        finish_reason=generation.FILTER,
    )
    error_answer = generation.Answer(
        answer_id="error",
        request_id="test",
        created=int(time.time() * 1000),
        received=int(time.time() * 1000),
    )
    error_answer.artifacts.append(error_artifact)

    error_response = CreateResponse(error_answer)
    assert error_response.result == "error"
    assert error_response.artifacts is None
    assert error_response.error is not None
    assert error_response.error.name == "invalid_prompts"

def test_create_response_error():
    error_artifact = Artifact(
        type=generation.ARTIFACT_TEXT,
        mime="text/plain",
        text="Error generating.",
        finish_reason=generation.ERROR,
    )
    error_answer = generation.Answer(
        answer_id="error",
        request_id="test",
        created=int(time.time() * 1000),
        received=int(time.time() * 1000),
    )
    error_answer.artifacts.append(error_artifact)

    error_response = CreateResponse(error_answer)
    assert error_response.result == "error"
    assert error_response.artifacts is None
    assert error_response.error is not None    
    assert error_response.error.name == "generation_error"
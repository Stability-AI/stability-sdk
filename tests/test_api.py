import pytest
import os, sys
import base64

from stability_sdk.api import CreateRequest, GenerationResponse


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

def test_response_success():
    test_result = {'result': 'success', 'artifacts': [{'base64': 'blahblah', 'id': 'blahblah', 'seed': 1}]}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'success'
    assert response.artifacts[0].id == 'blahblah'
    assert response.artifacts[0].seed == 1
    assert response.artifacts[0].base64 == 'blahblah'

def test_response_error():
    test_result = {'result': 'error', 'error': {'id': 'blahblah', 'message': 'blahblah', 'name': 'blahblah'}}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'error'
    assert response.error.id == 'blahblah'
    assert response.error.message == 'blahblah'
    assert response.error.name == 'blahblah'

def test_v1_error():
    test_result = {'result': 'error', 'id': 'blahblah', 'message': 'blahblah', 'name': 'blahblah'}
    response = GenerationResponse.parse_obj(test_result)
    assert response.result == 'error'
    assert response.error.id == 'blahblah'
    assert response.error.message == 'blahblah'
    assert response.error.name == 'blahblah'
    
import base64
from io import BytesIO
from enum import Enum
from PIL import Image, ImageOps
import pathlib
from typing import Dict,List,Optional,TypedDict
import sys

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

class TextPrompt(TypedDict):
    text: str
    weight: float

class GuidancePreset(Enum):
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    FAST_BLUE = "FAST_BLUE"
    FAST_GREEN = "FAST_GREEN"
    SLOW = "SLOW"
    SLOWER = "SLOWER"
    SLOWEST = "SLOWEST"

class Sampler(Enum):
    DDIM = "DDIM"
    DDPM = "DDPM"
    K_DPMPP_SDE = "K_DPMPP_SDE"    
    K_DPMPP_2M = "K_DPMPP_2M"
    K_DPMPP_2S_ANCESTRAL = "K_DPMPP_2S_ANCESTRAL"
    K_DPM_2 = "K_DPM_2"
    K_DPM_2_ANCESTRAL = "K_DPM_2_ANCESTRAL"
    K_EULER = "K_EULER"
    K_EULER_ANCESTRAL = "K_EULER_ANCESTRAL"
    K_HEUN = "K_HEUN"
    K_LMS = "K_LMS"

class InitImageMode(Enum):
    IMAGE_STRENGTH = "IMAGE_STRENGTH"
    STEP_SCHEDULE = "STEP_SCHEDULE"

class MaskSource(Enum):
    MASK_IMAGE_BLACK = "MASK_IMAGE_BLACK"
    MASK_IMAGE_WHITE = "MASK_IMAGE_WHITE"
    INIT_IMAGE_ALPHA = "INIT_IMAGE_ALPHA"

class GenerationRequest(TypedDict):    
    height: Optional[int]
    width: Optional[int]
    text_prompts: Optional[List[TextPrompt]]
    cfg_scale: Optional[float]
    clip_guidance_preset: Optional[GuidancePreset]
    sampler: Optional[Sampler]
    samples: Optional[int]
    seed: Optional[int]
    steps: Optional[int]
    style_preset: Optional[str]
    extras: Optional[Dict[str, any]]

    # image to image specific options
    init_image: Optional[str]
    init_image_mode: Optional[InitImageMode]
    image_strength: Optional[float]

    # image to image with masking specific options
    mask_source: Optional[MaskSource]
    mask_image: Optional[str]



def json_to_api_request(json: Dict[str, any]) -> GenerationRequest:
    """Converts a JSON request to a GenerationRequest."""
    request: GenerationRequest = {}
    request["text_prompts"] = []
    request["extras"] = {}

    for key, value in json.items():
        if key == "text_prompts":
            for text_prompt in value:
                if text_prompt.get("weight") is None:
                    request["text_prompts"].append(
                        TextPrompt(text=text_prompt["text"], weight=1.0)
                    )
                else:
                    request["text_prompts"].append(
                        TextPrompt(
                            text=text_prompt["text"], weight=text_prompt["weight"]
                        )
                    )
        elif key == "sampler":
            request["sampler"] = Sampler(value)    
        elif key == "mask_source":
            request["mask_source"] = MaskSource(value)    
        elif key == "extras":
            for extra_key, extra_value in value.items():
                request["extras"][extra_key] = extra_value
        else:
            request[key] = value

    return request


def api_sampler_to_proto(sampler: Sampler) -> generation.DiffusionSampler:
    mappings = {
        Sampler.DDIM: generation.SAMPLER_DDIM,
        Sampler.DDPM: generation.SAMPLER_DDPM,
        Sampler.K_EULER: generation.SAMPLER_K_EULER,
        Sampler.K_EULER_ANCESTRAL: generation.SAMPLER_K_EULER_ANCESTRAL,
        Sampler.K_HEUN: generation.SAMPLER_K_HEUN,
        Sampler.K_DPM_2: generation.SAMPLER_K_DPM_2,
        Sampler.K_DPM_2_ANCESTRAL: generation.SAMPLER_K_DPM_2_ANCESTRAL,
        Sampler.K_LMS: generation.SAMPLER_K_LMS,
        Sampler.K_DPMPP_2S_ANCESTRAL: generation.SAMPLER_K_DPMPP_2S_ANCESTRAL,
        Sampler.K_DPMPP_2M: generation.SAMPLER_K_DPMPP_2M,
        Sampler.K_DPMPP_SDE: generation.SAMPLER_K_DPMPP_SDE,
    }    
    repr = mappings.get(sampler, None)
    if repr is None:
        raise ValueError(f'Invalid sampler: "{sampler}"')
    return repr


def api_request_to_proto(req: GenerationRequest) -> generation.Request:
    """Converts a GenerationRequest to a protobuf Request."""    
    
    sampler_params = generation.SamplerParameters()    

    transform_type = None
    if req.get("sampler") is not None:        
        transform_type = generation.TransformType(diffusion=api_sampler_to_proto(req['sampler']))

    image_params = generation.ImageParameters(transform=transform_type)

    if req.get("height") is not None:
        image_params.height = req["height"]
    if req.get("width") is not None:
        image_params.width = req["width"]
    if req.get("steps") is not None:
        image_params.steps = req["steps"]
    if req.get("samples") is not None:
        image_params.samples = req["samples"]
    if req.get("seed") is not None:
        image_params.seed.append(req["seed"])
    

    if req.get("cfg_scale") is not None:
        sampler_params.cfg_scale = req["cfg_scale"]
        
    schedule_params = generation.ScheduleParameters()

    init_image_prompt = None
    mask_image_prompt = None

    if req.get("init_image") is not None:
        image_mode = req.get("init_image_mode", InitImageMode.IMAGE_STRENGTH)
        step_start = 0.65
        if image_mode == InitImageMode.IMAGE_STRENGTH:            
            if req.get("image_strength") is not None:
                step_start = 1.0 - req["image_strength"]            
        elif image_mode == InitImageMode.STEP_SCHEDULE:
            step
            if req.get("step_schedule_start") is not None:
                step_start = req["step_schedule_start"]
            if req.get("step_schedule_end") is not None:
                step_end = req["step_schedule_end"]
                schedule_params.end = min(1,max(step_end, 0))
        schedule_params.start=min(1,max(step_start, 0))
        image_binary = base64.b64decode(req["init_image"])
        init_image_params = generation.PromptParameters(init=True)
        init_image_artifact = generation.Artifact(binary=image_binary, type=generation.ARTIFACT_IMAGE)
        init_image_prompt = generation.Prompt(artifact=init_image_artifact, parameters=init_image_params)        

    if req.get("mask_source") is not None:
        mask_source = req["mask_source"]
        mask_binary = None

        if mask_source == MaskSource.INIT_IMAGE_ALPHA:
            # Extracts the alpha channel from the init image and uses it as a mask.
            init_image = Image.open(init_image_artifact.binary)
            if init_image.mode != "RGBA":
                init_image = init_image.convert("RGBA")
            mask_image = init_image.split()[-1] # Extract alpha channel
            mask_binary = mask_image.tobytes()
        elif mask_source == MaskSource.MASK_IMAGE_WHITE:
            # Inverts the provided mask image, having the effect of masking out white pixels.
            if req.get("mask_image") is None:
                raise ValueError("mask_image must be specified if mask_source is MASK_IMAGE_WHITE")
            image_bytes = BytesIO(base64.b64decode(req["mask_image"]))
            mask_image = Image.open(image_bytes)
            if mask_image.mode != "L":
                mask_image = mask_image.convert("L")
            mask_image = ImageOps.invert(mask_image)
            mask_binary = mask_image.tobytes()
        elif mask_source == MaskSource.MASK_IMAGE_BLACK:
            # Uses the given mask image as-is, so that black pixels are masked out.
            if req.get("mask_image") is None:
                raise ValueError("mask_image must be specified if mask_source is MASK_IMAGE_BLACK")
            mask_binary = base64.b64decode(req["mask_image"])
        else:
            raise ValueError(f'Invalid mask_source: "{mask_source}"')

        mask_image_artifact = generation.Artifact(binary=mask_binary, type=generation.ARTIFACT_MASK)
        mask_image_prompt = generation.Prompt(artifact=mask_image_artifact)

    # Ignored for now, not supported with SDXL
    # clip_guidance_preset            
    
    step_param = generation.StepParameter(scaled_step=0, sampler=sampler_params, schedule=schedule_params)        
    image_params.parameters.append(step_param)
    request = generation.Request(image=image_params)

    for text_prompt in req["text_prompts"]:
        if text_prompt.get("weight") is None:
            request.prompt.append(generation.Prompt(text=text_prompt["text"]))
        else:
            prompt_parameters = generation.PromptParameters(weight=text_prompt["weight"])
            request.prompt.append(
                generation.Prompt(text=text_prompt["text"], parameters=prompt_parameters)
            )
    if init_image_prompt is not None:
        request.prompt.append(init_image_prompt)
        if mask_image_prompt is not None:
            request.prompt.append(mask_image_prompt)

    if req.get("extras") is not None:        
        for extra_key, extra_value in req["extras"].items():
            request.extras[extra_key] = extra_value
    if req.get("style_preset") is not None:
        request.extras["$IPC"] = {"preset": str(req["style_preset"])}    
    
    return request


def get_finish_reason(reason: generation.FinishReason) -> str:
    mappings = {
        generation.NULL: "SUCCESS",
        generation.FILTER: "CONTENT_FILTERED",
        generation.ERROR: "ERROR",
        generation.LENGTH: "ERROR",
    }
    return mappings.get(reason, "unknown")

def CreateResponse(answer: generation.Answer) -> Dict[str, any]:
    """Converts a protobuf Answer to a JSON response."""    
    images = []
    error_id = answer.answer_id
    for artifact in answer.artifacts:
        if artifact.type == generation.ARTIFACT_TEXT:
            if artifact.finish_reason == generation.ERROR:
                return {"result": "error", "id": error_id, "name": "generation_error", "message": generation.text }
            if artifact.finish_reason == generation.FILTER:
                return {"result": "error", "id": error_id, "name": "invalid_prompts", "message": "One or more prompts contains filtered words."}
        if artifact.type == generation.ARTIFACT_IMAGE:
            artifact_b64 = base64.b64encode(artifact.binary).decode("utf-8")
            image = {'base64': artifact_b64, 'seed': artifact.seed, 'finishReason': get_finish_reason(artifact.finish_reason)}
            images.append(image)
    response = {"result": "success", "artifacts": images}
    return response

def CreateRequest(json: Dict[str, any]) -> generation.Request:
    """Converts a JSON request to a protobuf Request."""
    return api_request_to_proto(json_to_api_request(json))

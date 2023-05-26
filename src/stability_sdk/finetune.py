import logging
import mimetypes
from enum import Enum
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import stability_sdk.interfaces.gooseai.finetuning.finetuning_pb2 as finetuning
import stability_sdk.interfaces.gooseai.finetuning.finetuning_pb2_grpc as finetuning_grpc
import stability_sdk.interfaces.gooseai.project.project_pb2 as project
import stability_sdk.interfaces.gooseai.project.project_pb2_grpc as project_grpc

from .api import generation, generation_grpc, open_channel
from .utils import image_to_prompt

TRAINING_IMAGE_MAX_COUNT = 64
TRAINING_IMAGE_MIN_COUNT = 4

TRAINING_IMAGE_MAX_SIZE = 1024
TRAINING_IMAGE_MIN_SIZE = 384


#==============================================================================
# Types
#==============================================================================

class FineTuneMode(str, Enum):
    NONE   = "none"
    FACE   = "face"
    STYLE  = "style"
    OBJECT = "object"

class FineTuneStatus(str, Enum):
    NOT_STARTED = "not started"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    SUBMITTED   = "submitted"

class FineTuneModel(BaseModel):
    id: str = Field(description="UUID")
    name: str = Field(description="Name for the fine tuned model")
    mode: FineTuneMode = Field(description="Mode for the fine tuning")
    object_name: Optional[str] = Field(description="Name of the object to fine tune")
    project_id: str = Field(description="Project ID to fine tune")
    engine_id: str = Field(description="Engine ID to fine tune")
    user_id: str = Field(description="ID of the user who created the model")
    duration: Optional[float] = Field(description="Duration of the fine tuning")
    status: Optional[FineTuneStatus] = Field(description="Status of the fine tuning")

class FineTuneParameters(BaseModel):
    name: str = Field(description="Name for the fine tuned model")
    mode: FineTuneMode = Field(description="Mode for the fine tuning")
    object_name: Optional[str] = Field(description="Name of the object to fine tune")
    engine_id: str = Field(description="Engine ID to fine tune")

class Context:
    def __init__(self, host: str, key: str):
        channel = open_channel(host, key)
        self._stub_finetune = finetuning_grpc.FineTuningServiceStub(channel)
        self._stub_generation = generation_grpc.GenerationServiceStub(channel)
        self._stub_project = project_grpc.ProjectServiceStub(channel)

FINETUNE_MODE_MAP = {
    FineTuneMode.NONE: finetuning.FINE_TUNING_MODE_UNSPECIFIED,
    FineTuneMode.FACE: finetuning.FINE_TUNING_MODE_FACE,
    FineTuneMode.STYLE: finetuning.FINE_TUNING_MODE_STYLE,
    FineTuneMode.OBJECT: finetuning.FINE_TUNING_MODE_OBJECT,
}

FINETUNE_STATUS_MAP = {
    FineTuneStatus.NOT_STARTED: finetuning.FINE_TUNING_STATUS_NOT_STARTED,
    FineTuneStatus.RUNNING:     finetuning.FINE_TUNING_STATUS_RUNNING,
    FineTuneStatus.COMPLETED:   finetuning.FINE_TUNING_STATUS_COMPLETED,
    FineTuneStatus.FAILED:      finetuning.FINE_TUNING_STATUS_FAILED,
    FineTuneStatus.SUBMITTED:   finetuning.FINE_TUNING_STATUS_SUBMITTED,
}


#==============================================================================
# Core fine-tuning functions
#==============================================================================

def create_model(
    context: Context, 
    params: FineTuneParameters, 
    image_paths: List[str]
) -> FineTuneModel:
    
    # Validate number of images
    if len(image_paths) > TRAINING_IMAGE_MAX_COUNT:
        raise ValueError(f"Too many images, please use at most {TRAINING_IMAGE_MAX_COUNT}")
    if len(image_paths) < TRAINING_IMAGE_MIN_COUNT:
        raise ValueError(f"Too few images, please use at least {TRAINING_IMAGE_MIN_COUNT}")
    
    # Load and validate images
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if min(image.width, image.height) < TRAINING_IMAGE_MIN_SIZE:
            raise ValueError(f"Image {image_path} is too small, please use images with dimensions at least 384x384")
        if max(image.width, image.height) > TRAINING_IMAGE_MAX_SIZE:
            logging.warning(f"Image {image_path} is too large, resizing to max dimension {TRAINING_IMAGE_MAX_SIZE}")
            max_size = max(image.width, image.height)
            scale = TRAINING_IMAGE_MAX_SIZE / max_size
            image = image.resize((int(image.width * scale), int(image.height * scale)), resample=Image.LANCZOS)
            images.append(image)
        else:
            images.append(None)

    # Create project
    request = project.CreateProjectRequest(
        title=params.name, 
        project_type=project.PROJECT_TYPE_TRAINING,
        access=project.PROJECT_ACCESS_PRIVATE, 
        status=project.PROJECT_STATUS_ACTIVE
    )
    proj: project.Project = context._stub_project.Create(request)
    logging.info(f"Created project {proj.id}")

    # Upload images
    for i, image in enumerate(images):
        logging.info(f"Uploading image {image_paths[i]} to project {proj.id}")

        if image is None:
            # Directly use the file from disk if it was already the right size
            with open(image_paths[i], 'rb') as f:
                bytes = f.read()
            mime_type, _ = mimetypes.guess_type(image_paths[i])
            prompt = generation.Prompt(artifact=generation.Artifact(
                type=generation.ARTIFACT_IMAGE, 
                binary=bytes,
                mime=mime_type
            ))
        else:
            # Encode the resized image
            prompt = image_to_prompt(image)

        request = generation.Request(
            engine_id="aws-asset-service",
            prompt=[prompt],
            asset=generation.AssetParameters(
                action=generation.ASSET_PUT, 
                project_id=proj.id, 
                use=generation.ASSET_USE_INPUT
            )
        )
        for response in context._stub_generation.Generate(request):
            for artifact in response.artifacts:
                if artifact.type == generation.ARTIFACT_TEXT:
                    logging.info(f"Uploaded image {i}: {artifact.text}")
    
    # Create fine tuning model
    request = finetuning.CreateModelRequest(
        name=params.name,
        mode=mode_to_proto(params.mode),
        object_name=params.object_name,
        project_id=proj.id,
        engine_id=params.engine_id,
    )
    result = context._stub_finetune.CreateModel(request)
    return model_from_proto(result.model)

def delete_model(context: Context, model_id: str) -> FineTuneModel:
    request = finetuning.DeleteModelRequest(id=model_id)
    result = context._stub_finetune.DeleteModel(request)
    return model_from_proto(result.model)

def get_model(context: Context, model_id: str) -> FineTuneModel:
    request = finetuning.GetModelRequest(id=model_id)
    result = context._stub_finetune.GetModel(request)
    return model_from_proto(result.model)

def list_models(context: Context, org_id: str) -> List[FineTuneModel]:
    request = finetuning.ListModelsRequest(org_id=org_id)
    result = context._stub_finetune.ListModels(request)
    return [model_from_proto(model) for model in result.models]

def resubmit_model(context: Context, model_id: str) -> FineTuneModel:
    request = finetuning.ResubmitModelRequest(id=model_id)
    result = context._stub_finetune.ResubmitModel(request)
    return model_from_proto(result.model)

def update_model(
    context: Context, 
    model_id: str,
    name: Optional[str] = None,
    mode: Optional[FineTuneMode] = None,
    object_name: Optional[str] = None,
    engine_id: Optional[str] = None
) -> FineTuneModel:
    request = finetuning.UpdateModelRequest(
        id=model_id,
        name=name,
        mode=mode_to_proto(mode) if mode is not None else None,
        object_name=object_name,
        engine_id=engine_id,
    )
    result = context._stub_finetune.UpdateModel(request)
    return model_from_proto(result.model)


#==============================================================================
# Utility functions
#==============================================================================

def mode_from_proto(mode: finetuning.FineTuningMode) -> FineTuneMode:
    for key, value in FINETUNE_MODE_MAP.items():
        if value == mode:
            return key
    logging.warning(f"Unrecognized fine tuning mode {mode}")
    return FineTuneMode.NONE

def mode_to_proto(mode: FineTuneMode) -> finetuning.FineTuningMode:
    mapping = {
        FineTuneMode.NONE: finetuning.FINE_TUNING_MODE_UNSPECIFIED,
        FineTuneMode.FACE: finetuning.FINE_TUNING_MODE_FACE,
        FineTuneMode.STYLE: finetuning.FINE_TUNING_MODE_STYLE,
        FineTuneMode.OBJECT: finetuning.FINE_TUNING_MODE_OBJECT,
    }
    value = mapping.get(mode)
    if value is None:
        raise ValueError(f"Invalid fine tuning mode {mode}")
    return value

def model_from_proto(model: finetuning.FineTuningModel) -> FineTuneModel:
    return FineTuneModel(
        id=model.id,
        name=model.name,
        mode=mode_from_proto(model.mode),
        object_name=model.object_name,
        project_id=model.project_id,
        engine_id=model.engine_id,
        user_id=model.user_id,
        duration=model.duration,
        status=status_from_proto(model.status),
    )

def status_from_proto(status: finetuning.FineTuningStatus) -> FineTuneStatus:
    for key, value in FINETUNE_STATUS_MAP.items():
        if value == status:
            return key
    logging.warning(f"Unrecognized fine tuning status {status}")
    return FineTuneStatus.NOT_STARTED

def status_to_proto(status: FineTuneStatus) -> finetuning.FineTuningStatus:
    value = FINETUNE_STATUS_MAP.get(status)
    if value is None:
        raise ValueError(f"Invalid fine tuning status {status}")
    return value

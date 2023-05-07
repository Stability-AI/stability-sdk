from enum import Enum
from sagemaker import ModelPackage, Predictor
from stability_sdk.api import GenerationRequest, GenerationResponse
from stability_sdk_sagemaker.predictor import StabilityPredictor
from typing import Optional, Any
import json


class StabilityModelPackage(ModelPackage):
    endpoint_name: Optional[str] = None
    sagemaker_session: Optional[Any] = None
    predictor: Optional[Predictor] = None

    def __init__(
        self,
        role,
        model_package_arn: str,
        sagemaker_session=None,
        existing_endpoint_name=None,
    ):
        super().__init__(
            role=role,
            model_package_arn=model_package_arn,
            sagemaker_session=sagemaker_session,
        )
        self.sagemaker_session = sagemaker_session
        if existing_endpoint_name is not None:
            self.attach(existing_endpoint_name)

    def deploy(
        self,
        initial_instance_count,
        instance_type="ml.g5.xlarge",
        endpoint_name=None,
        **kwargs
    ):
        self.endpoint_name = endpoint_name
        return super().deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            predictor_cls=StabilityPredictor,
            wait=True,
            **kwargs
        )        

    def attach(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.predictor = StabilityPredictor(
            endpoint_name=endpoint_name, sagemaker_session=self.sagemaker_session
        )
        return self.predictor
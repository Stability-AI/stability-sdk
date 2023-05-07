from enum import Enum
from sagemaker import ModelPackage, Predictor
from stability_sdk.api import GenerationRequest, CreateRequest, CreateResponse
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
        predictor = super().deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=True,
            **kwargs
        )
        if predictor is None:
            # This does not get set for some reason
            predictor = Predictor(
                endpoint_name=endpoint_name, sagemaker_session=self.sagemaker_session
            )
        self.predictor = predictor
        return predictor

    def attach(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.predictor = Predictor(
            endpoint_name=endpoint_name, sagemaker_session=self.sagemaker_session
        )
        return self.predictor

    def predict(self, input: GenerationRequest):
        if not self.endpoint_name or not self.predictor:
            raise ValueError("Model not deployed")
        data = input.json(exclude_unset=True)
        print(data)
        endpoint_args = {
            "ContentType": "application/json",
            "Accept": "application/json;png",
        }
        return json.loads(self.predictor.predict(data=data, initial_args=endpoint_args))

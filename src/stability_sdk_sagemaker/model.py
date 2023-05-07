from enum import Enum
from sagemaker import ModelPackage, Predictor
from stability_sdk.api import GenerationRequest, CreateRequest, CreateResponse

class StabilityModel(Enum):
    SDXL_BETA_V2_2_2 = "stable-diffusion-xl-beta-v2-2-2"

modelArns = { SDXL_BETA_V2_2_2: "arn:aws:sagemaker:us-east-1:865070037744:model-package/stable-diffusion-xl-beta-v2-2-2"}

supportedInstanceTypes 

class StabilityModelPackage(ModelPackage):
    def __init__(self, role, model:StabilityModel, sagemaker_session=None):
        super().__init__(
            role=role,
            model_package_arn=modelArns[model],
            sagemaker_session=sagemaker_session
        )         

    def deploy(self, initial_instance_count, instance_type='ml.g5.xlarge', endpoint_name=None, **kwargs):        
        self.endpoint:Predictor = super().deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            **kwargs
        )
        return self.endpoint

    def predict(self, input:GenerationRequest, endpoint_name=None):
        data = CreateRequest(input)
        return CreateResponse(self.endpoint.predict(data=data))
import json
from sagemaker import Predictor
from stability_sdk.api import GenerationRequest, GenerationResponse

class StabilityPredictor(Predictor):
    def predict(self, data: GenerationRequest):
        endpoint_args = {
            "ContentType": "application/json",
            "Accept": "application/json;png",
        }
        return GenerationResponse.parse_obj(json.loads(super().predict(data.json(exclude_unset=True), initial_args=endpoint_args)))


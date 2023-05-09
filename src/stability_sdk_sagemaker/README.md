# Stability AI Sagemaker SDK
Provides helper functions for using Stability AI provided models with the [Sagemaker SDK]([https://sagemaker.readthedocs.io/en/stable/](https://github.com/aws/sagemaker-python-sdk))

*This branch is in **alpha** status and may have breaking changes in the future. Models are not yet available to the public.*

## Usage
Instructions are intended for Sagemaker Studio. You will need to manage your connection and roles yourself outside of that environment. 
(A notebook with examples is forthcoming)

### Install from this branch
```
!pip install git+https://git@github.com/Stability-AI/stability-sdk.git@palp/sagemaker#egg=stability_sdk[sagemaker]
```

### Import libraries
```
import sagemaker
from sagemaker import ModelPackage, get_execution_role
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk_sagemaker.models import get_model_package_arn
from stability_sdk.api import GenerationRequest,GenerationResponse,TextPrompt
```

### Retrieve the ARN for a model package
This retrieves the latest version of a published marketplace model ARN by name.
```
package_arn = get_model_package_arn(model_package_name='stable-diffusion-xl-beta-v2-2-2')
```


### Deploy a model
Passing `StabilityPredictor` into the `ModelPackage` call will allow `deploy` to return a predictor, if you omit this you will need
to create one yourself or invoke the endpoint another way.

```
session = sagemaker.Session()
role = get_execution_role()

# Change this value to a unqiue name you choose
endpoint_name = 'stable-diffusion-endpoint-name' 

model = ModelPackage(role=role, model_package_arn=package_arn, sagemaker_session=session, predictor_cls=StabilityPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.g5.xlarge', endpoint_name=endpoint_name)
```

### Run inference
```
request = GenerationRequest(text_prompts=[TextPrompt(text="A majestic goose flying in space")])
result = predictor.predict(request)

image_base64 = result.artifacts[0].base64
```

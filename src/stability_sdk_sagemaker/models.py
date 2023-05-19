
marketplace_aws_account_id = '865070037744'

marketplace_model_packages = {
    'stable-diffusion-xl-beta-v2-2-2': 'stable-diffusion-xl-beta-v2-2--80252c537a5a396280cdd21b5f8b298e'
}

def get_model_package_arn(model_package_name, region_name='us-east-1', account_id=marketplace_aws_account_id):
    if model_package_name in marketplace_model_packages:
        return f"arn:aws:sagemaker:{region_name}:{account_id}:model-package/{marketplace_model_packages[model_package_name]}"
    raise Exception('Model name not found')
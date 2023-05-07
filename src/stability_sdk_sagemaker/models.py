from boto3.session import Session

stability_aws_account_id = "188650660114"
stability_aws_account_region = "us-east-1"

def get_model_package_arn(model_package_name, region_name=stability_aws_account_region, account_id=stability_aws_account_id):
    return f"arn:aws:sagemaker:{region_name}:{account_id}:model-package/{model_package_name}"
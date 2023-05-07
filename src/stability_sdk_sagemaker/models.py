from boto3.session import Session

stability_aws_account_id = "188650660114"
stability_aws_account_region = "us-east-1"

def GetModelPackageArn(model_package_name, use_local_package=False, boto_session=None):
    if use_local_package:
        if boto_session is None:
            boto_session = Session(region_name="us-east-1")
        local_package_region = boto_session.region_name                                
        account_id = boto_session.client("sts").get_caller_identity()["Account"]
        return f"arn:aws:sagemaker:{local_package_region}:{account_id}:model-package/{model_package_name}"
    else:
        return f"arn:aws:sagemaker:us-east-1:{stability_aws_account_id}:model-package/{model_package_name}"
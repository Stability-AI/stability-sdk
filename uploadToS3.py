import boto3
from pprint import pprint
import pathlib
import os
import io

def upload_file_using_client(image, object_name):
    """
    Uploads file to S3 bucket using S3 client object
    :return: None
    """
    s3 = boto3.client("s3")
    bucket_name = "dev-generated-images"
    #file_name = os.path.join(pathlib.Path(__file__).parent.resolve(), image_name)
    response = s3.upload_fileobj(io.BytesIO(image), bucket_name, object_name, ExtraArgs={'ContentType':'image/png'})
    return response
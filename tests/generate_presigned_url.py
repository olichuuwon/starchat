import boto3
from botocore.client import Config

# Replace these with your MinIO values
minio_endpoint = "https://minio-api-starquery1.apps.nebula.sl"
access_key = "minio"
secret_key = "minio123"

# Initialize the MinIO client with Boto3 (using S3-compatible API)
s3 = boto3.client(
    "s3",
    endpoint_url=minio_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version="s3v4"),
)

# Bucket and file details
bucket_name = "gguf-files"
object_name = "Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

# Generate the presigned URL for a GET request
url = s3.generate_presigned_url(
    "get_object", Params={"Bucket": bucket_name, "Key": object_name}, ExpiresIn=3600
)  # URL expires in 1 hour

print("Presigned URL:", url)

"""
https://minio-api-starquery1.apps.nebula.sl/gguf-files/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20240912%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240912T074054Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=e74a476da10c0ee4f57c82d950db8a1411981cb67074521761f093317ff32765
"""

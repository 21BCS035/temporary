import boto3
import os
from datetime import datetime, timedelta, timezone

def get_service_req_counts(region):
    client = boto3.client(
        'cloudwatch',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', region)
    )
    
    # Define time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)  # Last 30 days

    # Define metrics and their dimensions
    services_metrics = {
        "S3": [
            {
                "Namespace": "AWS/S3",
                "MetricName": "NumberOfObjects",
                "Dimensions": [
                    {"Name": "BucketName", "Value": 'elasticbeanstalk-ap-south-1-975050313614'},
                    {"Name": "StorageType", "Value": "AllStorageTypes"}
                ],
            },
            {
                "Namespace": "AWS/S3",
                "MetricName": "BucketSizeBytes",
                "Dimensions": [
                    {"Name": "BucketName", "Value": 'elasticbeanstalk-ap-south-1-975050313614'},
                    {"Name": "StorageType", "Value": "StandardStorage"}
                ],
            }
        ],
        "EC2": [
            {
                "Namespace": "AWS/EC2",
                "MetricName": "CPUUtilization",
                "Dimensions": [
                    {"Name": "InstanceId", "Value": "i-024f191db839599ae"}
                ],
            },
            {
                "Namespace": "AWS/EC2",
                "MetricName": "NetworkIn",
                "Dimensions": [
                    {"Name": "InstanceId", "Value": "i-024f191db839599ae"}
                ],
            },
            {
                "Namespace": "AWS/EC2",
                "MetricName": "NetworkOut",
                "Dimensions": [
                    {"Name": "InstanceId", "Value": "i-024f191db839599ae"}
                ],
            }
        ]
    }

    result_counts = {}

    for service, metrics in services_metrics.items():
        result_counts[service] = {}
        for metric in metrics:
            try:
                response = client.get_metric_statistics(
                    Namespace=metric["Namespace"],
                    MetricName=metric["MetricName"],
                    Dimensions=metric.get("Dimensions", []),
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400,  # 1-day interval
                    Statistics=["Average"]
                )

                # Sum all datapoints
                count = sum(r.get('Average', 0) for r in response.get('Datapoints', []))
                result_counts[service][metric["MetricName"]] = count

            except Exception as e:
                print(f"Error fetching metrics for {service} - {metric['MetricName']}: {e}")
                result_counts[service][metric["MetricName"]] = None

    return result_counts
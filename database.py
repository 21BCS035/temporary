import boto3
import os
from datetime import datetime
from decimal import Decimal
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

class Database:
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.create_tables()

    def create_tables(self):
        """
        Creates required tables if they don't exist.
        Returns silently if tables already exist.
        """
        try:
            # Check if table exists
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']
            
            if 'UserUsage' not in existing_tables:
                self.dynamodb.create_table(
                    TableName='UserUsage',
                    KeySchema=[
                        {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'user_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                # Wait for table to be created
                table = self.dynamodb.Table('UserUsage')
                table.wait_until_exists()
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceInUseException':
                raise e

    def get_table(self, table_name):
        return self.dynamodb.Table(table_name)

    def log_usage(self, user_id, usage_type, amount, cost):
        table = self.get_table('UserUsage')
        timestamp = datetime.utcnow().isoformat()
        
        try:
            table.put_item(Item={
                'user_id': user_id,
                'timestamp': timestamp,
                'usage_type': usage_type,
                'amount': Decimal(str(amount)),  # Convert to Decimal for DynamoDB
                'cost': Decimal(str(cost))       # Convert to Decimal for DynamoDB
            })
        except ClientError as e:
            print(f"Error logging usage: {str(e)}")
            raise e
    def get_user_usage(self, user_id, start_date=None, end_date=None):
        table = self.get_table('UserUsage')
        
        # Query parameters
        query_params = {
            'KeyConditionExpression': Key('user_id').eq(user_id)
        }
        
        try:
            response = table.query(**query_params)
            
            # If start and end dates are provided, filter the items
            if start_date and end_date:
                # Convert start and end dates to full timestamp range
                start_timestamp = f"{start_date}T00:00:00"
                end_timestamp = f"{end_date}T23:59:59.999999"
                
                filtered_items = [
                    item for item in response['Items']
                    if start_timestamp <= item['timestamp'] <= end_timestamp
                ]
                
                return filtered_items
            
            return response['Items']
        
        except ClientError as e:
            print(f"Error getting user usage: {str(e)}")
            return []
    # def get_user_usage(self, user_id, start_date=None, end_date=None):
    #     table = self.get_table('UserUsage')
        
    #     try:
    #         # Initial scan
    #         response = table.scan(
    #             FilterExpression=Key('user_id').eq(user_id)
    #         )
    #         print(response)
    #         print(response['Items'])
    #         print(response['Items'][0])
    #         # print(start_date)
    #         # print(end_date)
    #         # If start and end dates are provided, filter further
    #         if start_date and end_date:
    #             filtered_items = [
    #                 item for item in response['Items']
    #                 if start_date <= item['timestamp'] <= end_date
    #             ]
    #             return filtered_items
            
    #         return response['Items']
    #     except ClientError as e:
    #         print(f"Error getting user usage: {str(e)}")
    #         return []

    def get_daily_usage(self, user_id, start_date=None, end_date=None):
        """
        Get aggregated daily usage for visualization
        """
        raw_usage = self.get_user_usage(user_id, start_date, end_date)
        daily_usage = {}
        
        for item in raw_usage:
            date = item['timestamp'].split('T')[0]  # Extract date part
            usage_type = item['usage_type']
            
            if date not in daily_usage:
                daily_usage[date] = {'chat': 0, 'image': 0, 'storage': 0}
                
            daily_usage[date][usage_type] += float(item['amount'])
            
        return daily_usage
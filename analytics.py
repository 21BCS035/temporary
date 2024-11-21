from decimal import Decimal

class Analytics:
    def __init__(self, db):
        self.db = db
        self.CHAT_COST_PER_TOKEN = Decimal('0.000001')  # $0.001 per 1000 tokens
        self.IMAGE_GENERATION_COST = Decimal('0.02')     # $0.02 per image
        self.STORAGE_COST_PER_GB = Decimal('0.023')     # $0.023 per GB per month

    def log_chat_usage(self, user_id, tokens_used):
        cost = Decimal(str(tokens_used)) * self.CHAT_COST_PER_TOKEN
        self.db.log_usage(user_id, 'chat', tokens_used, float(cost))
        return cost

    def log_image_usage(self, user_id):
        self.db.log_usage(user_id, 'image_generation', 1, float(self.IMAGE_GENERATION_COST))
        return self.IMAGE_GENERATION_COST

    def log_storage_usage(self, user_id, storage_gb):
        cost = Decimal(str(storage_gb)) * self.STORAGE_COST_PER_GB
        self.db.log_usage(user_id, 'storage', storage_gb, float(cost))
        return cost

    def get_usage_summary(self, user_id, start_date=None, end_date=None):
        usage_data = self.db.get_user_usage(user_id, start_date, end_date)
        
        summary = {
            'total_tokens': 0,
            'total_images': 0,
            'total_storage_gb': 0,
            'total_cost': 0.0,
            'chat_cost': 0.0,
            'image_cost': 0.0,
            'storage_cost': 0.0
        }
        
        for item in usage_data:
            # Convert Decimal to float for calculations
            amount = float(item.get('amount', 0))
            cost = float(item.get('cost', 0))
            usage_type = item.get('usage_type')

            if usage_type == 'chat':
                summary['total_tokens'] += amount
                summary['chat_cost'] += cost
            elif usage_type == 'image_generation':
                summary['total_images'] += amount
                summary['image_cost'] += cost
            elif usage_type == 'storage':
                summary['total_storage_gb'] += amount
                summary['storage_cost'] += cost
            
            summary['total_cost'] += cost
        
        return summary
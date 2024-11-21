# load_testing.py
import concurrent.futures
import time
import uuid
import statistics
from typing import List, Dict
from decimal import Decimal
from datetime import datetime

class LoadTester:
    def __init__(self, database):
        self.db = database

    def run_load_test(self, 
                     num_users: int, 
                     duration_seconds: int, 
                     endpoints: List[str]) -> Dict:
        """Run a load test with specified parameters"""
        test_id = str(uuid.uuid4())
        start_time = time.time()
        results = []

        def test_endpoint(endpoint: str) -> Dict:
            try:
                start = time.time()
                # Simulate API call - replace with actual endpoint testing
                time.sleep(0.1)  
                latency = time.time() - start
                return {'endpoint': endpoint, 'latency': latency, 'success': True}
            except Exception as e:
                return {'endpoint': endpoint, 'error': str(e), 'success': False}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            while time.time() - start_time < duration_seconds:
                futures = []
                for endpoint in endpoints:
                    futures.append(executor.submit(test_endpoint, endpoint))
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Log metrics to DynamoDB
                    self.db.load_test_table.put_item(Item={
                        'test_id': test_id,
                        'timestamp': datetime.utcnow().isoformat(),
                        'endpoint': result['endpoint'],
                        'latency': Decimal(str(result.get('latency', 0))),
                        'success': result['success']
                    })

        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        latencies = [r['latency'] for r in successful_results]
        
        stats = {
            'test_id': test_id,
            'total_requests': len(results),
            'successful_requests': len(successful_results),
            'avg_latency': statistics.mean(latencies) if latencies else 0,
            'p95_latency': statistics.quantiles(latencies, n=20)[18] if latencies else 0,
            'p99_latency': statistics.quantiles(latencies, n=100)[98] if latencies else 0,
            'duration': duration_seconds
        }

        return stats

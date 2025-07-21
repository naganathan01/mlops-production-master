import argparse
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import json

class LoadTester:
    """Performance testing for ML API"""
    
    def __init__(self, base_url: str, concurrent_users: int = 10):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.results = []
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, payload: dict) -> Dict:
        """Make a single request and record metrics"""
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                end_time = time.time()
                response_data = await response.text()
                
                return {
                    'success': response.status == 200,
                    'status_code': response.status,
                    'response_time': end_time - start_time,
                    'response_size': len(response_data)
                }
        except Exception as e:
            end_time = time.time()
            return {
                'success': False,
                'status_code': 0,
                'response_time': end_time - start_time,
                'error': str(e)
            }
    
    async def user_simulation(self, session: aiohttp.ClientSession, duration: int):
        """Simulate a single user's requests for the given duration"""
        end_time = time.time() + duration
        user_results = []
        
        while time.time() < end_time:
            # Simulate prediction request
            payload = {
                "features": {
                    f"feature_{i}": float(i * 0.1) 
                    for i in range(10)
                }
            }
            
            result = await self.make_request(session, "/predict", payload)
            user_results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        return user_results
    
    async def run_load_test(self, duration: int = 60) -> Dict:
        """Run load test with multiple concurrent users"""
        print(f"Starting load test: {self.concurrent_users} users for {duration}s")
        
        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create tasks for concurrent users
            tasks = [
                self.user_simulation(session, duration)
                for _ in range(self.concurrent_users)
            ]
            
            # Run all user simulations concurrently
            start_time = time.time()
            user_results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # Flatten results
            all_results = []
            for user_result in user_results:
                all_results.extend(user_result)
        
        return self.analyze_results(all_results, total_time)
    
    def analyze_results(self, results: List[Dict], total_time: float) -> Dict:
        """Analyze load test results"""
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = total_requests - successful_requests
        
        response_times = [r['response_time'] for r in results if r['success']]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = p95_response_time = p99_response_time = 0
        
        throughput = successful_requests / total_time
        error_rate = (failed_requests / total_requests) * 100
        
        results_summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'error_rate_percent': error_rate,
            'throughput_rps': throughput,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'total_duration': total_time
        }
        
        # Performance assertions
        assert error_rate < 5.0, f"Error rate too high: {error_rate}%"
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time}s"
        assert p95_response_time < 5.0, f"P95 response time too high: {p95_response_time}s"
        assert throughput > 10, f"Throughput too low: {throughput} rps"
        
        return results_summary

async def main():
    parser = argparse.ArgumentParser(description='ML API Load Testing')
    parser.add_argument('--endpoint', required=True, help='API endpoint URL')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--users', type=int, default=10, help='Concurrent users')
    args = parser.parse_args()
    
    tester = LoadTester(args.endpoint, args.users)
    results = await tester.run_load_test(args.duration)
    
    print("\n=== Load Test Results ===")
    print(json.dumps(results, indent=2))
    
    print(f"\n✅ Load test completed successfully!")
    print(f"📊 Processed {results['successful_requests']} requests")
    print(f"🚀 Throughput: {results['throughput_rps']:.1f} requests/sec")
    print(f"⏱️  Avg Response Time: {results['avg_response_time']:.3f}s")
    print(f"📈 P95 Response Time: {results['p95_response_time']:.3f}s")
    print(f"❌ Error Rate: {results['error_rate_percent']:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
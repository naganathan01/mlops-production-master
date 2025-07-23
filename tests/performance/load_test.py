# tests/performance/load_test.py
import argparse
import json
import time
import random
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests
from dataclasses import dataclass

@dataclass
class LoadTestResult:
    """Load test result data"""
    response_time: float
    status_code: int
    success: bool
    error: str = None

class LoadTester:
    """ML API Load Testing Framework"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results: List[LoadTestResult] = []
        
    def generate_test_features(self) -> Dict[str, float]:
        """Generate random test features"""
        return {
            f"feature_{i}": random.normalvariate(0, 1) 
            for i in range(10)
        }
    
    def make_prediction_request(self) -> LoadTestResult:
        """Make a single prediction request"""
        start_time = time.time()
        
        try:
            payload = {
                "features": self.generate_test_features(),
                "model_version": "latest"
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            return LoadTestResult(
                response_time=response_time,
                status_code=response.status_code,
                success=response.status_code == 200
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LoadTestResult(
                response_time=response_time,
                status_code=0,
                success=False,
                error=str(e)
            )
    
    def make_batch_request(self, batch_size: int = 10) -> LoadTestResult:
        """Make a batch prediction request"""
        start_time = time.time()
        
        try:
            payload = {
                "requests": [
                    {
                        "features": self.generate_test_features(),
                        "model_version": "latest"
                    }
                    for _ in range(batch_size)
                ]
            }
            
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload,
                timeout=60
            )
            
            response_time = time.time() - start_time
            
            return LoadTestResult(
                response_time=response_time,
                status_code=response.status_code,
                success=response.status_code == 200
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LoadTestResult(
                response_time=response_time,
                status_code=0,
                success=False,
                error=str(e)
            )
    
    def run_load_test(self, 
                     duration: int = 60, 
                     concurrent_users: int = 10,
                     test_type: str = "single") -> Dict[str, Any]:
        """Run load test for specified duration"""
        
        print(f"Starting load test:")
        print(f"  Duration: {duration} seconds")
        print(f"  Concurrent users: {concurrent_users}")
        print(f"  Test type: {test_type}")
        print(f"  Target: {self.base_url}")
        
        start_time = time.time()
        end_time = start_time + duration
        
        results = []
        
        def worker():
            """Worker function for each concurrent user"""
            while time.time() < end_time:
                if test_type == "single":
                    result = self.make_prediction_request()
                elif test_type == "batch":
                    result = self.make_batch_request()
                else:
                    # Mixed workload
                    if random.random() < 0.8:  # 80% single, 20% batch
                        result = self.make_prediction_request()
                    else:
                        result = self.make_batch_request(batch_size=5)
                
                results.append(result)
                
                # Small delay to avoid overwhelming the server
                time.sleep(random.uniform(0.01, 0.1))
        
        # Start concurrent workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_users)]
            
            # Monitor progress
            while time.time() < end_time:
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                completed = len(results)
                rps = completed / elapsed if elapsed > 0 else 0
                
                print(f"\rProgress: {elapsed:.1f}s elapsed, "
                      f"{remaining:.1f}s remaining, "
                      f"{completed} requests, "
                      f"{rps:.1f} RPS", end="", flush=True)
                
                time.sleep(1)
            
            # Wait for all workers to complete
            for future in as_completed(futures, timeout=10):
                pass
        
        print("\nLoad test completed!")
        
        # Calculate statistics
        return self._calculate_statistics(results, duration)
    
    def _calculate_statistics(self, results: List[LoadTestResult], duration: int) -> Dict[str, Any]:
        """Calculate load test statistics"""
        
        if not results:
            return {"error": "No results collected"}
        
        # Filter successful requests for response time analysis
        successful_results = [r for r in results if r.success]
        response_times = [r.response_time for r in successful_results]
        
        # Calculate metrics
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        rps = total_requests / duration
        
        # Response time statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Percentiles
            sorted_times = sorted(response_times)
            p95_response_time = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
            p99_response_time = sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0
        else:
            avg_response_time = median_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # Error analysis
        error_types = {}
        for result in results:
            if not result.success:
                error_key = f"HTTP_{result.status_code}" if result.status_code > 0 else "Connection_Error"
                error_types[error_key] = error_types.get(error_key, 0) + 1
        
        return {
            "test_duration": duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "requests_per_second": rps,
            "response_time_stats": {
                "average": avg_response_time,
                "median": median_response_time,
                "min": min_response_time,
                "max": max_response_time,
                "p95": p95_response_time,
                "p99": p99_response_time
            },
            "error_breakdown": error_types,
            "performance_grade": self._calculate_performance_grade(success_rate, avg_response_time, rps)
        }
    
    def _calculate_performance_grade(self, success_rate: float, avg_response_time: float, rps: float) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Success rate scoring (40% weight)
        if success_rate >= 0.99:
            score += 40
        elif success_rate >= 0.95:
            score += 30
        elif success_rate >= 0.90:
            score += 20
        else:
            score += 10
        
        # Response time scoring (40% weight)
        if avg_response_time <= 0.1:  # 100ms
            score += 40
        elif avg_response_time <= 0.5:  # 500ms
            score += 30
        elif avg_response_time <= 1.0:  # 1s
            score += 20
        else:
            score += 10
        
        # Throughput scoring (20% weight)
        if rps >= 100:
            score += 20
        elif rps >= 50:
            score += 15
        elif rps >= 20:
            score += 10
        else:
            score += 5
        
        # Convert to grade
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def run_stress_test(self, max_users: int = 100, ramp_up_time: int = 60) -> Dict[str, Any]:
        """Run stress test with gradual user ramp-up"""
        
        print(f"Starting stress test:")
        print(f"  Max users: {max_users}")
        print(f"  Ramp-up time: {ramp_up_time} seconds")
        
        results = []
        start_time = time.time()
        
        def worker(user_id: int, start_delay: float):
            """Worker with delayed start"""
            time.sleep(start_delay)
            
            # Run for remaining time
            end_time = start_time + ramp_up_time + 60  # Extra 60s at max load
            
            while time.time() < end_time:
                result = self.make_prediction_request()
                results.append((user_id, time.time() - start_time, result))
                time.sleep(random.uniform(0.1, 0.5))
        
        # Start workers with staggered delays
        with ThreadPoolExecutor(max_workers=max_users) as executor:
            futures = []
            
            for user_id in range(max_users):
                delay = (user_id / max_users) * ramp_up_time
                future = executor.submit(worker, user_id, delay)
                futures.append(future)
            
            # Monitor progress
            total_duration = ramp_up_time + 60
            while time.time() - start_time < total_duration:
                elapsed = time.time() - start_time
                current_users = min(max_users, int((elapsed / ramp_up_time) * max_users))
                completed = len(results)
                
                print(f"\rStress test: {elapsed:.1f}s, "
                      f"{current_users} active users, "
                      f"{completed} requests completed", end="", flush=True)
                
                time.sleep(1)
            
            # Wait for completion
            for future in as_completed(futures, timeout=30):
                pass
        
        print("\nStress test completed!")
        
        # Analyze results by time windows
        return self._analyze_stress_results(results, ramp_up_time)
    
    def _analyze_stress_results(self, results: List, ramp_up_time: int) -> Dict[str, Any]:
        """Analyze stress test results"""
        
        if not results:
            return {"error": "No stress test results"}
        
        # Group results by time windows (10-second intervals)
        window_size = 10
        windows = {}
        
        for user_id, timestamp, result in results:
            window = int(timestamp // window_size) * window_size
            if window not in windows:
                windows[window] = []
            windows[window].append(result)
        
        # Calculate metrics for each window
        window_stats = {}
        for window, window_results in windows.items():
            successful = [r for r in window_results if r.success]
            response_times = [r.response_time for r in successful]
            
            window_stats[window] = {
                "requests": len(window_results),
                "success_rate": len(successful) / len(window_results) if window_results else 0,
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "rps": len(window_results) / window_size
            }
        
        # Find performance degradation point
        degradation_point = None
        baseline_success_rate = None
        
        for window in sorted(windows.keys()):
            current_success_rate = window_stats[window]["success_rate"]
            
            if baseline_success_rate is None:
                baseline_success_rate = current_success_rate
            elif current_success_rate < baseline_success_rate * 0.9:  # 10% degradation
                degradation_point = window
                break
        
        return {
            "total_requests": len(results),
            "test_duration": ramp_up_time + 60,
            "window_statistics": window_stats,
            "performance_degradation_point": degradation_point,
            "max_sustainable_load": self._calculate_max_load(window_stats)
        }
    
    def _calculate_max_load(self, window_stats: Dict) -> Dict[str, Any]:
        """Calculate maximum sustainable load"""
        
        best_window = None
        best_rps = 0
        
        for window, stats in window_stats.items():
            if stats["success_rate"] >= 0.95 and stats["avg_response_time"] <= 1.0:
                if stats["rps"] > best_rps:
                    best_rps = stats["rps"]
                    best_window = window
        
        return {
            "max_rps": best_rps,
            "window": best_window,
            "recommended_max_users": int(best_rps * 2) if best_rps > 0 else 10
        }

def main():
    parser = argparse.ArgumentParser(description='ML API Load Testing')
    parser.add_argument('--host', required=True, help='API host URL')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--test-type', choices=['single', 'batch', 'mixed'], 
                       default='single', help='Type of test requests')
    parser.add_argument('--stress-test', action='store_true', help='Run stress test instead')
    parser.add_argument('--max-users', type=int, default=100, help='Max users for stress test')
    parser.add_argument('--ramp-up', type=int, default=60, help='Ramp-up time for stress test')
    parser.add_argument('--output', help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Validate API connectivity
    try:
        response = requests.get(f"{args.host}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ API health check failed: {response.status_code}")
            exit(1)
        print(f"✅ API connectivity verified")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        exit(1)
    
    # Initialize load tester
    tester = LoadTester(args.host)
    
    # Run appropriate test
    if args.stress_test:
        results = tester.run_stress_test(args.max_users, args.ramp_up)
    else:
        results = tester.run_load_test(args.duration, args.users, args.test_type)
    
    # Print results
    print("\n" + "="*60)
    print("LOAD TEST RESULTS")
    print("="*60)
    
    if args.stress_test:
        print(f"Total Requests: {results['total_requests']}")
        print(f"Test Duration: {results['test_duration']}s")
        if results.get('performance_degradation_point'):
            print(f"Performance Degradation: {results['performance_degradation_point']}s")
        else:
            print("Performance Degradation: None detected")
        
        max_load = results.get('max_sustainable_load', {})
        print(f"Max Sustainable RPS: {max_load.get('max_rps', 'Unknown')}")
        print(f"Recommended Max Users: {max_load.get('recommended_max_users', 'Unknown')}")
        
    else:
        print(f"Total Requests: {results['total_requests']}")
        print(f"Successful: {results['successful_requests']} ({results['success_rate']:.1%})")
        print(f"Failed: {results['failed_requests']}")
        print(f"Requests/Second: {results['requests_per_second']:.1f}")
        print(f"Performance Grade: {results['performance_grade']}")
        
        rt_stats = results['response_time_stats']
        print(f"\nResponse Time Statistics:")
        print(f"  Average: {rt_stats['average']:.3f}s")
        print(f"  Median: {rt_stats['median']:.3f}s")
        print(f"  95th Percentile: {rt_stats['p95']:.3f}s")
        print(f"  99th Percentile: {rt_stats['p99']:.3f}s")
        print(f"  Min: {rt_stats['min']:.3f}s")
        print(f"  Max: {rt_stats['max']:.3f}s")
        
        if results['error_breakdown']:
            print(f"\nError Breakdown:")
            for error_type, count in results['error_breakdown'].items():
                print(f"  {error_type}: {count}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with appropriate code
    if args.stress_test:
        exit(0)
    else:
        exit(0 if results['success_rate'] >= 0.95 else 1)

if __name__ == "__main__":
    main()
import unittest
import json
import os
import time
import threading
import requests
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from app import app
from gemini_wrapper import generate_text, generate_code, classify_text
from lru_cache import ResponseCache, RateLimitCache
from cost_tracker import CostTracker

class TestMultiTaskLLMAPI(unittest.TestCase):
    
    server_thread = None
    server_started = False
    base_url = "http://0.0.0.0:8081/api/v1"
    
    @classmethod
    def setUpClass(cls):
        """Set up live server and load environment"""
        load_dotenv()
        
        # Start the Flask server in a separate thread
        cls.server_thread = threading.Thread(target=cls._run_server, daemon=True)
        cls.server_thread.start()
        
        # Wait for server to start
        cls._wait_for_server()
        
        print(f"✅ Test server started at {cls.base_url}")
    
    @classmethod
    def _run_server(cls):
        """Run Flask server in thread"""
        try:
            app.run(host='0.0.0.0', port=8081, debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            print(f"❌ Server failed to start: {e}")
    
    @classmethod
    def _wait_for_server(cls, timeout=30):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://127.0.0.1:8081/api/v1/health", timeout=2)
                if response.status_code == 200:
                    cls.server_started = True
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
        
        raise RuntimeError(f"Server failed to start within {timeout} seconds")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        print("\n🧹 Cleaning up test environment...")
        # Note: Server thread will be cleaned up automatically as it's a daemon thread
        
    def test_01_env_api_key_configured(self):
        """Test that the GOOGLE_API_KEY is properly configured in .env"""
        print("\n🔑 Testing API Key Configuration...")
        
        # Check if .env file exists
        env_file_path = os.path.join(os.path.dirname(__file__), '.env')
        self.assertTrue(os.path.exists(env_file_path), ".env file should exist")
        
        # Check if API key is loaded
        api_key = os.getenv('GOOGLE_API_KEY')
        self.assertIsNotNone(api_key, "GOOGLE_API_KEY should be set in environment")
        self.assertTrue(len(api_key) > 0, "GOOGLE_API_KEY should not be empty")
        self.assertTrue(api_key.startswith('AIza'), "API key should start with 'AIza'")
        
        print(f"✅ API Key configured: {api_key[:10]}...{api_key[-5:]}")
        
    def test_02_generate_text_endpoint(self):
        """Test the /api/v1/generate/text endpoint"""
        print("\n📝 Testing Text Generation Endpoint...")
        
        payload = {
            "prompt": "Write a one-sentence story about a cat."
        }
        
        response = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('generated_text', data)
        self.assertIsNotNone(data['generated_text'])
        self.assertTrue(len(data['generated_text']) > 0)
        
        print(f"✅ Text generated: {data['generated_text'][:50]}...")
        
    def test_03_generate_code_endpoint(self):
        """Test the /api/v1/generate/code endpoint"""
        print("\n💻 Testing Code Generation Endpoint...")
        
        payload = {
            "prompt": "Create a simple Python function that adds two numbers"
        }
        
        response = requests.post(
            f'{self.base_url}/generate/code',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('generated_code', data)
        self.assertIsNotNone(data['generated_code'])
        self.assertTrue(len(data['generated_code']) > 0)
        self.assertIn('def', data['generated_code'])  # Should contain a function definition
        
        print(f"✅ Code generated: {data['generated_code'][:50]}...")
        
    def test_04_classify_text_endpoint(self):
        """Test the /api/v1/classify/text endpoint"""
        print("\n🏷️ Testing Text Classification Endpoint...")
        
        payload = {
            "text": "I love this amazing product! It works perfectly.",
            "categories": ["positive", "negative", "neutral"]
        }
        
        response = requests.post(
            f'{self.base_url}/classify/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('classification', data)
        self.assertIsNotNone(data['classification'])
        self.assertIn(data['classification'].lower(), ['positive', 'negative', 'neutral'])
        
        print(f"✅ Text classified as: {data['classification']}")
        
    def test_05_endpoint_error_handling(self):
        """Test error handling for missing required fields"""
        print("\n❌ Testing Error Handling...")
        
        # Test missing prompt for text generation
        response = requests.post(
            f'{self.base_url}/generate/text',
            json={},  # Empty payload - missing required 'prompt'
            headers={'Content-Type': 'application/json'}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)
        
        # Test missing fields for classification
        response = requests.post(
            f'{self.base_url}/classify/text',
            json={"text": "test"},  # missing 'categories' field
            headers={'Content-Type': 'application/json'}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('error', data)
        
        print("✅ Error handling working correctly")
    
    def test_08_lru_cache_functionality(self):
        """Test LRU cache functionality"""
        print("\n💾 Testing LRU Cache Functionality...")
        
        # Test response cache
        response_cache = ResponseCache(max_size=3, default_ttl=10)
        
        # Cache some responses
        response_cache.cache_response("test1", "text", "response1", 100, 200, 0.05)
        response_cache.cache_response("test2", "text", "response2", 150, 250, 0.075)
        response_cache.cache_response("test3", "text", "response3", 120, 220, 0.06)
        
        # Test cache hits
        cached1 = response_cache.get_cached_response("test1", "text")
        self.assertIsNotNone(cached1)
        self.assertEqual(cached1['response'], "response1")
        
        # Add one more (should evict least recently used)
        response_cache.cache_response("test4", "text", "response4", 130, 230, 0.065)
        
        # test2 should be evicted as it was least recently used
        cached2 = response_cache.get_cached_response("test2", "text")
        self.assertIsNone(cached2)
        
        print("✅ LRU Cache working correctly")
        
        # Test rate limit cache
        rate_cache = RateLimitCache(max_size=1000)
        
        # Test rate limiting
        count1 = rate_cache.increment_count("user1", 60)
        count2 = rate_cache.increment_count("user1", 60)
        count3 = rate_cache.increment_count("user2", 60)
        
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 2)
        self.assertEqual(count3, 1)
        
        print("✅ Rate limit cache working correctly")
    
    def test_09_cost_tracking(self):
        """Test cost tracking functionality"""
        print("\n💰 Testing Cost Tracking...")
        
        cost_tracker = CostTracker("logs/test_cost.log")
        
        # Track a request
        record = cost_tracker.track_request(
            "/api/v1/generate/text",
            "127.0.0.1",
            "Test prompt for cost tracking",
            "Generated response for cost tracking test"
        )
        
        self.assertIsNotNone(record)
        self.assertGreater(record.input_tokens, 0)
        self.assertGreater(record.output_tokens, 0)
        self.assertGreater(record.total_cost, 0)
        self.assertFalse(record.cached)
        
        print(f"✅ Cost tracked - Input: {record.input_tokens} tokens, Output: {record.output_tokens} tokens, Cost: ${record.total_cost:.6f}")
        
        # Get usage stats
        stats = cost_tracker.get_usage_stats(1)  # Last hour
        self.assertGreater(stats['total_requests'], 0)
        self.assertGreaterEqual(stats['total_cost'], 0)
        
        print(f"✅ Usage stats - Requests: {stats['total_requests']}, Total cost: ${stats['total_cost']:.6f}")
        
        # Close logger handlers before cleanup
        for handler in cost_tracker.logger.handlers[:]:
            handler.close()
            cost_tracker.logger.removeHandler(handler)
        
        # Cleanup test file
        if os.path.exists("logs/test_cost.log"):
            try:
                os.remove("logs/test_cost.log")
            except PermissionError:
                # File might still be locked, skip cleanup
                pass
    
    def test_10_api_headers(self):
        """Test cost and tracking headers in API responses"""
        print("\n📊 Testing API Response Headers...")
        
        payload = {
            "prompt": "Say hello in one word"
        }
        
        response = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Check for cost headers
        self.assertIn('X-Request-ID', response.headers)
        self.assertIn('X-Cost-Total', response.headers)
        self.assertIn('X-Cached', response.headers)
        
        cost_total = float(response.headers.get('X-Cost-Total', 0))
        cached = response.headers.get('X-Cached', 'False') == 'True'
        
        print(f"✅ Request ID: {response.headers.get('X-Request-ID')}")
        print(f"✅ Total cost: ${cost_total:.6f}")
        print(f"✅ Cached: {cached}")
        
        # Make the same request again to test caching
        response2 = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        cached2 = response2.headers.get('X-Cached', 'False') == 'True'
        cost_total2 = float(response2.headers.get('X-Cost-Total', 0))
        
        print(f"✅ Second request cached: {cached2}")
        print(f"✅ Second request cost: ${cost_total2:.6f}")
        
        # Second request should be cached with zero cost
        if cached2:
            self.assertEqual(cost_total2, 0.0)
            
    def test_07_direct_wrapper_functions(self):
        """Test the gemini_wrapper functions directly"""
        print("\n🔧 Testing Wrapper Functions Directly...")
        
        # Test generate_text function
        text_result = generate_text("Say 'test successful' in one sentence.")
        self.assertIsInstance(text_result, str)
        self.assertTrue(len(text_result) > 0)
        print(f"✅ Direct text generation: {text_result[:30]}...")
        
        # Test generate_code function  
        code_result = generate_code("Write a simple hello world function in Python")
        self.assertIsInstance(code_result, str)
        self.assertTrue(len(code_result) > 0)
        self.assertIn('def', code_result.lower())
        print(f"✅ Direct code generation: {code_result[:30]}...")
        
        # Test classify_text function
        classification_result = classify_text("This is great!", ["positive", "negative"])
        self.assertIsInstance(classification_result, str)
        self.assertTrue(len(classification_result) > 0)
        print(f"✅ Direct text classification: {classification_result}")
    
    def test_11_health_and_stats_endpoints(self):
        """Test health check and stats endpoints"""
        print("\n🔍 Testing Health and Stats Endpoints...")
        
        # Test health endpoint
        response = requests.get(f'{self.base_url}/health')
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        
        print("✅ Health endpoint working")
        
        # Test stats endpoint
        response = requests.get(f'{self.base_url}/stats')
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('cache_stats', data)
        self.assertIn('usage_stats', data)
        self.assertIn('rate_limit_stats', data)
        
        print("✅ Stats endpoint working")
        print(f"✅ Cache hit ratio: {data['cache_stats'].get('cache_hit_ratio', 0):.2%}")
    
    def test_12_overall_cost_tracking(self):
        """Test overall cost tracking across multiple requests"""
        print("\n💰 Testing Overall Cost Tracking...")
        
        # Make multiple requests to accumulate costs
        prompts = [
            "Say hello",
            "Count to three", 
            "Name a color"
        ]
        
        overall_costs = []
        
        for i, prompt in enumerate(prompts):
            payload = {"prompt": prompt}
            
            response = requests.post(
                f'{self.base_url}/generate/text',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            self.assertEqual(response.status_code, 200)
            
            cost_total = float(response.headers.get('X-Cost-Total', 0))
            cost_overall = float(response.headers.get('X-Cost-Overall', 0))
            overall_costs.append(cost_overall)
            
            print(f"✅ Request {i+1}: cost=${cost_total:.6f}, overall=${cost_overall:.6f}")
        
        # Overall cost should be non-decreasing
        for i in range(1, len(overall_costs)):
            self.assertGreaterEqual(overall_costs[i], overall_costs[i-1], 
                                  "Overall cost should be non-decreasing")
        
        print(f"✅ Overall cost tracking working correctly")
    
    def test_13_cache_functionality(self):
        """Test cache hit/miss and eviction using existing API"""
        print("\n💾 Testing Cache Functionality...")
        
        # Make initial request (should be cache miss)
        payload = {"prompt": "Cache test prompt 1"}
        response1 = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response1.status_code, 200)
        cached1 = response1.headers.get('X-Cached', 'False') == 'True'
        self.assertFalse(cached1, "First request should not be cached")
        print("✅ Cache miss working correctly")
        
        # Make same request again (should be cache hit)
        response2 = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response2.status_code, 200)
        cached2 = response2.headers.get('X-Cached', 'False') == 'True'
        self.assertTrue(cached2, "Second identical request should be cached")
        cost2 = float(response2.headers.get('X-Cost-Total', 0))
        self.assertEqual(cost2, 0.0, "Cached response should have zero cost")
        print("✅ Cache hit working correctly")
        
        # Verify cache stats
        stats_response = requests.get(f'{self.base_url}/stats')
        self.assertEqual(stats_response.status_code, 200)
        stats = stats_response.json()
        
        cache_stats = stats.get('cache_stats', {})
        self.assertGreater(cache_stats.get('hit_count', 0), 0, "Should have cache hits")
        print(f"✅ Cache stats: {cache_stats.get('hit_count', 0)} hits, {cache_stats.get('miss_count', 0)} misses")
    
    def test_14_log_files_presence(self):
        """Test that log files are created in logs directory"""
        print("\n📄 Testing Log Files Creation...")
        
        # Make a request to generate some logs
        payload = {"prompt": "Log test prompt"}
        response = requests.post(
            f'{self.base_url}/generate/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Check if logs directory exists
        logs_dir = os.path.join(os.getcwd(), 'logs')
        self.assertTrue(os.path.exists(logs_dir), "Logs directory should exist")
        print(f"✅ Logs directory exists: {logs_dir}")
        print(f"📝 Note: Live server running on http://0.0.0.0:8081")
        print(f"🔗 Production deployment: http://0.0.0.0:8081 (hardcoded)")
        
        # Check for expected log files
        expected_log_files = [
            'api_requests.log',
            'cost_tracking.log',
            'performance.log'
        ]
        
        existing_files = []
        for log_file in expected_log_files:
            log_path = os.path.join(logs_dir, log_file)
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                existing_files.append(log_file)
                print(f"✅ Log file exists and has content: {log_file}")
        
        self.assertGreater(len(existing_files), 0, "At least one log file should exist with content")
        
        # Check if we can read the request log
        request_log_path = os.path.join(logs_dir, 'api_requests.log')
        if os.path.exists(request_log_path):
            with open(request_log_path, 'r') as f:
                log_content = f.read()
                self.assertIn('REQUEST_START', log_content, "Request log should contain REQUEST_START entries")
                print("✅ Request log contains expected content")

if __name__ == '__main__':
    print("🚀 Starting Multi-Task LLM API Unit Tests...")
    print("📡 Testing against live server: http://0.0.0.0:8081")
    print("🔄 Server will start automatically and run tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiTaskLLMAPI)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All tests passed successfully!")
        print("🌐 Live server tested at: http://0.0.0.0:8081")
        print("📊 Swagger UI: http://0.0.0.0:8081/swagger/")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    print(f"📊 Tests run: {result.testsRun}")
    print("=" * 60)
    
    # Cleanup any test files
    test_files = ['logs/test_cost.log', 'logs/cost_tracking.log']
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

import unittest
import os
import sys
from dotenv import load_dotenv

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreEnterpriseTests(unittest.TestCase):
    """Core 5 unit tests for Enterprise LLM API with real components"""
    
    @classmethod
    def setUpClass(cls):
        """Load environment variables and validate API key"""
        load_dotenv()
        
        # Validate API key
        cls.api_key = os.getenv('GOOGLE_API_KEY')
        if not cls.api_key or not cls.api_key.startswith('AIza'):
            raise unittest.SkipTest("Valid GOOGLE_API_KEY not found in environment")
        
        print(f"Using API Key: {cls.api_key[:10]}...{cls.api_key[-5:]}")
        
        # Initialize enterprise components
        try:
            from lru_cache import LRUCache, ResponseCache
            from cost_tracker import CostTracker
            import gemini_wrapper
            
            cls.lru_cache = LRUCache(max_size=100, default_ttl=3600)
            cls.response_cache = ResponseCache(max_size=50, default_ttl=1800)
            cls.cost_tracker = CostTracker()
            cls.gemini_wrapper = gemini_wrapper
            
            print("Enterprise components loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required enterprise components not found: {e}")

    def test_01_text_generation(self):
        """Test 1: Text generation with real API"""
        print("Running Test 1: Text generation")
        
        # Test with simple prompt
        prompt = "Write a one-sentence story about a curious cat"
        result = self.gemini_wrapper.generate_text(prompt)
        
        # Assertions
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('cat', result.lower())
        
        print(f"PASS: Generated text: {result[:50]}...")

    def test_02_code_generation(self):
        """Test 2: Code generation with real API"""
        print("Running Test 2: Code generation")
        
        # Test with code prompt
        prompt = "Write a Python function to add two numbers"
        result = self.gemini_wrapper.generate_code(prompt)
        
        # Assertions
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn('def', result.lower())  # Should contain function definition
        
        print("PASS: Code generation contains function definition")

    def test_03_cache_functionality(self):
        """Test 3: LRU Cache functionality with real implementation"""
        print("Running Test 3: Cache functionality")
        
        # Test cache operations
        cache_key = "test_prompt_cache"
        cache_value = "cached_response_data"
        
        # Test cache miss
        result = self.lru_cache.get(cache_key)
        self.assertIsNone(result)
        
        # Test cache put
        self.lru_cache.put(cache_key, cache_value)
        
        # Test cache hit
        cached_result = self.lru_cache.get(cache_key)
        self.assertEqual(cached_result, cache_value)
        
        # Test cache size
        self.assertGreater(self.lru_cache.size(), 0)
        
        # Test cache stats
        stats = self.lru_cache.get_stats()
        self.assertIn('size', stats)
        self.assertIn('max_size', stats)
        self.assertIn('hit_ratio', stats)
        self.assertIn('default_ttl', stats)
        
        print(f"PASS: Cache working - Size: {stats['size']}, Hit ratio: {stats['hit_ratio']:.2%}")

    def test_04_cost_tracking(self):
        """Test 4: Cost tracking with real implementation"""
        print("Running Test 4: Cost tracking")
        
        # Test cost tracking for a request
        endpoint = "/api/v1/generate/text"
        user_ip = "127.0.0.1"
        prompt = "Test prompt for cost tracking"
        response = "Generated response for cost tracking test"
        
        # Track the request
        record = self.cost_tracker.track_request(endpoint, user_ip, prompt, response)
        
        # Assertions
        self.assertIsNotNone(record)
        self.assertGreater(record.input_tokens, 0)
        self.assertGreater(record.output_tokens, 0)
        self.assertGreaterEqual(record.total_cost, 0)
        self.assertEqual(record.endpoint, endpoint)
        self.assertEqual(record.user_ip, user_ip)
        
        # Test usage stats
        stats = self.cost_tracker.get_usage_stats(24)
        self.assertIn('total_requests', stats)
        self.assertIn('total_cost', stats)
        self.assertGreater(stats['total_requests'], 0)
        
        print(f"PASS: Cost tracked - Input: {record.input_tokens} tokens, Output: {record.output_tokens} tokens, Cost: ${record.total_cost:.6f}")
        print(f"PASS: Usage stats - Requests: {stats['total_requests']}, Total cost: ${stats['total_cost']:.6f}")

    def test_05_lru_cache_advanced(self):
        """Test 5: Advanced LRU Cache behavior"""
        print("Running Test 5: Advanced LRU Cache behavior")
        
        # Create a small ResponseCache for testing eviction (has hit/miss tracking)
        test_cache = self.response_cache.__class__(max_size=3, default_ttl=10)
        
        # Fill cache to capacity
        test_cache.put("key1", "value1")
        test_cache.put("key2", "value2")
        test_cache.put("key3", "value3")
        
        # Verify all items are cached
        self.assertEqual(test_cache.get("key1"), "value1")
        self.assertEqual(test_cache.get("key2"), "value2")
        self.assertEqual(test_cache.get("key3"), "value3")
        self.assertEqual(test_cache.size(), 3)
        
        # Access key1 to make it most recently used
        test_cache.get("key1")
        
        # Add new item (should evict key2 as it's least recently used)
        test_cache.put("key4", "value4")
        
        # Verify eviction behavior
        self.assertIsNone(test_cache.get("key2"))  # Should be evicted
        self.assertEqual(test_cache.get("key1"), "value1")  # Should still exist
        self.assertEqual(test_cache.get("key3"), "value3")  # Should still exist
        self.assertEqual(test_cache.get("key4"), "value4")  # Should exist
        self.assertEqual(test_cache.size(), 3)
        
        # Test cache statistics
        stats = test_cache.get_stats()
        self.assertIn('hit_count', stats)
        self.assertIn('miss_count', stats)
        self.assertIn('cache_hit_ratio', stats)
        
        print(f"PASS: LRU eviction working correctly")
        print(f"PASS: Cache stats - Hit ratio: {stats['cache_hit_ratio']:.2%}")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 60)
    print("[*] Core Enterprise LLM Unit Tests (5 Tests)")
    print("Testing with REAL API and Enterprise Components")
    print("=" * 60)
    
    # Check API key
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key or not api_key.startswith('AIza'):
        print("[ERROR] Valid GOOGLE_API_KEY not found!")
        return False
    
    print(f"[OK] Using API Key: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreEnterpriseTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core enterprise tests passed!")
        print("[OK] Enterprise components working correctly with real API")
        print("[OK] Text generation, Code generation, Cache, Cost tracking validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core Enterprise Tests")
    print("[*] 5 essential tests with real API and enterprise components")
    print("[*] Components: Text Gen, Code Gen, LRU Cache, Cost Tracking")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)
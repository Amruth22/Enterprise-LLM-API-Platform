#!/usr/bin/env python3
"""
Quick test runner to validate the W1D4S2 style tests
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from lru_cache import LRUCache, ResponseCache
        print("✅ LRU Cache modules imported successfully")
    except ImportError as e:
        print(f"❌ LRU Cache import failed: {e}")
        return False
    
    try:
        from cost_tracker import CostTracker
        print("✅ Cost Tracker imported successfully")
    except ImportError as e:
        print(f"❌ Cost Tracker import failed: {e}")
        return False
    
    try:
        import gemini_wrapper
        print("✅ Gemini Wrapper imported successfully")
    except ImportError as e:
        print(f"❌ Gemini Wrapper import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without API calls"""
    print("\nTesting basic functionality...")
    
    try:
        from lru_cache import LRUCache, ResponseCache
        from cost_tracker import CostTracker
        
        # Test LRU Cache
        cache = LRUCache(max_size=5, default_ttl=60)
        cache.put("test", "value")
        result = cache.get("test")
        assert result == "value", "Cache get/put failed"
        print("✅ LRU Cache basic operations working")
        
        # Test ResponseCache
        response_cache = ResponseCache(max_size=5, default_ttl=60)
        response_cache.cache_response("prompt", "text", "response", 10, 20, 0.01)
        cached = response_cache.get_cached_response("prompt", "text")
        assert cached is not None, "ResponseCache failed"
        print("✅ ResponseCache basic operations working")
        
        # Test Cost Tracker
        cost_tracker = CostTracker()
        record = cost_tracker.track_request("/test", "127.0.0.1", "test prompt", "test response")
        assert record is not None, "Cost tracking failed"
        print("✅ Cost Tracker basic operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 W1D4S2 Style Tests - Quick Validation")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        return False
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed!")
        return False
    
    print("\n✅ All quick validation tests passed!")
    print("🎯 Ready to run the full test suite with: python tests.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
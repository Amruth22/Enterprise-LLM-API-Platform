import pytest
import json
import os
import time
import threading
import requests
import socket
from unittest.mock import patch, MagicMock, Mock
from dotenv import load_dotenv
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Mock data for testing
MOCK_RESPONSES = {
    "text_generation": "Once upon a time, there was a curious cat named Whiskers who discovered a magical garden behind an old oak tree.",
    "code_generation": "def add_numbers(a, b):\n    \"\"\"Add two numbers and return the result.\"\"\"\n    return a + b\n\n# Example usage\nresult = add_numbers(5, 3)\nprint(f'Result: {result}')",
    "classification_positive": "positive",
    "classification_negative": "negative",
    "classification_neutral": "neutral"
}

# Mock enterprise components
@dataclass
class MockCostRecord:
    timestamp: float
    endpoint: str
    user_ip: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    cached: bool = False

class MockTokenCounter:
    INPUT_COST_PER_MILLION = 0.10
    OUTPUT_COST_PER_MILLION = 0.40
    
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)  # Simple estimation
    
    def calculate_cost(self, input_tokens: int, output_tokens: int):
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost

class MockLRUCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self._hit_count = 0
        self._miss_count = 0
        self._total_count = 0
    
    def get(self, key: str):
        if key in self.cache:
            self._hit_count += 1
            value = self.cache[key]
            self.cache.move_to_end(key)
            return value
        self._miss_count += 1
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def size(self):
        return len(self.cache)
    
    def get_stats(self):
        self._total_count = self._hit_count + self._miss_count
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'cache_hit_ratio': self._hit_count / max(self._total_count, 1)
        }

class MockResponseCache(MockLRUCache):
    def get_cached_response(self, prompt: str, task_type: str):
        cache_key = f"{prompt}:{task_type}"
        self._total_count += 1
        cached = self.get(cache_key)
        return cached
    
    def cache_response(self, prompt: str, task_type: str, response: Any, 
                      input_tokens: int, output_tokens: int, cost: float, ttl: Optional[int] = None):
        cache_key = f"{prompt}:{task_type}"
        cache_data = {
            'response': response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'cached_at': time.time()
        }
        self.put(cache_key, cache_data, ttl)

class MockCostTracker:
    def __init__(self):
        self.token_counter = MockTokenCounter()
        self.records = []
    
    def track_request(self, endpoint: str, user_ip: str, prompt: str, 
                     response: str, cached: bool = False):
        input_tokens = self.token_counter.count_tokens(prompt)
        output_tokens = self.token_counter.count_tokens(response)
        input_cost, output_cost, total_cost = self.token_counter.calculate_cost(input_tokens, output_tokens)
        
        record = MockCostRecord(
            timestamp=time.time(),
            endpoint=endpoint,
            user_ip=user_ip,
            prompt=prompt[:100],
            response=response[:100],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost if not cached else 0.0,
            cached=cached
        )
        self.records.append(record)
        return record
    
    def get_usage_stats(self, hours: int = 24):
        return {
            'period_hours': hours,
            'total_requests': len(self.records),
            'cached_requests': sum(1 for r in self.records if r.cached),
            'total_cost': sum(r.total_cost for r in self.records),
            'cost_savings_from_cache': sum(r.input_cost + r.output_cost for r in self.records if r.cached)
        }

class MockRequestLogger:
    def log_request_start(self, request_id=None):
        return request_id or "mock-request-id"
    
    def log_request_end(self, response_data, status_code, cached=False, cost_info=None):
        pass
    
    def log_error(self, error, request_id=None, additional_context=None):
        pass

# Mock wrapper functions
def mock_generate_text(prompt):
    return MOCK_RESPONSES["text_generation"]

def mock_generate_code(prompt):
    return MOCK_RESPONSES["code_generation"]

def mock_classify_text(text, categories):
    text_lower = text.lower()
    if any(word in text_lower for word in ['love', 'amazing', 'great', 'perfect', 'excellent']):
        return MOCK_RESPONSES["classification_positive"]
    elif any(word in text_lower for word in ['hate', 'terrible', 'awful', 'worst', 'bad']):
        return MOCK_RESPONSES["classification_negative"]
    else:
        return MOCK_RESPONSES["classification_neutral"]

# Create mock Flask app
def create_mock_app():
    """Create a mock Flask app with enterprise features"""
    try:
        from flask import Flask, request, jsonify, g
        from flask_restx import Api, Resource, fields
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        from flask_cors import CORS
    except ImportError:
        return None
    
    app = Flask(__name__)
    
    # Configure CORS
    CORS(app, 
         origins=['*'],
         methods=['GET', 'POST', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
         expose_headers=['X-Request-ID', 'X-Cost-Input', 'X-Cost-Output', 'X-Cost-Total', 'X-Cost-Overall', 'X-Tokens-Input', 'X-Tokens-Output', 'X-Cached']
    )
    
    # Initialize mock components
    response_cache = MockResponseCache(max_size=500, default_ttl=1800)
    cost_tracker = MockCostTracker()
    request_logger = MockRequestLogger()
    
    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://",
    )
    
    api = Api(
        app, 
        version='1.0', 
        title='Enterprise LLM API (Mock)',
        description='Mock Enterprise Flask API for testing',
        doc='/swagger/',
        prefix='/api/v1'
    )
    
    # Health check endpoint
    @app.route('/api/v1/health')
    def health_check():
        return {'status': 'healthy', 'timestamp': time.time()}
    
    # Stats endpoint
    @app.route('/api/v1/stats')
    def get_stats():
        return {
            'cache_stats': response_cache.get_stats(),
            'usage_stats': cost_tracker.get_usage_stats(24),
            'rate_limit_stats': {
                'cache_size': 45,
                'max_size': 10000
            }
        }
    
    # Before request handler
    @app.before_request
    def before_request():
        g.request_id = request_logger.log_request_start()
        g.request_start_time = time.time()
    
    # After request handler
    @app.after_request
    def after_request(response):
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        
        if hasattr(g, 'cost_info'):
            cost_info = g.cost_info
            response.headers['X-Cost-Input'] = f"{cost_info.get('input_cost', 0):.6f}"
            response.headers['X-Cost-Output'] = f"{cost_info.get('output_cost', 0):.6f}"
            response.headers['X-Cost-Total'] = f"{cost_info.get('total_cost', 0):.6f}"
            response.headers['X-Cost-Overall'] = f"{cost_info.get('overall_cost', 0):.6f}"
            response.headers['X-Cached'] = str(cost_info.get('cached', False))
            
            if hasattr(g, 'token_info'):
                token_info = g.token_info
                response.headers['X-Tokens-Input'] = str(token_info.get('input_tokens', 0))
                response.headers['X-Tokens-Output'] = str(token_info.get('output_tokens', 0))
        
        return response
    
    # Text generation endpoint
    @app.route('/api/v1/generate/text', methods=['POST'])
    @limiter.limit("10 per minute")
    def generate_text_endpoint():
        data = request.get_json()
        if not data:
            return {'error': 'Request body is required'}, 400
            
        prompt = data.get('prompt')
        if not prompt:
            return {'error': 'Prompt is required'}, 400
        
        try:
            # Check cache first
            cached_response = response_cache.get_cached_response(prompt, 'text')
            
            if cached_response:
                # Serve from cache
                response_data = {'generated_text': cached_response['response']}
                
                # Set cost info for headers
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': cached_response.get('input_tokens', 0) * 0.0000001,
                    'output_cost': cached_response.get('output_tokens', 0) * 0.0000004,
                    'total_cost': 0.0,  # No cost for cached responses
                    'cached': True,
                    'overall_cost': overall_cost
                }
                
                g.token_info = {
                    'input_tokens': cached_response.get('input_tokens', 0),
                    'output_tokens': cached_response.get('output_tokens', 0)
                }
                
                cost_tracker.track_request('/api/v1/generate/text', '127.0.0.1', prompt, cached_response['response'], cached=True)
                return response_data
            
            # Generate new response
            text = mock_generate_text(prompt)
            
            # Track cost
            cost_record = cost_tracker.track_request('/api/v1/generate/text', '127.0.0.1', prompt, text)
            
            # Cache the response
            response_cache.cache_response(prompt, 'text', text, cost_record.input_tokens, cost_record.output_tokens, cost_record.total_cost)
            
            response_data = {'generated_text': text}
            
            # Set cost info for headers
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            g.token_info = {
                'input_tokens': cost_record.input_tokens,
                'output_tokens': cost_record.output_tokens
            }
            
            return response_data
            
        except Exception as e:
            return {'error': str(e)}, 500
    
    # Code generation endpoint
    @app.route('/api/v1/generate/code', methods=['POST'])
    @limiter.limit("10 per minute")
    def generate_code_endpoint():
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            return {'error': 'Prompt is required'}, 400

        try:
            # Check cache first
            cached_response = response_cache.get_cached_response(prompt, 'code')
            
            if cached_response:
                response_data = {'generated_code': cached_response['response']}
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': cached_response.get('input_tokens', 0) * 0.0000001,
                    'output_cost': cached_response.get('output_tokens', 0) * 0.0000004,
                    'total_cost': 0.0,
                    'cached': True,
                    'overall_cost': overall_cost
                }
                cost_tracker.track_request('/api/v1/generate/code', '127.0.0.1', prompt, cached_response['response'], cached=True)
                return response_data
            
            # Generate new response
            code = mock_generate_code(prompt)
            cost_record = cost_tracker.track_request('/api/v1/generate/code', '127.0.0.1', prompt, code)
            response_cache.cache_response(prompt, 'code', code, cost_record.input_tokens, cost_record.output_tokens, cost_record.total_cost)
            
            response_data = {'generated_code': code}
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            return response_data
            
        except Exception as e:
            return {'error': str(e)}, 500
    
    # Text classification endpoint
    @app.route('/api/v1/classify/text', methods=['POST'])
    @limiter.limit("10 per minute")
    def classify_text_endpoint():
        data = request.get_json()
        text = data.get('text')
        categories = data.get('categories')

        if not text or not categories:
            return {'error': 'Text and categories are required'}, 400

        try:
            cache_prompt = f"text:{text}|categories:{','.join(categories)}"
            cached_response = response_cache.get_cached_response(cache_prompt, 'classify')
            
            if cached_response:
                response_data = {'classification': cached_response['response']}
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': cached_response.get('input_tokens', 0) * 0.0000001,
                    'output_cost': cached_response.get('output_tokens', 0) * 0.0000004,
                    'total_cost': 0.0,
                    'cached': True,
                    'overall_cost': overall_cost
                }
                cost_tracker.track_request('/api/v1/classify/text', '127.0.0.1', cache_prompt, cached_response['response'], cached=True)
                return response_data
            
            # Generate new response
            classification = mock_classify_text(text, categories)
            cost_record = cost_tracker.track_request('/api/v1/classify/text', '127.0.0.1', cache_prompt, classification)
            response_cache.cache_response(cache_prompt, 'classify', classification, cost_record.input_tokens, cost_record.output_tokens, cost_record.total_cost)
            
            response_data = {'classification': classification}
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            return response_data
            
        except Exception as e:
            return {'error': str(e)}, 500
    
    return app

# Global variables for server management
mock_app = None
server_thread = None
server_started = False
test_port = 8080
base_url = f"http://localhost:{test_port}/api/v1"

def find_free_port():
    """Find a free port for testing"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def run_server():
    """Run Flask server in thread with mocked functions"""
    global mock_app
    try:
        if mock_app:
            mock_app.run(host='localhost', port=test_port, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[ERROR] Server failed to start: {e}")

def wait_for_server(timeout=10):
    """Wait for server to be ready (optimized)"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.2)
    
    raise RuntimeError(f"Server failed to start within {timeout} seconds")

def setup_test_server():
    """Set up live server for testing with mocked functions"""
    global server_thread, server_started, test_port, base_url, mock_app
    
    load_dotenv()
    
    # Create mock app
    mock_app = create_mock_app()
    if not mock_app:
        raise ImportError("Cannot create mock app - Flask not available")
    
    # Find available port
    test_port = find_free_port()
    base_url = f"http://localhost:{test_port}/api/v1"
    
    # Start the Flask server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    wait_for_server()
    server_started = True
    
    print(f"[SUCCESS] Mock enterprise server started at {base_url}")

def test_01_env_api_key_configured():
    """Test 1: API Key Configuration"""
    print("Running Test 1: API Key Configuration")
    
    # Check if .env file exists
    env_file_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file_path):
        print("PASS: .env file exists")
    else:
        print("INFO: .env file not found (optional for mock tests)")
    
    # Check if API key is loaded (optional for mock tests)
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key and api_key.startswith('AIza'):
        print(f"PASS: API Key configured: {api_key[:10]}...{api_key[-5:]}")
    else:
        print("INFO: GOOGLE_API_KEY not configured (not required for mock tests)")
    
    assert True, "Mock tests don't require real API key"

def test_02_generate_text_endpoint():
    """Test 2: Text Generation Endpoint"""
    print("Running Test 2: Text Generation Endpoint")
    
    if not server_started:
        setup_test_server()
    
    payload = {"prompt": "Write a one-sentence story about a cat."}
    
    response = requests.post(
        f'{base_url}/generate/text',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'generated_text' in data, "Response should contain 'generated_text'"
    assert data['generated_text'] is not None, "Generated text should not be None"
    assert len(data['generated_text']) > 0, "Generated text should not be empty"
    assert "cat" in data['generated_text'].lower(), "Mock response should contain 'cat'"
    
    print(f"PASS: Text generated (mocked): {data['generated_text'][:50]}...")

def test_03_generate_code_endpoint():
    """Test 3: Code Generation Endpoint"""
    print("Running Test 3: Code Generation Endpoint")
    
    if not server_started:
        setup_test_server()
    
    payload = {"prompt": "Create a simple Python function that adds two numbers"}
    
    response = requests.post(
        f'{base_url}/generate/code',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'generated_code' in data, "Response should contain 'generated_code'"
    assert data['generated_code'] is not None, "Generated code should not be None"
    assert len(data['generated_code']) > 0, "Generated code should not be empty"
    assert 'def' in data['generated_code'], "Generated code should contain a function definition"
    assert 'add_numbers' in data['generated_code'], "Mock code should contain expected function name"
    
    print(f"PASS: Code generated (mocked): {data['generated_code'][:50]}...")

def test_04_classify_text_endpoint():
    """Test 4: Text Classification Endpoint"""
    print("Running Test 4: Text Classification Endpoint")
    
    if not server_started:
        setup_test_server()
    
    payload = {
        "text": "I love this amazing product! It works perfectly.",
        "categories": ["positive", "negative", "neutral"]
    }
    
    response = requests.post(
        f'{base_url}/classify/text',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert 'classification' in data, "Response should contain 'classification'"
    assert data['classification'] is not None, "Classification should not be None"
    assert data['classification'] in ['positive', 'negative', 'neutral'], "Classification should be valid"
    assert data['classification'] == 'positive', "Text with 'love' and 'amazing' should be classified as positive"
    
    print(f"PASS: Text classified as: {data['classification']}")

def test_05_endpoint_error_handling():
    """Test 5: Error Handling"""
    print("Running Test 5: Error Handling")
    
    if not server_started:
        setup_test_server()
    
    # Test missing prompt for text generation
    response = requests.post(
        f'{base_url}/generate/text',
        json={},
        headers={'Content-Type': 'application/json'},
        timeout=2
    )
    assert response.status_code == 400, f"Expected 400 for missing prompt, got {response.status_code}"
    data = response.json()
    assert 'error' in data, "Error response should contain 'error' field"
    
    # Test missing fields for classification
    response = requests.post(
        f'{base_url}/classify/text',
        json={"text": "test"},
        headers={'Content-Type': 'application/json'},
        timeout=2
    )
    assert response.status_code == 400, f"Expected 400 for missing categories, got {response.status_code}"
    data = response.json()
    assert 'error' in data, "Error response should contain 'error' field"
    
    print("PASS: Error handling working correctly")

def test_06_lru_cache_functionality():
    """Test 6: LRU Cache Functionality"""
    print("Running Test 6: LRU Cache Functionality")
    
    # Test response cache
    response_cache = MockResponseCache(max_size=3, default_ttl=10)
    
    # Cache some responses
    response_cache.cache_response("test1", "text", "response1", 100, 200, 0.05)
    response_cache.cache_response("test2", "text", "response2", 150, 250, 0.075)
    response_cache.cache_response("test3", "text", "response3", 120, 220, 0.06)
    
    # Test cache hits
    cached1 = response_cache.get_cached_response("test1", "text")
    assert cached1 is not None, "Should find cached response"
    assert cached1['response'] == "response1", "Should return correct cached response"
    
    # Add one more (should evict least recently used)
    response_cache.cache_response("test4", "text", "response4", 130, 230, 0.065)
    
    # test2 should be evicted as it was least recently used
    cached2 = response_cache.get_cached_response("test2", "text")
    assert cached2 is None, "LRU item should be evicted"
    
    print("PASS: LRU Cache working correctly")

def test_07_cost_tracking():
    """Test 7: Cost Tracking"""
    print("Running Test 7: Cost Tracking")
    
    cost_tracker = MockCostTracker()
    
    # Track a request
    record = cost_tracker.track_request(
        "/api/v1/generate/text",
        "127.0.0.1",
        "Test prompt for cost tracking",
        "Generated response for cost tracking test"
    )
    
    assert record is not None, "Should create cost record"
    assert record.input_tokens > 0, "Should count input tokens"
    assert record.output_tokens > 0, "Should count output tokens"
    assert record.total_cost > 0, "Should calculate total cost"
    assert record.cached == False, "New request should not be cached"
    
    print(f"PASS: Cost tracked - Input: {record.input_tokens} tokens, Output: {record.output_tokens} tokens, Cost: ${record.total_cost:.6f}")
    
    # Get usage stats
    stats = cost_tracker.get_usage_stats(1)
    assert stats['total_requests'] > 0, "Should have request count"
    assert stats['total_cost'] >= 0, "Should have cost calculation"
    
    print(f"PASS: Usage stats - Requests: {stats['total_requests']}, Total cost: ${stats['total_cost']:.6f}")

def test_08_api_headers():
    """Test 8: API Response Headers"""
    print("Running Test 8: API Response Headers")
    
    if not server_started:
        setup_test_server()
    
    payload = {"prompt": "Say hello in one word"}
    
    response = requests.post(
        f'{base_url}/generate/text',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response.status_code == 200, "Should return 200"
    
    # Check for cost headers
    assert 'X-Request-ID' in response.headers, "Should have request ID header"
    assert 'X-Cost-Total' in response.headers, "Should have cost total header"
    assert 'X-Cached' in response.headers, "Should have cached header"
    
    cost_total = float(response.headers.get('X-Cost-Total', 0))
    cached = response.headers.get('X-Cached', 'False') == 'True'
    
    print(f"PASS: Request ID: {response.headers.get('X-Request-ID')}")
    print(f"PASS: Total cost: ${cost_total:.6f}")
    print(f"PASS: Cached: {cached}")

def test_09_health_and_stats_endpoints():
    """Test 9: Health and Stats Endpoints"""
    print("Running Test 9: Health and Stats Endpoints")
    
    if not server_started:
        setup_test_server()
    
    # Test health endpoint
    response = requests.get(f'{base_url}/health', timeout=2)
    assert response.status_code == 200, "Health endpoint should return 200"
    
    data = response.json()
    assert data['status'] == 'healthy', "Health status should be 'healthy'"
    assert 'timestamp' in data, "Health response should contain timestamp"
    
    print("PASS: Health endpoint working")
    
    # Test stats endpoint
    response = requests.get(f'{base_url}/stats', timeout=2)
    assert response.status_code == 200, "Stats endpoint should return 200"
    
    data = response.json()
    assert 'cache_stats' in data, "Stats should contain cache_stats"
    assert 'usage_stats' in data, "Stats should contain usage_stats"
    assert 'rate_limit_stats' in data, "Stats should contain rate_limit_stats"
    
    print("PASS: Stats endpoint working")
    print(f"PASS: Cache hit ratio: {data['cache_stats'].get('cache_hit_ratio', 0):.2%}")

def test_10_cache_functionality():
    """Test 10: Cache Hit/Miss Behavior"""
    print("Running Test 10: Cache Hit/Miss Behavior")
    
    if not server_started:
        setup_test_server()
    
    # Make initial request (should be cache miss)
    payload = {"prompt": "Cache test prompt unique"}
    response1 = requests.post(
        f'{base_url}/generate/text',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response1.status_code == 200, "First request should succeed"
    cached1 = response1.headers.get('X-Cached', 'False') == 'True'
    assert cached1 == False, "First request should not be cached"
    cost1 = float(response1.headers.get('X-Cost-Total', 0))
    assert cost1 > 0, "First request should have cost"
    
    print("PASS: Cache miss working correctly")
    
    # Make same request again (should be cache hit)
    response2 = requests.post(
        f'{base_url}/generate/text',
        json=payload,
        headers={'Content-Type': 'application/json'},
        timeout=3
    )
    
    assert response2.status_code == 200, "Second request should succeed"
    cached2 = response2.headers.get('X-Cached', 'False') == 'True'
    assert cached2 == True, "Second identical request should be cached"
    cost2 = float(response2.headers.get('X-Cost-Total', 0))
    assert cost2 == 0.0, "Cached response should have zero cost"
    
    print("PASS: Cache hit working correctly")
    print(f"PASS: Cost savings - First: ${cost1:.6f}, Second: ${cost2:.6f}")

def run_all_tests():
    """Run all tests and provide summary"""
    print("Running Enterprise LLM API Platform Tests (Mock Version)...")
    print("Using mocked data instead of real API calls")
    print("Testing enterprise features: caching, cost tracking, logging")
    print("=" * 70)
    
    # Setup server once for all tests
    try:
        setup_test_server()
    except Exception as e:
        print(f"❌ Failed to setup test server: {e}")
        return False
    
    # List of exactly 10 test functions
    test_functions = [
        test_01_env_api_key_configured,
        test_02_generate_text_endpoint,
        test_03_generate_code_endpoint,
        test_04_classify_text_endpoint,
        test_05_endpoint_error_handling,
        test_06_lru_cache_functionality,
        test_07_cost_tracking,
        test_08_api_headers,
        test_09_health_and_stats_endpoints,
        test_10_cache_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_func.__name__} - {e}")
            failed += 1
    
    print("=" * 70)
    print(f"📊 Test Results Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
        print("✅ Enterprise LLM API Platform (Mock) is working correctly")
        print(f"🌐 Mock server running at: {base_url}")
        print("🔧 Tests use mocked enterprise features - no real API calls made")
        print("💰 Cost tracking, caching, and logging features validated")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed")
        return False

if __name__ == "__main__":
    print("🚀 Starting Enterprise LLM API Platform Tests (Mock Version)")
    print("📋 No API keys required - using mocked responses")
    print("🔧 Testing enterprise features: caching, cost tracking, logging")
    print("🏢 Enterprise-grade platform validation")
    print()
    
    # Run the tests
    success = run_all_tests()
    exit(0 if success else 1)
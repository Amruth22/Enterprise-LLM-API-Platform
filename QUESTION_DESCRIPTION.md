# Enterprise LLM API Platform - Expert-Level Coding Challenge

## 🎯 Problem Statement

Build a **Production-Ready Enterprise LLM API Platform** that provides multi-task AI capabilities (text generation, code generation, text classification) with comprehensive enterprise features including intelligent caching, cost tracking, enhanced security, performance monitoring, and production-grade logging. Your task is to create a scalable, secure, and cost-optimized API platform that can handle enterprise workloads while providing detailed analytics and monitoring capabilities.

## 📋 Requirements Overview

### Core System Components
You need to implement a complete enterprise API platform with:

1. **Multi-Task LLM API** with text generation, code generation, and text classification
2. **Intelligent Caching System** with LRU cache and TTL support for cost optimization
3. **Comprehensive Cost Tracking** with real-time token counting and usage analytics
4. **Enhanced Security & Logging** with multi-level audit trails and threat detection
5. **Production-Grade Features** including CORS, rate limiting, health checks, and monitoring
6. **Enterprise Testing Suite** with comprehensive validation and performance testing

## 🏗️ System Architecture

```
Client Request → [CORS/Security] → [Rate Limiting] → [Cache Check] → [LLM API] → [Cost Tracking] → [Logging] → Response
                        ↓                ↓              ↓              ↓              ↓              ↓
                [Security Audit] → [LRU Cache] → [Cache Miss] → [Gemini API] → [Token Count] → [Analytics]
                        ↓                ↓              ↓              ↓              ↓              ↓
                [Threat Detection] → [TTL Management] → [Retry Logic] → [Response] → [Cost Headers] → [Monitoring]
```

## 📚 Detailed Implementation Requirements

### 1. Core Flask API Application (`app.py`)

**Multi-Task API Endpoints:**

```python
# Text Generation Endpoint
@ns_generate.route('/text')
class TextGeneration(Resource):
    @ns_generate.expect(text_generation_model)
    @limiter.limit("10 per minute")
    def post(self):
        # Check cache first
        # Generate text if cache miss
        # Track costs and tokens
        # Update cache
        # Return response with cost headers

# Code Generation Endpoint  
@ns_generate.route('/code')
class CodeGeneration(Resource):
    @ns_generate.expect(code_generation_model)
    @limiter.limit("10 per minute")
    def post(self):
        # Optimized for code generation (temperature=0.2)
        # Cache code responses
        # Track development-specific metrics

# Text Classification Endpoint
@ns_classify.route('/text')
class TextClassification(Resource):
    @ns_classify.expect(text_classification_model)
    @limiter.limit("10 per minute")
    def post(self):
        # Deterministic classification (temperature=0.0)
        # Category-based caching
        # Classification accuracy tracking
```

**Enterprise Features Integration:**
- **CORS Configuration**: Cross-origin support with custom headers
- **Rate Limiting**: Flask-Limiter with LRU cache backend
- **Request/Response Middleware**: Cost tracking, logging, security checks
- **Health & Stats Endpoints**: Monitoring and analytics endpoints
- **Swagger Documentation**: Auto-generated API documentation

### 2. Intelligent Caching System (`lru_cache.py`)

**Thread-Safe LRU Cache:**

```python
class LRUCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        # Thread-safe LRU implementation with TTL
        # OrderedDict for O(1) operations
        # Automatic expiration handling
        # Memory-efficient storage
    
    def get(self, key: str) -> Optional[Any]:
        # Retrieve with LRU promotion
        # Automatic expiration checking
        # Thread-safe operations
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        # Insert with TTL management
        # LRU eviction when at capacity
        # Thread-safe updates

class ResponseCache(LRUCache):
    def get_cached_response(self, prompt: str, task_type: str) -> Optional[Dict]:
        # Task-specific caching (text, code, classify)
        # Prompt-based key generation
        # Cost information preservation
    
    def cache_response(self, prompt: str, task_type: str, response: Any, 
                      input_tokens: int, output_tokens: int, cost: float):
        # Store response with metadata
        # Token count preservation
        # Cost tracking integration

class RateLimitCache(LRUCache):
    def increment_count(self, key: str, window_size: int = 60) -> int:
        # Sliding window rate limiting
        # IP-based tracking
        # Configurable time windows
```

### 3. Comprehensive Cost Tracking (`cost_tracker.py`)

**Token Counting & Cost Analysis:**

```python
class TokenCounter:
    # Gemini 2.0 Flash pricing
    INPUT_COST_PER_MILLION = 0.10
    OUTPUT_COST_PER_MILLION = 0.40
    
    def __init__(self):
        # tiktoken integration for accurate counting
        # Fallback estimation methods
    
    def count_tokens(self, text: str) -> int:
        # Accurate token counting using tiktoken
        # Fallback to character-based estimation
        # Handle edge cases and encoding issues
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> tuple:
        # Real-time cost calculation
        # Separate input/output pricing
        # Return detailed cost breakdown

class CostTracker:
    def __init__(self, log_file: str = "logs/cost_tracking.log"):
        # Initialize cost tracking system
        # Setup structured logging
        # Thread-safe operations
    
    def track_request(self, endpoint: str, user_ip: str, prompt: str, 
                     response: str, cached: bool = False) -> CostRecord:
        # Track individual request costs
        # Calculate token counts and costs
        # Store detailed records
        # Handle cached vs. fresh requests
    
    def get_usage_stats(self, hours: int = 24) -> Dict:
        # Generate usage analytics
        # Calculate cost savings from caching
        # Per-endpoint statistics
        # Time-based analysis
    
    def export_records(self, output_file: str, hours: Optional[int] = None) -> int:
        # Export cost data for billing
        # JSON format for integration
        # Configurable time ranges
```

### 4. Enhanced Security & Logging (`enhanced_logging.py`)

**Multi-Level Logging System:**

```python
class RequestLogger:
    def __init__(self, log_dir: str = "logs"):
        # Setup multiple specialized loggers:
        # - api_requests.log: Request/response audit trail
        # - security_audit.log: Security events and threats
        # - api_errors.log: Error tracking and debugging
        # - performance.log: Performance metrics and optimization
    
    def log_request_start(self, request_id: str = None) -> str:
        # Generate unique request ID
        # Log request details (IP, user agent, endpoint)
        # Security threat detection
        # Performance timing start
    
    def log_request_end(self, response_data: Dict, status_code: int, 
                       cached: bool = False, cost_info: Optional[Dict] = None):
        # Complete request audit trail
        # Performance metrics calculation
        # Cost information logging
        # Error categorization
    
    def log_security_event(self, event_type: str, severity: str, details: Dict):
        # Security threat detection and logging
        # Suspicious pattern identification
        # Audit trail for compliance
        # Severity-based alerting
    
    def _check_security_issues(self, request_data: Dict):
        # SQL injection detection
        # XSS pattern identification
        # Suspicious user agent detection
        # Path traversal attempts
```

### 5. Gemini API Integration (`gemini_wrapper.py`)

**Resilient LLM Integration:**

```python
@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_text(prompt):
    # Gemini 2.0 Flash integration
    # Temperature optimization (0.7 for creativity)
    # Exponential backoff retry logic
    # Rate limit detection and handling

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_code(prompt):
    # Code-optimized settings (temperature=0.2)
    # Programming-specific prompting
    # Consistent code generation
    # Error handling for code tasks

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def classify_text(text, categories):
    # Deterministic classification (temperature=0.0)
    # Category-constrained responses
    # Classification accuracy optimization
    # Multi-category support
```

## 🧪 Test Cases & Validation

Your implementation will be tested against these comprehensive scenarios:

### Test Case 1: Environment & Configuration (2 Tests)
```python
def test_01_env_api_key_configured(self):
    """Test API key configuration and format validation"""
    # Check .env file exists
    # Validate GOOGLE_API_KEY format (starts with 'AIza')
    # Ensure key is not empty or placeholder
    
def test_02_flask_app_initialization(self):
    """Test Flask app and extensions initialization"""
    # Verify Flask-RESTX integration
    # Check CORS configuration
    # Validate rate limiter setup
    # Test Swagger UI accessibility
```

### Test Case 2: Core API Functionality (3 Tests)
```python
def test_03_generate_text_endpoint(self):
    """Test text generation endpoint functionality"""
    payload = {"prompt": "Write a one-sentence story about a cat."}
    response = self.client.post('/api/v1/generate/text', json=payload)
    assert response.status_code == 200
    assert 'generated_text' in response.json
    assert len(response.json['generated_text']) > 0

def test_04_generate_code_endpoint(self):
    """Test code generation endpoint functionality"""
    payload = {"prompt": "Create a simple Python function that adds two numbers"}
    response = self.client.post('/api/v1/generate/code', json=payload)
    assert response.status_code == 200
    assert 'generated_code' in response.json
    assert 'def' in response.json['generated_code']

def test_05_classify_text_endpoint(self):
    """Test text classification endpoint functionality"""
    payload = {
        "text": "I love this amazing product!",
        "categories": ["positive", "negative", "neutral"]
    }
    response = self.client.post('/api/v1/classify/text', json=payload)
    assert response.status_code == 200
    assert response.json['classification'] in ['positive', 'negative', 'neutral']
```

### Test Case 3: Enterprise Features (4 Tests)
```python
def test_06_lru_cache_functionality(self):
    """Test LRU cache implementation and TTL"""
    # Test ResponseCache with different task types
    # Verify LRU eviction when at capacity
    # Test TTL expiration handling
    # Validate thread safety

def test_07_cost_tracking_accuracy(self):
    """Test cost tracking and token counting"""
    # Verify token counting accuracy
    # Test cost calculation formulas
    # Validate cost record creation
    # Check usage statistics generation

def test_08_api_response_headers(self):
    """Test cost and tracking headers in responses"""
    # Verify X-Request-ID presence
    # Check cost headers (X-Cost-Total, X-Cost-Overall)
    # Validate token count headers
    # Test cached response indicators

def test_09_health_and_stats_endpoints(self):
    """Test monitoring and analytics endpoints"""
    # Health check endpoint functionality
    # Stats endpoint data structure
    # Cache statistics accuracy
    # Usage analytics validation
```

### Test Case 4: Security & Performance (3 Tests)
```python
def test_10_error_handling_robustness(self):
    """Test comprehensive error handling"""
    # Missing required fields (400 errors)
    # Invalid input validation
    # API failure scenarios
    # Rate limit exceeded responses

def test_11_cache_hit_miss_behavior(self):
    """Test caching behavior and cost optimization"""
    # First request (cache miss)
    # Second identical request (cache hit)
    # Cost reduction verification
    # Cache statistics updates

def test_12_log_files_creation(self):
    """Test logging system and file creation"""
    # Verify logs directory creation
    # Check log file presence and content
    # Validate log rotation settings
    # Test structured logging format
```

### Test Case 5: Integration & Performance (3 Tests)
```python
def test_13_direct_wrapper_functions(self):
    """Test Gemini wrapper functions directly"""
    # Direct function calls without Flask
    # Retry logic validation
    # Rate limit handling
    # Response format consistency

def test_14_overall_cost_tracking(self):
    """Test cumulative cost tracking across requests"""
    # Multiple requests cost accumulation
    # Overall cost header accuracy
    # Cost savings calculation
    # Usage pattern analysis

def test_15_concurrent_request_handling(self):
    """Test thread safety and concurrent operations"""
    # Multiple simultaneous requests
    # Cache thread safety
    # Cost tracking accuracy under load
    # Rate limiting effectiveness
```

## 📊 Evaluation Criteria

Your solution will be evaluated on:

1. **API Functionality** (25%): Multi-task endpoints, Swagger integration, error handling
2. **Enterprise Features** (25%): Caching, cost tracking, security, monitoring
3. **Performance & Scalability** (20%): Thread safety, optimization, resource management
4. **Code Quality & Architecture** (15%): Clean design, documentation, best practices
5. **Testing & Validation** (15%): Comprehensive test coverage, edge case handling

## 🔧 Technical Requirements

### Dependencies
```txt
# Core Framework
Flask>=2.3.0
Flask-RESTX>=1.1.0
Flask-Limiter>=3.0.0
Flask-CORS>=4.0.0
python-dotenv>=1.0.0

# LLM Integration
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-community>=0.0.20

# Enterprise Features
tenacity>=8.2.0
tiktoken>=0.5.0
waitress>=2.1.0
```

### Environment Configuration
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### File Structure
```
Enterprise-LLM-API-Platform/
├── app.py                    # Main Flask application with enterprise features
├── gemini_wrapper.py         # LLM API integration with retry logic
├── lru_cache.py             # Thread-safe LRU cache implementation
├── cost_tracker.py          # Cost tracking and analytics system
├── enhanced_logging.py      # Multi-level logging and security
├── test_unit.py            # Comprehensive test suite (15 tests)
├── requirements.txt         # All dependencies
├── .env                     # Environment configuration
├── README.md               # Project documentation
├── ENTERPRISE_FEATURES.md  # Enterprise features documentation
└── logs/                   # Auto-created log directory
    ├── api_requests.log
    ├── security_audit.log
    ├── api_errors.log
    ├── performance.log
    └── cost_tracking.log
```

### Performance Requirements
- **Response Time**: < 2 seconds for cached responses, < 10 seconds for fresh requests
- **Throughput**: Support 50+ concurrent requests with proper rate limiting
- **Memory Efficiency**: LRU cache with configurable size limits
- **Cost Optimization**: 30-50% cost reduction through intelligent caching
- **Uptime**: Production-ready with comprehensive error handling

## 🚀 Advanced Features (Bonus Points)

Implement these for extra credit:

1. **Advanced Analytics**: Real-time dashboards, usage trends, cost forecasting
2. **Multi-Model Support**: Integration with multiple LLM providers (OpenAI, Anthropic)
3. **API Versioning**: Backward compatibility and version management
4. **Database Integration**: Persistent storage for analytics and user management
5. **Container Deployment**: Docker containerization with orchestration
6. **Load Balancing**: Multi-instance deployment with session affinity
7. **Real-time Monitoring**: Prometheus metrics, Grafana dashboards
8. **Advanced Security**: JWT authentication, API key management, encryption

## 📝 Implementation Guidelines

### Enterprise API Pattern
```python
@app.before_request
def before_request():
    # Generate unique request ID
    # Start performance timing
    # Security threat detection
    # Rate limiting validation
    
@app.after_request
def after_request(response):
    # Add cost tracking headers
    # Complete audit logging
    # Performance metrics calculation
    # Security event logging
```

### Intelligent Caching Strategy
```python
def handle_request_with_cache(prompt, task_type, generator_func):
    # Check cache first
    cached_response = response_cache.get_cached_response(prompt, task_type)
    if cached_response:
        # Serve from cache with zero cost
        return cached_response, True
    
    # Generate fresh response
    response = generator_func(prompt)
    
    # Track costs and cache
    cost_record = cost_tracker.track_request(...)
    response_cache.cache_response(prompt, task_type, response, ...)
    
    return response, False
```

### Cost Optimization Logic
```python
def calculate_cost_savings():
    # Calculate potential costs without caching
    total_requests = len(cost_tracker.records)
    cached_requests = len([r for r in cost_tracker.records if r.cached])
    
    # Estimate savings
    cache_hit_rate = cached_requests / total_requests
    estimated_savings = cache_hit_rate * total_api_costs
    
    return {
        'cache_hit_rate': cache_hit_rate,
        'estimated_savings': estimated_savings,
        'cost_reduction_percentage': cache_hit_rate * 100
    }
```

## 🎯 Success Criteria

Your implementation is successful when:

- ✅ All 15 test cases pass with comprehensive validation
- ✅ Supports all 3 LLM tasks (text, code, classification) with proper optimization
- ✅ Implements thread-safe LRU caching with TTL support
- ✅ Provides accurate cost tracking with real-time analytics
- ✅ Includes comprehensive security logging and threat detection
- ✅ Generates detailed API documentation with Swagger UI
- ✅ Demonstrates 30%+ cost reduction through intelligent caching
- ✅ Handles enterprise workloads with proper rate limiting and monitoring

## 📋 Submission Requirements

### Required Files
1. **Core Application** (2 files):
   - `app.py`: Main Flask application with all enterprise features
   - `gemini_wrapper.py`: LLM integration with retry logic

2. **Enterprise Components** (3 files):
   - `lru_cache.py`: Thread-safe LRU cache implementation
   - `cost_tracker.py`: Cost tracking and analytics system
   - `enhanced_logging.py`: Multi-level logging and security

3. **Testing & Documentation** (4 files):
   - `test_unit.py`: Comprehensive test suite (15 tests)
   - `requirements.txt`: All required dependencies
   - `README.md`: Project documentation
   - `ENTERPRISE_FEATURES.md`: Enterprise features documentation

4. **Configuration** (1 file):
   - `.env`: Environment template (without actual API key)

### Code Quality Standards
- **Enterprise Architecture**: Modular design with clean separation of concerns
- **Thread Safety**: All shared resources properly synchronized
- **Error Handling**: Production-grade exception management
- **Performance Optimization**: Efficient algorithms and resource usage
- **Security**: Comprehensive threat detection and audit logging
- **Documentation**: Clear API documentation and inline comments

## 🔍 Sample Usage Examples

### Basic API Usage
```bash
# Text Generation
curl -X POST http://localhost:5000/api/v1/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a short story about AI"}'

# Code Generation
curl -X POST http://localhost:5000/api/v1/generate/code \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a Python function for binary search"}'

# Text Classification
curl -X POST http://localhost:5000/api/v1/classify/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!", "categories": ["positive", "negative", "neutral"]}'
```

### Enterprise Monitoring
```bash
# Health Check
curl http://localhost:5000/api/v1/health

# Usage Statistics
curl http://localhost:5000/api/v1/stats

# Swagger Documentation
open http://localhost:5000/swagger/
```

### Cost Analysis Example
```bash
# Response headers show detailed cost information:
# X-Cost-Input: 0.000015
# X-Cost-Output: 0.000060  
# X-Cost-Total: 0.000075
# X-Cost-Overall: 2.450000
# X-Cached: false
```

## ⚠️ Important Notes

- **API Key Security**: Never commit real API keys to version control
- **Rate Limiting**: Implement proper rate limiting to avoid API quota exhaustion
- **Thread Safety**: Ensure all shared resources are properly synchronized
- **Memory Management**: Implement proper cache eviction and resource cleanup
- **Error Handling**: System should never crash on API failures or invalid input
- **Performance**: Optimize for both speed and cost efficiency
- **Security**: Implement comprehensive threat detection and audit logging

Build a production-ready enterprise LLM API platform that demonstrates expert-level skills in API development, performance optimization, cost management, and enterprise security! 🚀
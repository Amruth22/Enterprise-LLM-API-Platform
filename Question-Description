# 🚀 Multi-Task LLM API - Expert-Level Full-Stack Development Challenge

## 🎯 Problem Statement

Build a **Production-Ready Multi-Task LLM API** that provides comprehensive text generation, code generation, and text classification capabilities using Google's Gemini 2.0 Flash model. Your task is to create a complete enterprise-grade Flask API system with intelligent rate limiting, exponential backoff retry logic, interactive Swagger documentation, comprehensive error handling, and production-ready deployment capabilities.

## 📋 Requirements Overview

### Core System Components
You need to implement a complete production API platform with:

1. **Multi-Task AI Capabilities** with optimized model configurations for different use cases
2. **RESTful API Design** with proper HTTP methods and status codes
3. **Interactive Swagger Documentation** with comprehensive API specification
4. **Intelligent Rate Limiting** with per-endpoint controls and memory storage
5. **Exponential Backoff Retry Logic** with automatic Gemini API rate limit handling
6. **Comprehensive Error Handling** with graceful degradation and proper responses
7. **Production Deployment** with WSGI server integration and CORS support
8. **Complete Test Suite** with live server testing and validation

## 🏗️ System Architecture

```
Core API Framework → [Flask + Flask-RESTX] → [Swagger UI] → [Documentation]
                           ↓                    ↓              ↓
LLM Integration → [Gemini Wrapper] → [Rate Limiting] → [Error Handling]
                           ↓              ↓                ↓
Production System → [CORS Support] → [Health Checks] → [Live Testing]
                           ↓              ↓                ↓
Deployment Ready → [Waitress WSGI] → [Environment Config] → [Monitoring]
```

## 📚 Detailed Implementation Requirements

### 1. Core Flask API Framework (`app.py`)

**Enhanced Flask Application with Flask-RESTX:**

```python
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS

app = Flask(__name__)

# Configure comprehensive CORS
CORS(app, 
     origins=['*'],  # Configure specific origins in production
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With']
)

# Initialize sophisticated rate limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Configure interactive API documentation
api = Api(
    app, 
    version='1.0', 
    title='Multi-Task LLM API',
    description='Production Flask API with text generation, code generation, and classification',
    doc='/swagger/',
    prefix='/api/v1'
)

# Implement 4 core endpoints:
# 1. Health Check: GET /api/v1/health
# 2. Text Generation: POST /api/v1/generate/text
# 3. Code Generation: POST /api/v1/generate/code  
# 4. Text Classification: POST /api/v1/classify/text
```

**Advanced Rate Limiting Configuration:**

```python
@limiter.limit("10 per minute")  # Per-endpoint limits
def post(self):
    # Individual endpoint rate limiting
    # Prevents API abuse and ensures fair usage
    # Memory-based storage for development
```

### 2. Intelligent Gemini Wrapper (`gemini_wrapper.py`)

**Advanced LLM Integration with Retry Logic:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

def is_rate_limit_error(exception):
    """Intelligent rate limit detection"""
    # Detect multiple rate limit indicators:
    # - HTTP 429 status codes
    # - "rate limit exceeded" messages  
    # - "quota exceeded" errors
    # - "resource exhausted" responses
    
@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_text(prompt):
    """Text generation with optimized settings"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.7,  # Creative and varied outputs
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Implement comprehensive error handling and logging

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def generate_code(prompt):
    """Code generation with consistency settings"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,  # Consistent and reliable code
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Enhanced prompt engineering for code generation

@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60), 
    reraise=True
)
def classify_text(text, categories):
    """Text classification with deterministic settings"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,  # Deterministic classification
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Structured classification prompt engineering
```

### 3. Interactive API Documentation

**Comprehensive Swagger Models:**

```python
# API Models for complete documentation
text_generation_model = api.model('TextGeneration', {
    'prompt': fields.String(required=True, description='Text prompt for generation', 
                           example='Write a short story about a robot')
})

code_generation_model = api.model('CodeGeneration', {
    'prompt': fields.String(required=True, description='Prompt for code generation',
                           example='Create a Python function to calculate fibonacci numbers')
})

text_classification_model = api.model('TextClassification', {
    'text': fields.String(required=True, description='Text to classify',
                         example='This movie was amazing!'),
    'categories': fields.List(fields.String, required=True, description='Classification categories',
                             example=['positive', 'negative', 'neutral'])
})

# Response models with comprehensive documentation
text_response_model = api.model('TextResponse', {
    'generated_text': fields.String(description='Generated text response')
})

error_model = api.model('Error', {
    'error': fields.String(description='Detailed error message')
})
```

### 4. Production-Ready Testing Suite (`unit_test.py`)

**Comprehensive Live Server Testing:**

```python
class TestMultiTaskLLMAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup live Flask server for testing"""
        # Dynamic port finding for Windows compatibility
        # Thread-based server execution
        # Automatic server readiness detection
        
    def test_01_env_api_key_configured(self):
        """Validate API key configuration and format"""
        # Test .env file existence
        # Validate API key format (starts with 'AIza')
        # Ensure proper environment loading
        
    def test_02_flask_app_initialization(self):
        """Test Flask app and extensions initialization"""
        # Validate health endpoint functionality
        # Test CORS configuration
        # Verify rate limiter setup
        
    def test_03_generate_text_endpoint(self):
        """Test text generation with live API calls"""
        # Real Gemini API integration testing
        # Response format validation
        # Content quality verification
        
    def test_04_generate_code_endpoint(self):
        """Test code generation with function validation"""
        # Code syntax verification
        # Function definition detection
        # Programming language accuracy
        
    def test_05_classify_text_endpoint(self):
        """Test classification with multiple scenarios"""
        # Multi-case classification testing
        # Category accuracy validation
        # Edge case handling
        
    def test_06_endpoint_error_handling(self):
        """Test comprehensive error scenarios"""
        # Missing field validation
        # Invalid request handling
        # Proper HTTP status codes
        
    def test_07_direct_wrapper_functions(self):
        """Test LLM wrapper functions directly"""
        # Direct function call validation
        # Retry logic verification
        # Error handling testing
        
    def test_08_swagger_documentation(self):
        """Test Swagger UI accessibility"""
        # Documentation endpoint validation
        # Interactive UI functionality
        # API specification accuracy
        
    def test_09_rate_limiting(self):
        """Test rate limiting functionality"""
        # Multiple request scenarios
        # Rate limit threshold testing
        # Recovery behavior validation
```

### 5. Example Usage and Validation (`example_usage.py`)

**Comprehensive API Testing Script:**

```python
def test_health_check():
    """Validate API health and availability"""
    # Server connectivity testing
    # Health endpoint validation
    # Status response verification
    
def test_text_generation():
    """Test creative text generation capabilities"""
    # Creative writing prompt testing
    # Response quality validation
    # Content length verification
    
def test_code_generation():
    """Test programming assistance functionality"""
    # Algorithm implementation requests
    # Code syntax validation
    # Function completeness testing
    
def test_text_classification():
    """Test multi-category classification"""
    # Sentiment analysis testing
    # Multi-case validation scenarios
    # Category accuracy measurement
    
def test_error_handling():
    """Test API error scenarios"""
    # Invalid request testing
    # Missing field validation
    # Error response format verification

def main():
    """Complete API validation workflow"""
    # Comprehensive test execution
    # Results summary generation
    # Success/failure reporting
```

## 🧪 Test Cases & Validation

Your implementation will be tested against these comprehensive scenarios:

### Test Case 1: Environment & Configuration (3 Tests)
```python
def test_01_env_api_key_configured(self):
    """Validate Google API key configuration"""
    # Check .env file existence
    # Validate API key format (AIza prefix)
    # Ensure environment variable loading

def test_02_flask_app_initialization(self):
    """Test Flask application setup"""
    # Health endpoint accessibility
    # CORS configuration validation
    # Extension initialization verification

def test_03_swagger_documentation(self):
    """Validate interactive documentation"""
    # Swagger UI accessibility at /swagger/
    # API specification completeness
    # Interactive testing capability
```

### Test Case 2: Core API Functionality (3 Tests)
```python
def test_04_generate_text_endpoint(self):
    """Test text generation with live API"""
    # Real Gemini API integration
    # Creative content generation
    # Response format validation

def test_05_generate_code_endpoint(self):
    """Test code generation capabilities"""
    # Programming language accuracy
    # Function definition generation
    # Code syntax verification

def test_06_classify_text_endpoint(self):
    """Test text classification accuracy"""
    # Multi-category classification
    # Sentiment analysis validation
    # Edge case handling
```

### Test Case 3: Production Readiness (3 Tests)
```python
def test_07_endpoint_error_handling(self):
    """Test comprehensive error scenarios"""
    # Missing field validation (400 errors)
    # Invalid request handling
    # Proper HTTP status code responses

def test_08_rate_limiting(self):
    """Test rate limiting functionality"""
    # Multiple rapid request scenarios
    # Rate limit threshold validation
    # Recovery behavior testing

def test_09_direct_wrapper_functions(self):
    """Test LLM wrapper functions directly"""
    # Retry logic verification
    # Exponential backoff testing
    # Error handling validation
```

## 📊 Evaluation Criteria

Your solution will be evaluated on:

1. **API Design & Documentation** (25%): RESTful design, Swagger integration, comprehensive documentation
2. **LLM Integration & Reliability** (25%): Gemini integration, retry logic, rate limit handling
3. **Production Readiness** (20%): Error handling, CORS, health checks, deployment preparation
4. **Code Quality & Testing** (15%): Clean architecture, comprehensive test coverage, live validation
5. **Advanced Features** (15%): Rate limiting, exponential backoff, Windows compatibility

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

# Production & Reliability
tenacity>=8.2.0
waitress>=2.1.0
```

### Environment Configuration
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### File Structure
```
Multi-Task-LLM-API/
├── app.py                    # Main Flask application with Swagger
├── gemini_wrapper.py         # LLM integration with retry logic  
├── unit_test.py             # Comprehensive test suite (9 tests)
├── example_usage.py         # API validation and demonstration
├── requirements.txt         # All dependencies
├── .env                     # Environment configuration
├── .gitignore              # Git ignore patterns
├── README.md               # Comprehensive documentation
└── FEATURES.md             # Feature specifications
```

### Performance Requirements
- **API Response Time**: <5 seconds per request under normal conditions
- **Rate Limiting**: 200/day, 50/hour, 10/minute per endpoint
- **Error Recovery**: 100% graceful handling of API failures with exponential backoff
- **Scalability**: Support concurrent requests with thread-safe operations
- **Documentation**: Interactive Swagger UI with complete API specification

## 🚀 Advanced Features (Bonus Points)

Implement these for extra credit:

1. **Advanced Error Analytics**: Detailed error categorization and logging
2. **Performance Monitoring**: Response time tracking and optimization
3. **Custom Model Configurations**: Per-use-case temperature and parameter tuning
4. **Batch Processing**: Multiple request handling in single API calls
5. **API Key Management**: Multi-user API key support and usage tracking
6. **Caching Layer**: Response caching for improved performance
7. **WebSocket Support**: Real-time streaming for long-form content generation
8. **Docker Containerization**: Complete containerized deployment solution

## 📝 Implementation Guidelines

### Flask Application Pattern
```python
@ns_generate.route('/text')
class TextGeneration(Resource):
    @ns_generate.expect(text_generation_model)
    @ns_generate.doc(description='Generate text using Gemini 2.0 Flash')
    @limiter.limit("10 per minute")
    def post(self):
        """Production-ready text generation endpoint"""
        # Comprehensive input validation
        # Error handling with proper HTTP codes
        # Structured response formatting
        # Logging and monitoring integration
```

### Retry Logic Implementation
```python
@retry(
    retry=retry_on_rate_limit,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    reraise=True
)
def api_function(prompt):
    """Intelligent retry with exponential backoff"""
    # Gemini API rate limit detection
    # Automatic retry with increasing delays
    # Comprehensive error logging
    # Graceful failure handling
```

### Production Deployment
```bash
# Development server
python app.py
# Runs on http://0.0.0.0:8081

# Production server with Waitress
waitress-serve --host=0.0.0.0 --port=8081 app:app
# Production WSGI server with better performance
```

## 🎯 Success Criteria

Your implementation is successful when:

- ✅ All 9 unit tests pass with live server validation
- ✅ Swagger UI is accessible and fully functional at `/swagger/`
- ✅ All three AI capabilities work with real Gemini API integration
- ✅ Rate limiting prevents API abuse while allowing normal usage
- ✅ Exponential backoff handles Gemini's 60 RPM rate limits gracefully
- ✅ Error handling provides meaningful responses for all failure scenarios
- ✅ CORS is properly configured for cross-origin requests
- ✅ Production deployment works with Waitress WSGI server
- ✅ Example usage script demonstrates all functionality successfully

## 📋 Submission Requirements

### Required Files
1. **Core Application** (4 files):
   - `app.py`: Flask application with Swagger integration
   - `gemini_wrapper.py`: LLM wrapper with retry logic
   - `unit_test.py`: Comprehensive test suite (9 tests)
   - `example_usage.py`: API validation script

2. **Configuration & Documentation** (4 files):
   - `requirements.txt`: All required dependencies
   - `.env`: Environment template (without actual API key)
   - `README.md`: Comprehensive documentation
   - `FEATURES.md`: Feature specifications

3. **Development Support** (1 file):
   - `.gitignore`: Git ignore patterns for Python projects

### Code Quality Standards
- **Production Architecture**: Clean separation of concerns, modular design
- **Error Handling**: Comprehensive exception management with proper HTTP codes
- **API Design**: RESTful principles with consistent response formats
- **Documentation**: Interactive Swagger UI with complete endpoint documentation
- **Testing**: Live server testing with real API integration
- **Reliability**: Exponential backoff and intelligent retry logic

## 🔍 Sample Usage Examples

### Complete API Testing
```bash
# Start the production server
python app.py

# Expected output:
# * Running on http://0.0.0.0:8081
# * Swagger UI available at: http://0.0.0.0:8081/swagger/
```

### Run Comprehensive Tests
```bash
# Execute full test suite with live server
python unit_test.py

# Expected results:
# test_01_env_api_key_configured ... ok
# test_02_flask_app_initialization ... ok  
# test_03_swagger_documentation ... ok
# test_04_generate_text_endpoint ... ok
# test_05_generate_code_endpoint ... ok
# test_06_classify_text_endpoint ... ok
# test_07_endpoint_error_handling ... ok
# test_08_rate_limiting ... ok
# test_09_direct_wrapper_functions ... ok
#
# Ran 9 tests in X.XXXs - OK
```

### API Validation Script
```bash
# Test all endpoints with example usage
python example_usage.py

# Expected output:
# 🏥 Testing Health Check... ✅
# 📝 Testing Text Generation... ✅  
# 💻 Testing Code Generation... ✅
# 🏷️ Testing Text Classification... ✅
# ❌ Testing Error Handling... ✅
#
# 🎯 Overall: 5/5 tests passed
# 🎉 All tests completed successfully!
```

## ⚠️ Important Notes

- **API Key Security**: Never commit real API keys to version control
- **Rate Limiting**: Implement proper delays and retry logic for Gemini's 60 RPM limit
- **Error Resilience**: System should never crash on API failures
- **Cross-Platform**: Code must work on Windows, macOS, and Linux
- **Production Ready**: Use Waitress for production deployment
- **Documentation**: Swagger UI must be fully functional and comprehensive
- **Testing**: All tests must pass with live API integration

Build a production-ready Multi-Task LLM API that demonstrates expert-level skills in Flask development, LLM integration, API design, error handling, and production deployment! 🚀

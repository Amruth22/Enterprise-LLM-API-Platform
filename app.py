
from flask import Flask, request, jsonify, g
from flask_restx import Api, Resource, fields
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import os
import time

from gemini_wrapper import generate_text, generate_code, classify_text
from lru_cache import ResponseCache, RateLimitCache
from cost_tracker import CostTracker
from enhanced_logging import request_logger

app = Flask(__name__)

# Configure CORS
CORS(app, 
     origins=['*'],  # Configure specific origins in production
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'],
     expose_headers=['X-Request-ID', 'X-Cost-Input', 'X-Cost-Output', 'X-Cost-Total', 'X-Cost-Overall', 'X-Tokens-Input', 'X-Tokens-Output', 'X-Cached']
)

# Initialize caches and trackers
response_cache = ResponseCache(max_size=500, default_ttl=1800)  # 30 minutes
rate_limit_cache = RateLimitCache(max_size=10000, default_ttl=3600)  # 1 hour
cost_tracker = CostTracker()

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

api = Api(
    app, 
    version='1.0', 
    title='Multi-Task LLM API',
    description='A Flask API that provides text generation, code generation, and text classification using Google Gemini Pro',
    doc='/swagger/',
    prefix='/api/v1'
)

# Custom rate limiting function using LRU cache
def get_rate_limit_key():
    return get_remote_address()

def check_rate_limit(key: str, limit: int, window: int = 60) -> bool:
    """Check if rate limit is exceeded"""
    current_count = rate_limit_cache.increment_count(key, window)
    return current_count <= limit

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",  # Keep as fallback
)

ns_generate = api.namespace('generate', description='Text and code generation operations')
ns_classify = api.namespace('classify', description='Text classification operations')

text_generation_model = api.model('TextGeneration', {
    'prompt': fields.String(required=True, description='The text prompt for generation', example='Write a short story about a robot')
})

code_generation_model = api.model('CodeGeneration', {
    'prompt': fields.String(required=True, description='The prompt for code generation', example='Create a Python function to calculate fibonacci numbers')
})

text_classification_model = api.model('TextClassification', {
    'text': fields.String(required=True, description='The text to classify', example='This movie was amazing!'),
    'categories': fields.List(fields.String, required=True, description='List of categories to classify into', example=['positive', 'negative', 'neutral'])
})

text_response_model = api.model('TextResponse', {
    'generated_text': fields.String(description='The generated text response')
})

code_response_model = api.model('CodeResponse', {
    'generated_code': fields.String(description='The generated code response')
})

classification_response_model = api.model('ClassificationResponse', {
    'classification': fields.String(description='The classification result')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

# Add health check endpoint
@app.route('/api/v1/health')
def health_check():
    return {'status': 'healthy', 'timestamp': time.time()}

# Add cache stats endpoint
@app.route('/api/v1/stats')
def get_stats():
    return {
        'cache_stats': response_cache.get_stats(),
        'usage_stats': cost_tracker.get_usage_stats(24),  # Last 24 hours
        'rate_limit_stats': {
            'cache_size': rate_limit_cache.size(),
            'max_size': rate_limit_cache.max_size
        }
    }

# Before request handler
@app.before_request
def before_request():
    # Start request logging
    request_logger.log_request_start()
    
    # Custom rate limiting check
    if request.endpoint in ['textgeneration', 'codegeneration', 'textclassification']:
        client_ip = get_remote_address()
        if not check_rate_limit(client_ip, 10, 60):  # 10 requests per minute
            return jsonify({'error': 'Rate limit exceeded'}), 429

# After request handler
@app.after_request
def after_request(response):
    # Add request ID header
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    # Add cost headers if available
    if hasattr(g, 'cost_info'):
        cost_info = g.cost_info
        response.headers['X-Cost-Input'] = f"{cost_info.get('input_cost', 0):.6f}"
        response.headers['X-Cost-Output'] = f"{cost_info.get('output_cost', 0):.6f}"
        response.headers['X-Cost-Total'] = f"{cost_info.get('total_cost', 0):.6f}"
        response.headers['X-Cost-Overall'] = f"{cost_info.get('overall_cost', 0):.6f}"
        response.headers['X-Cached'] = str(cost_info.get('cached', False))
        
        # Add token count headers for transparency
        if hasattr(g, 'token_info'):
            token_info = g.token_info
            response.headers['X-Tokens-Input'] = str(token_info.get('input_tokens', 0))
            response.headers['X-Tokens-Output'] = str(token_info.get('output_tokens', 0))
    
    return response




@ns_generate.route('/text')
class TextGeneration(Resource):
    @ns_generate.expect(text_generation_model)
    @ns_generate.doc(
        description='Generate text based on a given prompt using Google Gemini Pro',
        responses={
            200: 'Success - Text generated',
            400: 'Bad Request - Missing prompt',
            500: 'Internal Server Error'
        }
    )
    @limiter.limit("10 per minute")
    def post(self):
        data = request.get_json()
        prompt = data.get('prompt')
        if not prompt:
            response_data = {'error': 'Prompt is required'}
            request_logger.log_request_end(response_data, 400)
            return response_data, 400
        
        try:
            # Check cache first
            cached_response = response_cache.get_cached_response(prompt, 'text')
            
            if cached_response:
                # Serve from cache
                response_data = {'generated_text': cached_response['response']}
                
                # Calculate original costs from cached data
                input_tokens = cached_response.get('input_tokens', 0)
                output_tokens = cached_response.get('output_tokens', 0)
                input_cost, output_cost, _ = cost_tracker.token_counter.calculate_cost(input_tokens, output_tokens)
                
                # Set cost info for headers (including overall stats)
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': input_cost,
                    'output_cost': output_cost, 
                    'total_cost': 0.0,  # No cost for cached responses
                    'cached': True,
                    'overall_cost': overall_cost
                }
                
                # Set token info for headers
                g.token_info = {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens
                }
                
                # Track cached request
                cost_tracker.track_request(
                    '/api/v1/generate/text',
                    get_remote_address(),
                    prompt,
                    cached_response['response'],
                    cached=True
                )
                
                request_logger.log_request_end(response_data, 200, cached=True, cost_info=g.cost_info)
                return response_data
            
            # Generate new response
            text = generate_text(prompt)
            
            # Track cost
            cost_record = cost_tracker.track_request(
                '/api/v1/generate/text',
                get_remote_address(),
                prompt,
                text
            )
            
            # Cache the response
            response_cache.cache_response(
                prompt, 'text', text,
                cost_record.input_tokens,
                cost_record.output_tokens,
                cost_record.total_cost
            )
            
            response_data = {'generated_text': text}
            
            # Set cost info for headers (including overall stats)
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            # Set token info for headers
            g.token_info = {
                'input_tokens': cost_record.input_tokens,
                'output_tokens': cost_record.output_tokens
            }
            
            request_logger.log_request_end(response_data, 200, cached=False, cost_info=g.cost_info)
            return response_data
            
        except Exception as e:
            request_logger.log_error(e)
            error_response = {'error': str(e)}
            request_logger.log_request_end(error_response, 500)
            return error_response, 500

@ns_generate.route('/code')
class CodeGeneration(Resource):
    @ns_generate.expect(code_generation_model)
    @ns_generate.doc(
        description='Generate code based on a given prompt using Google Gemini Pro with optimized settings for code generation',
        responses={
            200: 'Success - Code generated',
            400: 'Bad Request - Missing prompt',
            500: 'Internal Server Error'
        }
    )
    @limiter.limit("10 per minute")
    def post(self):
        data = request.get_json()
        prompt = data.get('prompt')

        if not prompt:
            response_data = {'error': 'Prompt is required'}
            request_logger.log_request_end(response_data, 400)
            return response_data, 400

        try:
            # Check cache first
            cached_response = response_cache.get_cached_response(prompt, 'code')
            
            if cached_response:
                # Serve from cache
                response_data = {'generated_code': cached_response['response']}
                
                # Calculate original costs from cached data
                input_tokens = cached_response.get('input_tokens', 0)
                output_tokens = cached_response.get('output_tokens', 0)
                input_cost, output_cost, _ = cost_tracker.token_counter.calculate_cost(input_tokens, output_tokens)
                
                # Set cost info for headers (including overall stats)
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': input_cost,
                    'output_cost': output_cost,
                    'total_cost': 0.0,  # No cost for cached responses
                    'cached': True,
                    'overall_cost': overall_cost
                }
                
                # Track cached request
                cost_tracker.track_request(
                    '/api/v1/generate/code',
                    get_remote_address(),
                    prompt,
                    cached_response['response'],
                    cached=True
                )
                
                request_logger.log_request_end(response_data, 200, cached=True, cost_info=g.cost_info)
                return response_data
            
            # Generate new response
            code = generate_code(prompt)
            
            # Track cost
            cost_record = cost_tracker.track_request(
                '/api/v1/generate/code',
                get_remote_address(),
                prompt,
                code
            )
            
            # Cache the response
            response_cache.cache_response(
                prompt, 'code', code,
                cost_record.input_tokens,
                cost_record.output_tokens,
                cost_record.total_cost
            )
            
            response_data = {'generated_code': code}
            
            # Set cost info for headers (including overall stats)
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            # Set token info for headers
            g.token_info = {
                'input_tokens': cost_record.input_tokens,
                'output_tokens': cost_record.output_tokens
            }
            
            request_logger.log_request_end(response_data, 200, cached=False, cost_info=g.cost_info)
            return response_data
            
        except Exception as e:
            request_logger.log_error(e)
            error_response = {'error': str(e)}
            request_logger.log_request_end(error_response, 500)
            return error_response, 500

@ns_classify.route('/text')
class TextClassification(Resource):
    @ns_classify.expect(text_classification_model)
    @ns_classify.doc(
        description='Classify text into one of the provided categories using Google Gemini Pro',
        responses={
            200: 'Success - Text classified',
            400: 'Bad Request - Missing text or categories',
            500: 'Internal Server Error'
        }
    )
    @limiter.limit("10 per minute")
    def post(self):
        data = request.get_json()
        text = data.get('text')
        categories = data.get('categories')

        if not text or not categories:
            response_data = {'error': 'Text and categories are required'}
            request_logger.log_request_end(response_data, 400)
            return response_data, 400

        try:
            # Create cache key combining text and categories
            cache_prompt = f"text:{text}|categories:{','.join(categories)}"
            
            # Check cache first
            cached_response = response_cache.get_cached_response(cache_prompt, 'classify')
            
            if cached_response:
                # Serve from cache
                response_data = {'classification': cached_response['response']}
                
                # Calculate original costs from cached data
                input_tokens = cached_response.get('input_tokens', 0)
                output_tokens = cached_response.get('output_tokens', 0)
                input_cost, output_cost, _ = cost_tracker.token_counter.calculate_cost(input_tokens, output_tokens)
                
                # Set cost info for headers (including overall stats)
                overall_cost = sum(r.total_cost for r in cost_tracker.records)
                g.cost_info = {
                    'input_cost': input_cost,
                    'output_cost': output_cost,
                    'total_cost': 0.0,  # No cost for cached responses
                    'cached': True,
                    'overall_cost': overall_cost
                }
                
                # Track cached request
                cost_tracker.track_request(
                    '/api/v1/classify/text',
                    get_remote_address(),
                    cache_prompt,
                    cached_response['response'],
                    cached=True
                )
                
                request_logger.log_request_end(response_data, 200, cached=True, cost_info=g.cost_info)
                return response_data
            
            # Generate new response
            classification = classify_text(text, categories)
            
            # Track cost
            cost_record = cost_tracker.track_request(
                '/api/v1/classify/text',
                get_remote_address(),
                cache_prompt,
                classification
            )
            
            # Cache the response
            response_cache.cache_response(
                cache_prompt, 'classify', classification,
                cost_record.input_tokens,
                cost_record.output_tokens,
                cost_record.total_cost
            )
            
            response_data = {'classification': classification}
            
            # Set cost info for headers (including overall stats)
            overall_cost = sum(r.total_cost for r in cost_tracker.records)
            g.cost_info = {
                'input_cost': cost_record.input_cost,
                'output_cost': cost_record.output_cost,
                'total_cost': cost_record.total_cost,
                'cached': False,
                'overall_cost': overall_cost
            }
            
            # Set token info for headers
            g.token_info = {
                'input_tokens': cost_record.input_tokens,
                'output_tokens': cost_record.output_tokens
            }
            
            request_logger.log_request_end(response_data, 200, cached=False, cost_info=g.cost_info)
            return response_data
            
        except Exception as e:
            request_logger.log_error(e)
            error_response = {'error': str(e)}
            request_logger.log_request_end(error_response, 500)
            return error_response, 500

if __name__ == '__main__':
    app.run(debug=True)

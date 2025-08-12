import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from flask import request, g
import uuid


class RequestLogger:
    """Enhanced request/response logging with audit trails"""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize enhanced logging
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup different loggers for different purposes
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup various loggers"""
        
        # Request/Response Logger
        self.request_logger = logging.getLogger('api_requests')
        self.request_logger.setLevel(logging.INFO)
        request_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'api_requests.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        request_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        request_handler.setFormatter(request_formatter)
        self.request_logger.addHandler(request_handler)
        
        # Security/Audit Logger
        self.security_logger = logging.getLogger('security_audit')
        self.security_logger.setLevel(logging.WARNING)
        security_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'security_audit.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        self.security_logger.addHandler(security_handler)
        
        # Error Logger
        self.error_logger = logging.getLogger('api_errors')
        self.error_logger.setLevel(logging.ERROR)
        error_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'api_errors.log'),
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10
        )
        error_formatter = logging.Formatter(
            '%(asctime)s - ERROR - %(message)s - %(exc_info)s'
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Performance Logger
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)
        perf_handler = RotatingFileHandler(
            os.path.join(self.log_dir, 'performance.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
    
    def log_request_start(self, request_id: str = None) -> str:
        """
        Log the start of a request
        
        Args:
            request_id: Optional request ID (generated if not provided)
            
        Returns:
            Request ID for tracking
        """
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store request start time and ID in Flask's g object
        g.request_id = request_id
        g.request_start_time = time.time()
        
        # Get request details
        user_ip = self._get_client_ip()
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        request_data = {
            'request_id': request_id,
            'method': request.method,
            'endpoint': request.endpoint,
            'path': request.path,
            'user_ip': user_ip,
            'user_agent': user_agent,
            'content_type': request.content_type,
            'content_length': request.content_length,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        # Log request details
        self.request_logger.info(f"REQUEST_START: {json.dumps(request_data)}")
        
        # Security logging for suspicious activity
        self._check_security_issues(request_data)
        
        return request_id
    
    def log_request_end(self, response_data: Dict[str, Any], status_code: int, 
                       cached: bool = False, cost_info: Optional[Dict] = None):
        """
        Log the end of a request
        
        Args:
            response_data: Response data
            status_code: HTTP status code
            cached: Whether response was cached
            cost_info: Cost tracking information
        """
        request_id = getattr(g, 'request_id', 'unknown')
        start_time = getattr(g, 'request_start_time', time.time())
        
        duration = time.time() - start_time
        
        response_log = {
            'request_id': request_id,
            'status_code': status_code,
            'response_size': len(json.dumps(response_data)) if response_data else 0,
            'duration_ms': round(duration * 1000, 2),
            'cached': cached,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        # Add cost information if provided
        if cost_info:
            response_log['cost_info'] = cost_info
        
        self.request_logger.info(f"REQUEST_END: {json.dumps(response_log)}")
        
        # Performance logging
        self._log_performance(request_id, duration, cached, cost_info)
        
        # Error logging if applicable
        if status_code >= 400:
            self._log_error(request_id, status_code, response_data, duration)
    
    def log_error(self, error: Exception, request_id: str = None, 
                  additional_context: Dict[str, Any] = None):
        """
        Log an error with full context
        
        Args:
            error: Exception object
            request_id: Request ID for tracking
            additional_context: Additional context information
        """
        request_id = request_id or getattr(g, 'request_id', 'unknown')
        
        error_data = {
            'request_id': request_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'endpoint': request.endpoint if request else 'unknown',
            'user_ip': self._get_client_ip(),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        if additional_context:
            error_data['context'] = additional_context
        
        self.error_logger.error(f"API_ERROR: {json.dumps(error_data)}", exc_info=True)
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """
        Log security-related events
        
        Args:
            event_type: Type of security event
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details
        """
        security_data = {
            'event_type': event_type,
            'severity': severity,
            'user_ip': self._get_client_ip(),
            'user_agent': request.headers.get('User-Agent', 'Unknown') if request else 'Unknown',
            'endpoint': request.endpoint if request else 'unknown',
            'request_id': getattr(g, 'request_id', 'unknown'),
            'details': details,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        if severity in ['HIGH', 'CRITICAL']:
            self.security_logger.critical(f"SECURITY_EVENT: {json.dumps(security_data)}")
        elif severity == 'MEDIUM':
            self.security_logger.warning(f"SECURITY_EVENT: {json.dumps(security_data)}")
        else:
            self.security_logger.info(f"SECURITY_EVENT: {json.dumps(security_data)}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address"""
        if not request:
            return 'unknown'
            
        # Check for forwarded IP (behind proxy/load balancer)
        if 'X-Forwarded-For' in request.headers:
            return request.headers['X-Forwarded-For'].split(',')[0].strip()
        elif 'X-Real-IP' in request.headers:
            return request.headers['X-Real-IP']
        else:
            return request.remote_addr or 'unknown'
    
    def _check_security_issues(self, request_data: Dict[str, Any]):
        """Check for potential security issues"""
        user_ip = request_data.get('user_ip', '')
        path = request_data.get('path', '')
        user_agent = request_data.get('user_agent', '')
        
        # Check for suspicious patterns
        suspicious_patterns = [
            '../', '..\\', '<script>', 'DROP TABLE', 'SELECT *', 
            'UNION SELECT', 'eval(', 'javascript:', 'data:text/html'
        ]
        
        if any(pattern.lower() in path.lower() for pattern in suspicious_patterns):
            self.log_security_event(
                'SUSPICIOUS_PATH',
                'HIGH',
                {'path': path, 'patterns_detected': [p for p in suspicious_patterns if p.lower() in path.lower()]}
            )
        
        # Check for missing or suspicious User-Agent
        if not user_agent or user_agent.lower() in ['unknown', 'curl', 'wget', 'python-requests']:
            self.log_security_event(
                'SUSPICIOUS_USER_AGENT',
                'MEDIUM',
                {'user_agent': user_agent}
            )
    
    def _log_performance(self, request_id: str, duration: float, 
                        cached: bool, cost_info: Optional[Dict]):
        """Log performance metrics"""
        perf_data = {
            'request_id': request_id,
            'duration_ms': round(duration * 1000, 2),
            'cached': cached,
            'endpoint': request.endpoint if request else 'unknown'
        }
        
        if cost_info:
            perf_data.update({
                'input_tokens': cost_info.get('input_tokens', 0),
                'output_tokens': cost_info.get('output_tokens', 0),
                'total_cost': cost_info.get('total_cost', 0.0)
            })
        
        # Flag slow requests
        if duration > 5.0:  # More than 5 seconds
            perf_data['slow_request'] = True
            
        self.performance_logger.info(json.dumps(perf_data))
    
    def _log_error(self, request_id: str, status_code: int, 
                   response_data: Dict[str, Any], duration: float):
        """Log error responses"""
        error_data = {
            'request_id': request_id,
            'status_code': status_code,
            'error_message': response_data.get('error', 'Unknown error') if response_data else 'No response data',
            'duration_ms': round(duration * 1000, 2),
            'endpoint': request.endpoint if request else 'unknown',
            'user_ip': self._get_client_ip(),
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        if status_code >= 500:
            self.error_logger.error(f"SERVER_ERROR: {json.dumps(error_data)}")
        elif status_code >= 400:
            self.error_logger.warning(f"CLIENT_ERROR: {json.dumps(error_data)}")


# Global logger instance
request_logger = RequestLogger()
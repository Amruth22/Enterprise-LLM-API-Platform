import time
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import tiktoken


@dataclass
class CostRecord:
    """Record for tracking API costs"""
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


class TokenCounter:
    """Token counting utility for Gemini models"""
    
    # Gemini 2.0 Flash pricing per 1M tokens
    INPUT_COST_PER_MILLION = 0.10
    OUTPUT_COST_PER_MILLION = 0.40
    
    def __init__(self):
        """Initialize token counter"""
        try:
            # Use cl100k_base encoding as approximation for Gemini
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logging.warning(f"Could not load tiktoken encoding: {e}")
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logging.warning(f"Error counting tokens with tiktoken: {e}")
        
        # Fallback: rough estimation (1 token ≈ 4 characters for English)
        return max(1, len(text) // 4)
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> tuple[float, float, float]:
        """
        Calculate cost for input and output tokens
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost


class CostTracker:
    """Track and manage API costs"""
    
    def __init__(self, log_file: str = "logs/cost_tracking.log"):
        """
        Initialize cost tracker
        
        Args:
            log_file: Path to cost log file
        """
        self.log_file = log_file
        self.token_counter = TokenCounter()
        self.records: List[CostRecord] = []
        self.lock = threading.RLock()
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup cost logger
        self.logger = logging.getLogger('cost_tracker')
        self.logger.setLevel(logging.INFO)
        
        # File handler for cost logs
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def track_request(self, endpoint: str, user_ip: str, prompt: str, 
                     response: str, cached: bool = False) -> CostRecord:
        """
        Track a single API request
        
        Args:
            endpoint: API endpoint used
            user_ip: User's IP address
            prompt: User prompt
            response: API response
            cached: Whether response was served from cache
            
        Returns:
            CostRecord object
        """
        with self.lock:
            # Count tokens
            input_tokens = self.token_counter.count_tokens(prompt)
            output_tokens = self.token_counter.count_tokens(response)
            
            # Calculate costs
            input_cost, output_cost, total_cost = self.token_counter.calculate_cost(
                input_tokens, output_tokens
            )
            
            # Create cost record
            record = CostRecord(
                timestamp=time.time(),
                endpoint=endpoint,
                user_ip=user_ip,
                prompt=prompt[:100] + "..." if len(prompt) > 100 else prompt,  # Truncate for storage
                response=response[:100] + "..." if len(response) > 100 else response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost if not cached else 0.0,  # No cost for cached responses
                cached=cached
            )
            
            # Add to records
            self.records.append(record)
            
            # Log the cost
            self.logger.info(json.dumps({
                'timestamp': record.timestamp,
                'endpoint': record.endpoint,
                'user_ip': record.user_ip,
                'input_tokens': record.input_tokens,
                'output_tokens': record.output_tokens,
                'input_cost': round(record.input_cost, 6),
                'output_cost': round(record.output_cost, 6),
                'total_cost': round(record.total_cost, 6),
                'cached': record.cached
            }))
            
            return record
    
    def get_usage_stats(self, hours: int = 24) -> Dict:
        """
        Get usage statistics for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with usage statistics
        """
        with self.lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_records = [r for r in self.records if r.timestamp >= cutoff_time]
            
            if not recent_records:
                return {
                    'period_hours': hours,
                    'total_requests': 0,
                    'cached_requests': 0,
                    'cache_hit_rate': 0.0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_cost': 0.0,
                    'average_cost_per_request': 0.0,
                    'endpoints': {}
                }
            
            total_requests = len(recent_records)
            cached_requests = sum(1 for r in recent_records if r.cached)
            total_input_tokens = sum(r.input_tokens for r in recent_records)
            total_output_tokens = sum(r.output_tokens for r in recent_records)
            total_cost = sum(r.total_cost for r in recent_records)
            
            # Per-endpoint stats
            endpoint_stats = {}
            for record in recent_records:
                endpoint = record.endpoint
                if endpoint not in endpoint_stats:
                    endpoint_stats[endpoint] = {
                        'requests': 0,
                        'cached_requests': 0,
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'cost': 0.0
                    }
                
                stats = endpoint_stats[endpoint]
                stats['requests'] += 1
                if record.cached:
                    stats['cached_requests'] += 1
                stats['input_tokens'] += record.input_tokens
                stats['output_tokens'] += record.output_tokens
                stats['cost'] += record.total_cost
            
            return {
                'period_hours': hours,
                'total_requests': total_requests,
                'cached_requests': cached_requests,
                'cache_hit_rate': cached_requests / total_requests if total_requests > 0 else 0.0,
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_cost': round(total_cost, 6),
                'average_cost_per_request': round(total_cost / total_requests, 6) if total_requests > 0 else 0.0,
                'cost_savings_from_cache': round(
                    sum(r.input_cost + r.output_cost for r in recent_records if r.cached), 6
                ),
                'endpoints': endpoint_stats
            }
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """
        Remove records older than specified days
        
        Args:
            days: Number of days to keep records
            
        Returns:
            Number of records removed
        """
        with self.lock:
            cutoff_time = time.time() - (days * 24 * 3600)
            original_count = len(self.records)
            self.records = [r for r in self.records if r.timestamp >= cutoff_time]
            removed_count = original_count - len(self.records)
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old cost records")
            
            return removed_count
    
    def export_records(self, output_file: str, hours: Optional[int] = None) -> int:
        """
        Export cost records to JSON file
        
        Args:
            output_file: Path to output file
            hours: Number of hours to export (None for all records)
            
        Returns:
            Number of records exported
        """
        with self.lock:
            if hours:
                cutoff_time = time.time() - (hours * 3600)
                records_to_export = [r for r in self.records if r.timestamp >= cutoff_time]
            else:
                records_to_export = self.records
            
            # Convert to dict for JSON serialization
            export_data = {
                'export_timestamp': time.time(),
                'export_date': datetime.now().isoformat(),
                'total_records': len(records_to_export),
                'records': [asdict(record) for record in records_to_export]
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return len(records_to_export)
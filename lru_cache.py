import time
import threading
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import json
import hashlib


class LRUCache:
    """Thread-safe LRU Cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize LRU Cache
        
        Args:
            max_size: Maximum number of items to store
            default_ttl: Default Time To Live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        
    def _generate_key(self, data: Any) -> str:
        """Generate a consistent key from data"""
        if isinstance(data, dict):
            # Sort dict to ensure consistent hashing
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _is_expired(self, expire_time: float) -> bool:
        """Check if item has expired"""
        return time.time() > expire_time
    
    def _evict_expired(self):
        """Remove expired items from cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, (_, expire_time) in self.cache.items():
            if current_time > expire_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                value, expire_time = self.cache[key]
                if not self._is_expired(expire_time):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return value
                else:
                    # Item expired, remove it
                    del self.cache[key]
            
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put item in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        with self.lock:
            ttl = ttl or self.default_ttl
            expire_time = time.time() + ttl
            
            # Remove expired items first
            self._evict_expired()
            
            # Update existing item
            if key in self.cache:
                self.cache[key] = (value, expire_time)
                self.cache.move_to_end(key)
                return
            
            # Add new item
            self.cache[key] = (value, expire_time)
            self.cache.move_to_end(key)
            
            # Evict least recently used if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            self._evict_expired()
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            self._evict_expired()
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_ratio': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_count', 1), 1),
                'default_ttl': self.default_ttl
            }


class ResponseCache(LRUCache):
    """Specialized LRU cache for API responses"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 1800):  # 30 minutes default
        super().__init__(max_size, default_ttl)
        self._hit_count = 0
        self._miss_count = 0
        self._total_count = 0
    
    def get_cached_response(self, prompt: str, task_type: str) -> Optional[Dict]:
        """
        Get cached response for a prompt and task type
        
        Args:
            prompt: User prompt
            task_type: Type of task (text, code, classify)
            
        Returns:
            Cached response or None
        """
        cache_key = self._generate_key({'prompt': prompt, 'task_type': task_type})
        
        with self.lock:
            self._total_count += 1
            cached_response = self.get(cache_key)
            
            if cached_response:
                self._hit_count += 1
                return cached_response
            else:
                self._miss_count += 1
                return None
    
    def cache_response(self, prompt: str, task_type: str, response: Any, 
                      input_tokens: int, output_tokens: int, cost: float, ttl: Optional[int] = None) -> None:
        """
        Cache an API response
        
        Args:
            prompt: User prompt
            task_type: Type of task
            response: API response
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost of the API call
            ttl: Time to live (optional)
        """
        cache_key = self._generate_key({'prompt': prompt, 'task_type': task_type})
        
        cache_data = {
            'response': response,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'cached_at': time.time()
        }
        
        self.put(cache_key, cache_data, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        base_stats = super().get_stats()
        base_stats.update({
            'hit_count': self._hit_count,
            'miss_count': self._miss_count,
            'total_requests': self._total_count,
            'cache_hit_ratio': self._hit_count / max(self._total_count, 1)
        })
        return base_stats


class RateLimitCache(LRUCache):
    """Specialized LRU cache for rate limiting"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):  # 1 hour default
        super().__init__(max_size, default_ttl)
    
    def increment_count(self, key: str, window_size: int = 60) -> int:
        """
        Increment rate limit counter for a key
        
        Args:
            key: Rate limit key (usually IP address or user ID)
            window_size: Time window in seconds
            
        Returns:
            Current count for this key
        """
        current_time = int(time.time())
        window_start = current_time - (current_time % window_size)
        
        rate_key = f"{key}:{window_start}"
        
        with self.lock:
            current_count = self.get(rate_key) or 0
            new_count = current_count + 1
            self.put(rate_key, new_count, window_size)
            return new_count
    
    def get_count(self, key: str, window_size: int = 60) -> int:
        """
        Get current count for a key
        
        Args:
            key: Rate limit key
            window_size: Time window in seconds
            
        Returns:
            Current count
        """
        current_time = int(time.time())
        window_start = current_time - (current_time % window_size)
        
        rate_key = f"{key}:{window_start}"
        return self.get(rate_key) or 0
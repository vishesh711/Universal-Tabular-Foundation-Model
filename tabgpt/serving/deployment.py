"""Model deployment and serving infrastructure."""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import signal
import sys

import torch
import pandas as pd
import numpy as np

from .inference import InferenceEngine, OptimizedInferenceEngine, InferenceConfig, InferenceResult
from ..utils import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for model server."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Request handling
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 30
    max_concurrent_requests: int = 100
    
    # Model settings
    model_path: str = ""
    tokenizer_path: str = ""
    inference_config: Optional[InferenceConfig] = None
    
    # Health check
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Security
    enable_auth: bool = False
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")


class HealthChecker:
    """Health monitoring for model server."""
    
    def __init__(self, inference_engine: InferenceEngine, config: ServerConfig):
        """
        Initialize health checker.
        
        Args:
            inference_engine: Inference engine to monitor
            config: Server configuration
        """
        self.inference_engine = inference_engine
        self.config = config
        self.is_healthy = True
        self.last_check_time = time.time()
        self.health_history = []
        self.lock = threading.Lock()
        
        # Start health monitoring
        self.monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _health_monitor_loop(self):
        """Main health monitoring loop."""
        while True:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                with self.lock:
                    self.is_healthy = False
    
    def _perform_health_check(self):
        """Perform health check."""
        start_time = time.time()
        
        try:
            # Create simple test data
            test_data = pd.DataFrame({
                'feature1': [1.0],
                'feature2': ['test']
            })
            
            # Test inference
            result = self.inference_engine.predict(test_data)
            
            # Check if result is valid
            if result.predictions is None or len(result.predictions) == 0:
                raise Exception("Invalid prediction result")
            
            check_duration = time.time() - start_time
            
            # Update health status
            with self.lock:
                self.is_healthy = True
                self.last_check_time = time.time()
                
                # Keep history of last 100 checks
                self.health_history.append({
                    'timestamp': self.last_check_time,
                    'duration_ms': check_duration * 1000,
                    'status': 'healthy'
                })
                
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
            
            logger.debug(f"Health check passed in {check_duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
            with self.lock:
                self.is_healthy = False
                self.health_history.append({
                    'timestamp': time.time(),
                    'status': 'unhealthy',
                    'error': str(e)
                })
                
                if len(self.health_history) > 100:
                    self.health_history.pop(0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            stats = self.inference_engine.get_stats()
            
            return {
                'healthy': self.is_healthy,
                'last_check': self.last_check_time,
                'uptime_seconds': time.time() - (self.health_history[0]['timestamp'] if self.health_history else time.time()),
                'total_requests': stats.get('total_requests', 0),
                'total_errors': stats.get('errors', 0),
                'avg_response_time_ms': stats.get('avg_time_per_request_ms', 0),
                'cache_hit_rate': stats.get('cache_hit_rate', 0),
                'recent_checks': self.health_history[-10:]  # Last 10 checks
            }


class MetricsCollector:
    """Metrics collection for monitoring."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            'requests_total': 0,
            'requests_success': 0,
            'requests_error': 0,
            'response_time_sum': 0.0,
            'response_time_count': 0,
            'active_requests': 0,
            'model_inference_time_sum': 0.0,
            'model_inference_count': 0
        }
        self.lock = threading.Lock()
        
        # Histogram buckets for response times (in milliseconds)
        self.response_time_buckets = [10, 50, 100, 500, 1000, 5000, 10000]
        self.response_time_histogram = {bucket: 0 for bucket in self.response_time_buckets}
        self.response_time_histogram['inf'] = 0
    
    def record_request_start(self):
        """Record start of a request."""
        with self.lock:
            self.metrics['requests_total'] += 1
            self.metrics['active_requests'] += 1
    
    def record_request_end(self, success: bool, response_time_ms: float, inference_time_ms: float = 0):
        """Record end of a request."""
        with self.lock:
            self.metrics['active_requests'] -= 1
            
            if success:
                self.metrics['requests_success'] += 1
            else:
                self.metrics['requests_error'] += 1
            
            # Response time metrics
            self.metrics['response_time_sum'] += response_time_ms
            self.metrics['response_time_count'] += 1
            
            # Model inference time
            if inference_time_ms > 0:
                self.metrics['model_inference_time_sum'] += inference_time_ms
                self.metrics['model_inference_count'] += 1
            
            # Update histogram
            for bucket in self.response_time_buckets:
                if response_time_ms <= bucket:
                    self.response_time_histogram[bucket] += 1
                    break
            else:
                self.response_time_histogram['inf'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            metrics = self.metrics.copy()
            
            # Calculate derived metrics
            if metrics['response_time_count'] > 0:
                metrics['avg_response_time_ms'] = metrics['response_time_sum'] / metrics['response_time_count']
            else:
                metrics['avg_response_time_ms'] = 0
            
            if metrics['model_inference_count'] > 0:
                metrics['avg_inference_time_ms'] = metrics['model_inference_time_sum'] / metrics['model_inference_count']
            else:
                metrics['avg_inference_time_ms'] = 0
            
            if metrics['requests_total'] > 0:
                metrics['success_rate'] = metrics['requests_success'] / metrics['requests_total']
                metrics['error_rate'] = metrics['requests_error'] / metrics['requests_total']
            else:
                metrics['success_rate'] = 0
                metrics['error_rate'] = 0
            
            metrics['response_time_histogram'] = self.response_time_histogram.copy()
            
            return metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            for key in self.metrics:
                if isinstance(self.metrics[key], (int, float)):
                    self.metrics[key] = 0
            
            for bucket in self.response_time_histogram:
                self.response_time_histogram[bucket] = 0


class ModelServer:
    """High-performance model server for TabGPT."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize model server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.inference_engine = None
        self.health_checker = None
        self.metrics_collector = MetricsCollector()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))
        
        # Load model and setup inference
        self._setup_inference_engine()
        
        # Setup health monitoring
        if self.inference_engine:
            self.health_checker = HealthChecker(self.inference_engine, config)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_inference_engine(self):
        """Setup inference engine."""
        try:
            logger.info("Loading model and tokenizer...")
            
            # This is a simplified setup - in practice, you'd load actual models
            # model = TabGPTModel.from_pretrained(self.config.model_path)
            # tokenizer = TabGPTTokenizer.from_pretrained(self.config.tokenizer_path)
            
            # For demonstration, create mock objects
            class MockModel:
                def eval(self): return self
                def to(self, device): return self
                def parameters(self): return []
                def __call__(self, **kwargs):
                    # Mock prediction
                    batch_size = kwargs.get('input_ids', torch.tensor([[1]])).shape[0]
                    return type('Output', (), {
                        'predictions': torch.randn(batch_size, 1),
                        'logits': torch.randn(batch_size, 1)
                    })()
            
            class MockTokenizer:
                def encode_batch(self, df):
                    return {
                        'input_ids': torch.randint(0, 1000, (len(df), 10)),
                        'attention_mask': torch.ones(len(df), 10)
                    }
            
            model = MockModel()
            tokenizer = MockTokenizer()
            
            # Create inference engine
            inference_config = self.config.inference_config or InferenceConfig()
            self.inference_engine = OptimizedInferenceEngine(
                model=model,
                tokenizer=tokenizer,
                config=inference_config
            )
            
            logger.info("Model and inference engine loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup inference engine: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    async def predict_async(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async prediction endpoint.
        
        Args:
            request_data: Request data containing input
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        self.metrics_collector.record_request_start()
        
        try:
            # Validate request
            if 'data' not in request_data:
                raise ValidationError("Missing 'data' field in request")
            
            # Convert data to DataFrame
            data = request_data['data']
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise ValidationError("Data must be dict or list of dicts")
            
            # Get prediction options
            return_probabilities = request_data.get('return_probabilities', False)
            return_features = request_data.get('return_features', False)
            
            # Make prediction
            result = self.inference_engine.predict(
                df,
                return_probabilities=return_probabilities,
                return_features=return_features
            )
            
            # Format response
            response = {
                'predictions': result.predictions.tolist(),
                'metadata': {
                    'batch_size': result.batch_size,
                    'inference_time_ms': result.inference_time_ms,
                    'model_time_ms': result.model_time_ms,
                    'preprocessing_time_ms': result.preprocessing_time_ms,
                    'postprocessing_time_ms': result.postprocessing_time_ms
                }
            }
            
            if result.probabilities is not None:
                response['probabilities'] = result.probabilities.tolist()
            
            if result.features is not None:
                response['features'] = result.features.tolist()
            
            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_request_end(
                success=True,
                response_time_ms=response_time_ms,
                inference_time_ms=result.inference_time_ms
            )
            
            return response
            
        except Exception as e:
            # Record error
            response_time_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_request_end(
                success=False,
                response_time_ms=response_time_ms
            )
            
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_sync(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous prediction endpoint.
        
        Args:
            request_data: Request data containing input
            
        Returns:
            Prediction response
        """
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.predict_async(request_data))
        finally:
            loop.close()
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        if self.health_checker:
            return self.health_checker.get_health_status()
        else:
            return {'healthy': False, 'error': 'Health checker not initialized'}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        metrics = self.metrics_collector.get_metrics()
        
        # Add inference engine stats
        if self.inference_engine:
            engine_stats = self.inference_engine.get_stats()
            metrics.update({
                'engine_' + k: v for k, v in engine_stats.items()
            })
        
        return metrics
    
    def get_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            'server_version': '1.0.0',
            'model_path': self.config.model_path,
            'tokenizer_path': self.config.tokenizer_path,
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'workers': self.config.workers,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'request_timeout_seconds': self.config.request_timeout_seconds
            },
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def start(self):
        """Start the model server."""
        logger.info(f"Starting TabGPT model server on {self.config.host}:{self.config.port}")
        
        self.start_time = time.time()
        self.is_running = True
        
        try:
            # In a real implementation, you would start a web server here
            # For example, using FastAPI, Flask, or similar
            logger.info("Server started successfully")
            
            # Keep server running
            while not self.shutdown_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.is_running = False
    
    def shutdown(self):
        """Shutdown the server gracefully."""
        logger.info("Shutting down server...")
        
        self.shutdown_event.set()
        self.is_running = False
        
        # Shutdown inference engine
        if hasattr(self.inference_engine, 'shutdown'):
            self.inference_engine.shutdown()
        
        logger.info("Server shutdown complete")


def create_server_from_config(config_path: str) -> ModelServer:
    """
    Create server from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ModelServer instance
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create inference config if provided
    inference_config = None
    if 'inference_config' in config_dict:
        inference_config = InferenceConfig(**config_dict.pop('inference_config'))
    
    # Create server config
    config = ServerConfig(**config_dict)
    config.inference_config = inference_config
    
    return ModelServer(config)


def run_server_cli():
    """Command-line interface for running the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TabGPT Model Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    try:
        if args.config:
            # Load from config file
            server = create_server_from_config(args.config)
        else:
            # Create from CLI arguments
            config = ServerConfig(
                host=args.host,
                port=args.port,
                model_path=args.model_path or "",
                tokenizer_path=args.tokenizer_path or "",
                workers=args.workers,
                log_level=args.log_level
            )
            server = ModelServer(config)
        
        # Start server
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server_cli()
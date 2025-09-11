"""
Enhanced Performance Profiling and Optimization Module

This module provides comprehensive performance profiling and optimization
capabilities for the trading application.
"""

import asyncio
import time
import cProfile
import pstats
import io
import tracemalloc
import threading
import psutil
import gc
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from contextlib import contextmanager
import functools

from util.config_manager import get_config

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    function_name: str
    execution_time: float
    cpu_percent: float
    memory_usage: float
    call_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProfilingResult:
    """Complete profiling result."""
    function_name: str
    total_time: float
    calls: int
    avg_time_per_call: float
    cumulative_time: float
    callers: List[str]
    memory_allocated: Optional[float] = None
    memory_peak: Optional[float] = None

class PerformanceProfiler:
    """Enhanced performance profiler with comprehensive monitoring."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.profiling_results: Dict[str, ProfilingResult] = {}
        self._active_profiles: Dict[str, cProfile.Profile] = {}
        self.config = get_config()
        self._lock = threading.RLock()
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._profile_function_call(func, *args, **kwargs)
        return wrapper
    
    def _profile_function_call(self, func: Callable, *args, **kwargs):
        """Profile a single function call."""
        # Start memory tracking
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Start CPU monitoring
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop timing
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Stop memory tracking
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            
            memory_allocated = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
            memory_peak = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
            
            # Get CPU usage
            cpu_after = process.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Store metrics
            with self._lock:
                metric = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    cpu_percent=cpu_after - cpu_before,
                    memory_usage=memory_after - memory_before,
                    call_count=1,
                    additional_info={
                        'memory_allocated_mb': memory_allocated,
                        'memory_peak_mb': memory_peak
                    }
                )
                self.metrics.append(metric)
            
            # Log performance warning if function is slow
            if execution_time > 1.0:  # More than 1 second
                logger.warning(f"Slow function detected: {func.__name__} took {execution_time:.2f}s")
            
            tracemalloc.stop()
    
    @contextmanager
    def profile_block(self, block_name: str):
        """Context manager to profile a block of code."""
        # Start memory tracking
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Start CPU monitoring
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Stop timing
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Stop memory tracking
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            
            memory_allocated = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
            memory_peak = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
            
            # Get CPU usage
            cpu_after = process.cpu_percent()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Store metrics
            with self._lock:
                metric = PerformanceMetrics(
                    function_name=block_name,
                    execution_time=execution_time,
                    cpu_percent=cpu_after - cpu_before,
                    memory_usage=memory_after - memory_before,
                    call_count=1,
                    additional_info={
                        'memory_allocated_mb': memory_allocated,
                        'memory_peak_mb': memory_peak
                    }
                )
                self.metrics.append(metric)
            
            # Log performance warning if block is slow
            if execution_time > 2.0:  # More than 2 seconds
                logger.warning(f"Slow code block detected: {block_name} took {execution_time:.2f}s")
            
            tracemalloc.stop()
    
    def start_profiling(self, profile_name: str = "default") -> cProfile.Profile:
        """Start detailed profiling."""
        profile = cProfile.Profile()
        profile.enable()
        self._active_profiles[profile_name] = profile
        logger.info(f"Started profiling session: {profile_name}")
        return profile
    
    def stop_profiling(self, profile_name: str = "default") -> ProfilingResult:
        """Stop detailed profiling and return results."""
        if profile_name not in self._active_profiles:
            raise ValueError(f"No active profiling session named {profile_name}")
        
        profile = self._active_profiles[profile_name]
        profile.disable()
        
        # Get profiling results
        s = io.StringIO()
        ps = pstats.Stats(profile, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Parse the results
        result_lines = s.getvalue().split('\n')
        total_time = 0.0
        calls = 0
        cumulative_time = 0.0
        callers = []
        
        for line in result_lines:
            if line.strip() and not line.startswith('   ncalls'):
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        calls += int(parts[0].replace(',', ''))
                        total_time += float(parts[1])
                        cumulative_time += float(parts[3])
                        # Extract function name (this is simplified)
                        func_name = ' '.join(parts[5:])
                        callers.append(func_name)
                    except (ValueError, IndexError):
                        continue
        
        profiling_result = ProfilingResult(
            function_name=profile_name,
            total_time=total_time,
            calls=calls,
            avg_time_per_call=total_time / calls if calls > 0 else 0,
            cumulative_time=cumulative_time,
            callers=callers[:10]  # Top 10 callers
        )
        
        self.profiling_results[profile_name] = profiling_result
        del self._active_profiles[profile_name]
        
        logger.info(f"Stopped profiling session: {profile_name}")
        return profiling_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics:
            return {"message": "No performance metrics collected"}
        
        with self._lock:
            # Group metrics by function
            function_metrics = {}
            for metric in self.metrics:
                if metric.function_name not in function_metrics:
                    function_metrics[metric.function_name] = []
                function_metrics[metric.function_name].append(metric)
            
            # Calculate statistics
            summary = {}
            for func_name, metrics in function_metrics.items():
                execution_times = [m.execution_time for m in metrics]
                cpu_usages = [m.cpu_percent for m in metrics]
                memory_usages = [m.memory_usage for m in metrics]
                
                summary[func_name] = {
                    'call_count': len(metrics),
                    'total_time': sum(execution_times),
                    'avg_time': sum(execution_times) / len(execution_times),
                    'max_time': max(execution_times),
                    'min_time': min(execution_times),
                    'avg_cpu_percent': sum(cpu_usages) / len(cpu_usages),
                    'avg_memory_mb': sum(memory_usages) / len(memory_usages),
                    'slow_calls': len([t for t in execution_times if t > 1.0])
                }
            
            return summary
    
    def get_slow_functions(self, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get functions that exceed the time threshold."""
        slow_functions = []
        
        with self._lock:
            for metric in self.metrics:
                if metric.execution_time > threshold:
                    slow_functions.append({
                        'function_name': metric.function_name,
                        'execution_time': metric.execution_time,
                        'cpu_percent': metric.cpu_percent,
                        'memory_usage': metric.memory_usage,
                        'timestamp': metric.timestamp.isoformat()
                    })
        
        # Sort by execution time
        slow_functions.sort(key=lambda x: x['execution_time'], reverse=True)
        return slow_functions
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self.metrics.clear()
            self.profiling_results.clear()
            self._active_profiles.clear()
        logger.info("Performance metrics reset")

class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory."""
        collected = gc.collect()
        logger.info(f"Garbage collection: {collected} objects collected")
        return collected
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def optimize_dataframe_memory(df) -> Any:
        """Optimize pandas DataFrame memory usage."""
        if hasattr(df, 'memory_usage'):
            start_mem = df.memory_usage(deep=True).sum() / 1024**2
            logger.info(f'Memory usage of dataframe is {start_mem:.2f} MB')
            
            for col in df.columns:
                col_type = df[col].dtype
                
                if col_type != object:
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if str(col_type)[:3] == 'int':
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)
                    else:
                        if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                            df[col] = df[col].astype(np.float16)
                        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                        else:
                            df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype('category')
            
            end_mem = df.memory_usage(deep=True).sum() / 1024**2
            logger.info(f'Memory usage after optimization is {end_mem:.2f} MB')
            logger.info(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
            
            return df
        else:
            logger.warning("Object is not a DataFrame, cannot optimize memory")
            return df

class PerformanceOptimizer:
    """Main performance optimization class."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.config = get_config()
        
    def optimize_application(self) -> Dict[str, Any]:
        """Run comprehensive optimization on the application."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'memory_optimization': {},
            'garbage_collection': {},
            'performance_summary': {}
        }
        
        # Optimize memory
        memory_before = self.memory_optimizer.get_memory_usage()
        gc_count = self.memory_optimizer.force_garbage_collection()
        memory_after = self.memory_optimizer.get_memory_usage()
        
        results['memory_optimization'] = {
            'before': memory_before,
            'after': memory_after,
            'difference': {
                'rss_mb': memory_after['rss_mb'] - memory_before['rss_mb'],
                'percent': memory_after['percent'] - memory_before['percent']
            }
        }
        
        results['garbage_collection'] = {
            'objects_collected': gc_count
        }
        
        # Get performance summary
        results['performance_summary'] = self.profiler.get_performance_summary()
        
        logger.info("Application optimization completed")
        return results
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for performance optimization."""
        recommendations = []
        
        # Check for slow functions
        slow_functions = self.profiler.get_slow_functions(threshold=0.5)
        if slow_functions:
            recommendations.append(f"Found {len(slow_functions)} slow functions (>0.5s)")
            for func in slow_functions[:5]:  # Top 5
                recommendations.append(f"  - {func['function_name']}: {func['execution_time']:.2f}s")
        
        # Check memory usage
        memory_usage = self.memory_optimizer.get_memory_usage()
        if memory_usage['percent'] > 80:
            recommendations.append(f"High memory usage: {memory_usage['percent']:.1f}%")
            recommendations.append("Consider optimizing data structures or running garbage collection")
        
        # Check for frequent calls to the same functions
        summary = self.profiler.get_performance_summary()
        if isinstance(summary, dict):
            for func_name, stats in summary.items():
                if isinstance(stats, dict) and 'call_count' in stats:
                    if stats['call_count'] > 100:
                        recommendations.append(f"Function {func_name} called {stats['call_count']} times")
                        recommendations.append("Consider caching or reducing call frequency")
        
        if not recommendations:
            recommendations.append("No immediate optimization issues detected")
            
        return recommendations

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Async version of the profiler
class AsyncPerformanceProfiler:
    """Async version of the performance profiler."""
    
    def __init__(self):
        self.sync_profiler = PerformanceProfiler()
    
    def profile_async_function(self, func: Callable) -> Callable:
        """Decorator to profile an async function."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Start memory tracking
            tracemalloc.start()
            snapshot1 = tracemalloc.take_snapshot()
            
            # Start CPU monitoring
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Start timing
            start_time = time.perf_counter()
            
            try:
                # Execute async function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Stop timing
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Stop memory tracking
                snapshot2 = tracemalloc.take_snapshot()
                top_stats = snapshot2.compare_to(snapshot1, 'lineno')
                
                memory_allocated = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
                memory_peak = sum(stat.size_diff for stat in top_stats) / 1024 / 1024  # MB
                
                # Get CPU usage
                cpu_after = process.cpu_percent()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                # Store metrics
                metric = PerformanceMetrics(
                    function_name=func.__name__,
                    execution_time=execution_time,
                    cpu_percent=cpu_after - cpu_before,
                    memory_usage=memory_after - memory_before,
                    call_count=1,
                    additional_info={
                        'memory_allocated_mb': memory_allocated,
                        'memory_peak_mb': memory_peak
                    }
                )
                self.sync_profiler.metrics.append(metric)
                
                # Log performance warning if function is slow
                if execution_time > 1.0:  # More than 1 second
                    logger.warning(f"Slow async function detected: {func.__name__} took {execution_time:.2f}s")
                
                tracemalloc.stop()
        
        return wrapper

# Example usage and testing function
def run_performance_demo():
    """Run a demonstration of the performance profiling."""
    print("Enhanced Performance Profiling and Optimization Module")
    print("=" * 50)
    
    # Create optimizer instance
    optimizer = performance_optimizer
    profiler = optimizer.profiler
    
    # Test profiling a function
    @profiler.profile_function
    def sample_function():
        """Sample function for testing."""
        time.sleep(0.1)  # Simulate work
        return "completed"
    
    print("Testing function profiling...")
    
    # Run the function multiple times
    for i in range(5):
        result = sample_function()
        print(f"  Run {i+1}: {result}")
    
    # Get performance summary
    summary = profiler.get_performance_summary()
    print(f"\nPerformance Summary:")
    for func_name, stats in summary.items():
        print(f"  {func_name}:")
        print(f"    Calls: {stats['call_count']}")
        print(f"    Avg Time: {stats['avg_time']:.3f}s")
        print(f"    Total Time: {stats['total_time']:.3f}s")
    
    # Test memory optimization
    print("\nTesting memory optimization...")
    memory_before = optimizer.memory_optimizer.get_memory_usage()
    print(f"Memory before optimization: {memory_before['rss_mb']:.2f} MB")
    
    # Force garbage collection
    collected = optimizer.memory_optimizer.force_garbage_collection()
    print(f"Objects collected: {collected}")
    
    memory_after = optimizer.memory_optimizer.get_memory_usage()
    print(f"Memory after optimization: {memory_after['rss_mb']:.2f} MB")
    print(f"Memory reduction: {memory_before['rss_mb'] - memory_after['rss_mb']:.2f} MB")
    
    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print(f"\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    print("\nPerformance profiling module initialized successfully")

if __name__ == "__main__":
    run_performance_demo()
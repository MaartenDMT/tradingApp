"""
Parallel processing utilities for the Trading Application.

This module provides functions for parallel processing to improve performance.
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable, Any, List, Optional

def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run a function in a separate thread.
    
    Args:
        func (Callable): The function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: The result of the function
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result()

def run_in_process(func: Callable, *args, **kwargs) -> Any:
    """
    Run a function in a separate process.
    
    Args:
        func (Callable): The function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: The result of the function
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result()

def parallel_map(func: Callable, items: List[Any], max_workers: Optional[int] = None) -> List[Any]:
    """
    Apply a function to a list of items in parallel.
    
    Args:
        func (Callable): The function to apply
        items (List[Any]): The items to process
        max_workers (Optional[int]): Maximum number of worker threads/processes
        
    Returns:
        List[Any]: The results of applying the function to each item
    """
    if max_workers is None:
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

def parallel_process_map(func: Callable, items: List[Any], max_workers: Optional[int] = None) -> List[Any]:
    """
    Apply a function to a list of items in parallel using processes.
    
    Args:
        func (Callable): The function to apply
        items (List[Any]): The items to process
        max_workers (Optional[int]): Maximum number of worker threads/processes
        
    Returns:
        List[Any]: The results of applying the function to each item
    """
    if max_workers is None:
        max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

class ParallelProcessor:
    """
    A context manager for parallel processing with automatic resource management.
    """
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.executor = None
    
    def __enter__(self):
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self.executor
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.executor:
            self.executor.shutdown(wait=True)
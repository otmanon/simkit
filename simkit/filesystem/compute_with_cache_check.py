import numpy as np
import os

def compute_with_cache_check(func, cache_path, read_cache=True):
    """Compute a function's result with optional caching.
    
    This function either loads previously computed results from a cache file
    or computes them from scratch and saves them to the cache.
    
    Args:
        func (callable): Function to compute results from
        cache_path (str): Path to the cache file (.npz format)
        read_cache (bool, optional): Whether to attempt reading from cache. Defaults to True.
        
    Returns:
        tuple: The computed results. If func returns a single value, it will be wrapped in a tuple.
    """
    if read_cache:
        try:  
            data = np.load(cache_path, allow_pickle=True)
            output2 = ()
            for k in sorted(data.files, key=lambda s: int(s[1:])):
                try:
                    # print(data[k].shape)
                    output2 += (data[k].item(), )
                except:
                    output2 += (data[k], )
            output = output2
        except:  
                print("Could not read from cache. Recomputing from scratch...")  
                output = func()
                if not isinstance(output, tuple):
                    output = (output,)
                np.savez(cache_path, **{f'v{i}': arg for i, arg in enumerate(output)})
    else:
        print("Will not read from cache. Recomputing from scratch...")  
        output = func()
        if not isinstance(output, tuple):
            output = (output,)
        np.savez(cache_path, **{f'v{i}': arg for i, arg in enumerate(output)})
        
    if not isinstance(output, tuple):
        output = (output,)
    return output 
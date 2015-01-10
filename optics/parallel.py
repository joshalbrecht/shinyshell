
"""
Code related to parallelizing computations, using threads and processes, etc
"""

import os
import sys
import traceback

from optics.base import *

ERROR_SUFFIX = '.error'

def check_for_errors(prefix, error_directory='.'):
    for f in os.listdir(error_directory):
        abs_file = os.path.join(error_directory, f)
        if os.path.isfile(abs_file) and str(f).endswith(ERROR_SUFFIX):
            with open(abs_file, 'rb') as infile:
                print infile.read()
            os.remove(abs_file)

def remote_runner(func, fargs=None, fkwargs=None, prefix=None):
    try:
        if fargs == None:
            fargs = ()
        if fkwargs == None:
            fkwargs = {}
        value = func(*fargs, **fkwargs)
        return value
    except Exception, e:
        output = str(e) + '\n' + ''.join(traceback.format_exception(*sys.exc_info()))
        with open(prefix + ERROR_SUFFIX, 'wb') as out_file:
            out_file.write(output)
        raise Exception("Remote error")
    
def call_via_pool(pool, *args, **kwargs):
    """
    Synchronously call in process pool. Meant to be called from a thread.
    Basically just checks that errors dont happen in that process
    """
    prefix = random_string()
    result = pool.apply_async(remote_runner, args=[remote_runner], kwds={'fargs': args, 'fkwargs': kwargs, 'prefix': prefix})
    result.wait()
    check_for_errors(prefix)
    return result.get()
    

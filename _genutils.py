"""Module geomodeloutputs: easily use files that are geoscience model outputs.

Copyright (2024-now) Institut des Géosciences de l'Environnement (IGE), France.

This software is released under the terms of the BSD 3-clause license:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    (1) Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    (2) Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    (3) The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.

"""

import functools

def keyify_arg(arg):
    """Return key representing given argument."""
    if isinstance(arg, str):
        return arg
    else:
        raise TypeError("I cannot keyify argument of type %s." % type(arg))

def keyify_args(*args, **kwargs):
    """Return a single key representing all given arguments."""
    return (tuple(keyify_arg(a) for a in args),
            tuple((k, keyify_arg(v)) for k,v in kwargs.items()))

def method_cacher(method):
    """Decorator that adds a cache functionality to class methods.

    Important: this decorator relies on the fact that the target class instance
    has an attribute named _cache that is a dictionary dedicated to this cache
    functionality.

    """
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        key = (method.__name__, keyify_args(*args[1:], **kwargs))
        try:
            answer = args[0]._cache[key]
        except KeyError:
            answer = args[0]._cache[key] = method(*args, **kwargs)
        return answer
    return wrapper

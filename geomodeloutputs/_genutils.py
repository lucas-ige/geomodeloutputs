# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: generic utilities."""

import functools
from datetime import datetime
import pyproj
import numpy as np
import xarray as xr
from .dateutils import (
    datetime_plus_nmonths,
    CF_CALENDARTYPE_DEFAULT,
    CF_CALENDARTYPE_360DAYS,
)


def keyify_arg(arg):
    """Return unique key representing given argument."""
    if isinstance(arg, str):
        return arg
    else:
        raise TypeError("I cannot keyify argument of type %s." % type(arg))


def keyify_args(*args, **kwargs):
    """Return a unique key representing all given arguments."""
    return (
        tuple(keyify_arg(a) for a in args),
        tuple((k, keyify_arg(v)) for k, v in kwargs.items()),
    )


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


def unique_guess_in_iterable(guesses, iterable):
    """Return unique guess that is found in iterable, error otherwise."""
    found = [guess in iterable for guess in guesses]
    if sum(found) != 1:
        raise ValueError("Zero or more than one guess(es) is in iterable.")
    return guesses[found.index(True)]


def preprocess_dataset(ds):
    """Preprocessing function to open non CF-compliant datasets.

    This function exists to handle NetCDF files that use "months since..." time
    units but a calendar that is not a 360-day calendar.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset opened with vanilla xarray.open_dataset.

    Returns
    -------
    xarray.Dataset
        The processed dataset.

    """
    units = ds["time"].attrs["units"]
    if units.startswith("MONTHS since "):
        f = "%Y-%m-%d %H:%M:%S"
        if len(units) == 18 and units.endswith(":0"):
            units += "0"
        start = datetime.strptime(units[13:], f)
        try:
            calendar = ds["time"].attrs["calendar"]
        except KeyError:
            calendar = CF_CALENDARTYPE_DEFAULT
        if calendar in CF_CALENDARTYPE_360DAYS:
            raise ValueError(
                'This function is meant to deal with "months since" time data '
                'with calendars other than "360-day calendars."'
            )
        convert = lambda t: datetime_plus_nmonths(start, t, calendar)
        convert_all = np.vectorize(convert)
        out = ds.assign_coords(time=convert_all(ds["time"].values))
        return out
    else:
        return ds


def open_dataset(filepath, **kwargs):
    """Open dataset.

    This function acts as xarray.open_dataset, except that it can handle files
    that use "months since..." time units but a calendar that is not a 360-day
    calendar.

    Parameters
    ----------
    filepath : str
        The location of the file on disk.
    **kwargs
        These are passed "as is" to xarray.open_dataset.

    Returns
    -------
    xarray.Dataset
        The opened dataset.

    """
    return preprocess_dataset(xr.open_dataset(filepath, **kwargs))


def open_mfdataset(filepath, **kwargs):
    """Open multiple-file dataset.

    This function acts as xarray.open_mfdataset, except that it can handle
    files that use "months since..." time units but a calendar that is not a
    360-day calendar.

    Parameters
    ----------
    filepath : str
        The location of the file(s) on disk. It can be any pattern accepted by
        xarray.open_mfdataset.
    **kwargs
        These are passed "as is" to xarray.open_dataset, with one exception:
        named argument "preprocess" is not allowed here.

    Returns
    -------
    xarray.Dataset
        The opened dataset.

    Raises
    ------
    ValueError
        If "preprocess" is present as a named argument.

    """
    if "preprocess" in kwargs:
        raise ValueError(
            "This wrapper around xarray.open_mfdataset does not accept "
            '"preprocess" as a keyword argument.'
        )
    return xr.open_mfdataset(filepath, preprocess=preprocess_dataset, **kwargs)

# Copyright (c) 2024-2025, Institut des Géosciences de l'Environnement, France.
#
# License: BSD 3-clause "new" or "revised" license (BSD-3-Clause).

"""Module geomodeloutputs: date- and time-related utilities."""

from datetime import datetime, timedelta

# CF convention constants
CF_CALENDARTYPE_GREGORIAN = ("gregorian", "standard")
CF_CALENDARTYPE_NOLEAP = ("noleap", "365_day")
CF_CALENDARTYPE_ALLLEAP = ("all_leap", "366_day")
CF_CALENDARTYPE_360DAYS = ("360_day",)
CF_CALENDARTYPE_ALL = (
    CF_CALENDARTYPE_GREGORIAN
    + CF_CALENDARTYPE_NOLEAP
    + CF_CALENDARTYPE_ALLLEAP
    + CF_CALENDARTYPE_360DAYS
)
CF_CALENDARTYPE_DEFAULT = "gregorian"


def ndays_in_year(year, calendar=CF_CALENDARTYPE_DEFAULT):
    """Return the number of days in given year according to given calendar.

    Parameters
    ----------
    year : int
        The year of interest.
    calendar : str
        The type of calendar as a CF-compliant calendar name.

    Returns
    -------
    int
        The number of days in given year and for given calendar.

    """
    if calendar in CF_CALENDARTYPE_NOLEAP:
        return 365
    elif calendar in CF_CALENDARTYPE_ALLLEAP:
        return 366
    elif calendar in CF_CALENDARTYPE_360DAYS:
        return 360
    elif calendar in CF_CALENDARTYPE_GREGORIAN:
        return int(datetime(year, 12, 31).strftime("%j"))
    raise ValueError("Invalid calendar value: %s." % calendar)


def ndays_in_month(year, month, calendar=CF_CALENDARTYPE_DEFAULT):
    """Return the number of days in given month according to given calendar.

    Parameters
    ----------
    year : int
        The year of interest.
    month : int
        The month of interest (1, 2, ..., 12).
    calendar : str
        The type of calendar as a CF-compliant calendar name.

    Returns
    -------
    int
        The number of days in given month and year and for given calendar.

    """
    if calendar not in CF_CALENDARTYPE_ALL:
        raise ValueError("Invalid calendar value: %s." % calendar)
    if calendar in CF_CALENDARTYPE_360DAYS:
        return 30
    elif month == 2 and calendar in CF_CALENDARTYPE_NOLEAP:
        return 28
    elif month == 2 and calendar in CF_CALENDARTYPE_ALLLEAP:
        return 29
    else:
        month += 1
        if month == 13:
            year += 1
            month = 1
        return (datetime(year, month, 1) - timedelta(days=1)).day


def datetime_plus_nmonths(start, nmonths, calendar=CF_CALENDARTYPE_DEFAULT):
    """Return (start + nmonths).

    Adding a number of months to a date is ill-defined for calendars other than
    360-day, because the number of days in a month varies throughout the year.
    This function does what it can to calculate a reasonable value for all
    calendar types.

    Parameters
    ----------
    start : datetime
        The start date.
    nmonths : int
        The number of months to add to start.
    calendar : str
        The type of calendar as a CF-compliant calendar name.

    Returns
    -------
    datetime
        The value of start + nmonths.

    """
    if calendar not in CF_CALENDARTYPE_GREGORIAN:
        raise NotImplementedError("This calendar is not yet supported.")
    n, r = divmod(nmonths, 1)
    sign = [1, -1][n < 0]
    if n < 0 and r != 0:
        n += 1
        r -= 1
    year, month = start.year, start.month
    for i in range(abs(int(n))):
        month += sign
        if month == 13:
            year += 1
            month = 1
        elif month == 0:
            year -= 1
            month = 12
    yearmonth = ("%4d-%2d" % (year, month)).replace(" ", "0")
    dhsm = start.strftime("-%d %H:%M:%S")
    out = datetime.strptime(yearmonth + dhsm, "%Y-%m-%d %H:%M:%S")
    oneday = timedelta(days=1)
    return out + r * ndays_in_month(year, month, calendar=calendar) * oneday

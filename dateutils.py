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

from datetime import datetime, timedelta

# CF convention constants
CF_CALENDARTYPE_GREGORIAN = ("gregorian", "standard")
CF_CALENDARTYPE_NOLEAP = ("noleap", "365_day")
CF_CALENDARTYPE_ALLLEAP = ("all_leap", "366_day")
CF_CALENDARTYPE_360DAYS = ("360_day",)
CF_CALENDARTYPE_ALL = (CF_CALENDARTYPE_GREGORIAN +
                       CF_CALENDARTYPE_NOLEAP +
                       CF_CALENDARTYPE_ALLLEAP +
                       CF_CALENDARTYPE_360DAYS)
CF_CALENDARTYPE_DEFAULT = "gregorian"

def ndays_in_year(year: int, calendar: str = CF_CALENDARTYPE_DEFAULT) -> int:
    """Return number of days in given year according to given calendar."""
    if calendar in CF_CALENDARTYPE_NOLEAP:
        return 365
    elif calendar in CF_CALENDARTYPE_ALLLEAP:
        return 366
    elif calendar in CF_CALENDARTYPE_360DAYS:
        return 360
    elif calendar in CF_CALENDARTYPE_GREGORIAN:
        return int(datetime(year, 12, 31).strftime("%j"))
    raise ValueError("Invalid calendar value: %s." % calendar)

def ndays_in_month(
        year: int,
        month: int,
        calendar: str = CF_CALENDARTYPE_DEFAULT
) -> int:
    """Return number of days in given month according to given calendar."""
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

def datetime_plus_nmonths(
        start: datetime,
        nmonths: int,
        calendar: str = CF_CALENDARTYPE_DEFAULT
) -> datetime:
    """Return (start+nmonths)."""
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
    return out + r*ndays_in_month(year, month, calendar=calendar)*oneday

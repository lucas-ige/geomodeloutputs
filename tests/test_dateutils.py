# Copyright (2025-now) Institut des Géosciences de l'Environnement, France.
#
# This software is released under the terms of the BSD 3-clause license:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#     (3) The name of the author may not be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Module geomodeloutputs: tests for dateutils submodule."""

from datetime import datetime
from geomodeloutputs.dateutils import (
    ndays_in_year,
    ndays_in_month,
    datetime_plus_nmonths,
)

def test_ndays_in_year():
    # Non-leap year (test all calendars)
    year = 2023
    assert ndays_in_year(year) == 365
    assert ndays_in_year(year, "standard") == 365
    assert ndays_in_year(year, "gregorian") == 365
    assert ndays_in_year(year, "noleap") == 365
    assert ndays_in_year(year, "365_day") == 365
    assert ndays_in_year(year, "all_leap") == 366
    assert ndays_in_year(year, "366_day") == 366
    assert ndays_in_year(year, "360_day") == 360
    # Leap year (test all calendars)
    year = 2024
    assert ndays_in_year(year) == 366
    assert ndays_in_year(year, "standard") == 366
    assert ndays_in_year(year, "gregorian") == 366
    assert ndays_in_year(year, "noleap") == 365
    assert ndays_in_year(year, "365_day") == 365
    assert ndays_in_year(year, "all_leap") == 366
    assert ndays_in_year(year, "366_day") == 366
    assert ndays_in_year(year, "360_day") == 360

def test_ndays_in_month():
    # Non-leap year, July (test all calendars)
    year, month = 2023, 7
    assert ndays_in_month(year, month) == 31
    assert ndays_in_month(year, month, "standard") == 31
    assert ndays_in_month(year, month, "gregorian") == 31
    assert ndays_in_month(year, month, "noleap") == 31
    assert ndays_in_month(year, month, "365_day") == 31
    assert ndays_in_month(year, month, "all_leap") == 31
    assert ndays_in_month(year, month, "366_day") == 31
    assert ndays_in_month(year, month, "360_day") == 30
    # Non-leap year, February (test all calendars)
    year, month = 2023, 2
    assert ndays_in_month(year, month) == 28
    assert ndays_in_month(year, month, "standard") == 28
    assert ndays_in_month(year, month, "gregorian") == 28
    assert ndays_in_month(year, month, "noleap") == 28
    assert ndays_in_month(year, month, "365_day") == 28
    assert ndays_in_month(year, month, "all_leap") == 29
    assert ndays_in_month(year, month, "366_day") == 29
    assert ndays_in_month(year, month, "360_day") == 30
    # Leap year, July (test all calendars)
    year, month = 2024, 7
    assert ndays_in_month(year, month) == 31
    assert ndays_in_month(year, month, "standard") == 31
    assert ndays_in_month(year, month, "gregorian") == 31
    assert ndays_in_month(year, month, "noleap") == 31
    assert ndays_in_month(year, month, "365_day") == 31
    assert ndays_in_month(year, month, "all_leap") == 31
    assert ndays_in_month(year, month, "366_day") == 31
    assert ndays_in_month(year, month, "360_day") == 30
    # Leap year, February (test all calendars)
    year, month = 2024, 2
    assert ndays_in_month(year, month) == 29
    assert ndays_in_month(year, month, "standard") == 29
    assert ndays_in_month(year, month, "gregorian") == 29
    assert ndays_in_month(year, month, "noleap") == 28
    assert ndays_in_month(year, month, "365_day") == 28
    assert ndays_in_month(year, month, "all_leap") == 29
    assert ndays_in_month(year, month, "366_day") == 29
    assert ndays_in_month(year, month, "360_day") == 30

def test_datetime_plus_nmonths():
    start = datetime(2025, 1, 15)
    assert datetime_plus_nmonths(start, 0) == datetime(2025, 1, 15)
    assert datetime_plus_nmonths(start, 1) == datetime(2025, 2, 15)
    assert datetime_plus_nmonths(start, -1) == datetime(2024, 12, 15)
    assert datetime_plus_nmonths(start, 12) == datetime(2026, 1, 15)
    assert datetime_plus_nmonths(start, -12) == datetime(2024, 1, 15)
    start = datetime(2025, 6, 1)
    assert datetime_plus_nmonths(start, 0.5) == datetime(2025, 6, 16)
    assert datetime_plus_nmonths(start, -0.5) == datetime(2025, 5, 17)
    start = datetime(2025, 2, 1)
    assert datetime_plus_nmonths(start, 0.5) == datetime(2025, 2, 15)
    assert datetime_plus_nmonths(start, -0.5) == datetime(2025, 1, 18)
    start = datetime(2024, 2, 1)
    assert datetime_plus_nmonths(start, 0.5) == datetime(2024, 2, 15, 12)
    assert datetime_plus_nmonths(start, -0.5) == datetime(2024, 1, 17, 12)

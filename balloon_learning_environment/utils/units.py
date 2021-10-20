# coding=utf-8
# Copyright 2021 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common unit conversion functions and classes."""

import datetime as dt
import typing

import numpy as np

_METERS_PER_FOOT = 0.3048


class Distance:
  """A compact distance unit."""

  def __init__(self,
               *,
               m: float = 0.0,
               meters: float = 0.0,
               km: float = 0.0,
               kilometers: float = 0.0,
               feet: float = 0.0):
    # Note: distance is stored as meters.
    self._distance = (
        m + meters + (km + kilometers) * 1000.0 + feet * _METERS_PER_FOOT)

  @property
  def m(self) -> float:
    """Gets distance in meters."""
    return self._distance

  @property
  def meters(self) -> float:
    """Gets distance in meters."""
    return self.m

  @property
  def km(self) -> float:
    """Gets distance in kilometers."""
    return self._distance / 1000.0

  @property
  def kilometers(self) -> float:
    """Gets distance in kilometers."""
    return self.km

  @property
  def feet(self) -> float:
    return self._distance / _METERS_PER_FOOT

  def __add__(self, other: 'Distance') -> 'Distance':
    if isinstance(other, Distance):
      return Distance(m=self.m + other.m)
    else:
      raise NotImplementedError(f'Cannot add Distance and {type(other)}')

  def __sub__(self, other: 'Distance') -> 'Distance':
    if isinstance(other, Distance):
      return Distance(m=self.m - other.m)
    else:
      raise NotImplementedError(f'Cannot subtract Distance and {type(other)}')

  @typing.overload
  def __truediv__(self, other: float) -> 'Distance':
    ...

  @typing.overload
  def __truediv__(self, other: dt.timedelta) -> 'Velocity':
    # velocity = change in distance / change in time.
    ...

  @typing.overload
  def __truediv__(self, other: 'Distance') -> float:
    ...

  def __truediv__(self, other):
    if isinstance(other, (int, float)):
      return Distance(m=self.m / other)
    elif isinstance(other, dt.timedelta):
      return Velocity(mps=self.m / other.total_seconds())
    elif isinstance(other, Distance):
      return self.m / other.m
    else:
      raise NotImplementedError(f'Cannot divide distance by {type(other)}')

  def __mul__(self, other: float) -> 'Distance':
    if isinstance(other, (int, float)):
      return Distance(m=self.m * other)
    else:
      raise NotImplementedError(f'Cannot multiply Distance and {type(other)}')

  def __rmul__(self, other: float) -> 'Distance':
    return self.__mul__(other)

  def __eq__(self, other: 'Distance') -> bool:
    return abs(self.m - other.m) < 1e-9

  def __neq__(self, other: 'Distance') -> bool:
    return not self.__eq__(other)

  def __lt__(self, other: 'Distance') -> bool:
    return self.m < other.m

  def __le__(self, other: 'Distance') -> bool:
    return self.m <= other.m

  def __gt__(self, other: 'Distance') -> bool:
    return self.m > other.m

  def __ge__(self, other: 'Distance') -> bool:
    return self.m >= other.m


class Velocity:
  """A compact velocity unit."""

  def __init__(self,
               *,
               mps: float = 0.0,
               meters_per_second: float = 0.0,
               kmph: float = 0.0,
               kilometers_per_hour: float = 0.0):
    # Note: distance is stored as meters per second.
    self._velocity = (
        mps + meters_per_second + (kmph + kilometers_per_hour) * 1000 / 3600)

  @property
  def mps(self) -> float:
    """Gets velocity in meters per second."""
    return self._velocity

  @property
  def meters_per_second(self) -> float:
    """Gets velocity in meters per second."""
    return self.mps

  @property
  def kmph(self) -> float:
    """Gets velocity in kilometers per hour."""
    return self._velocity * 3600 / 1000

  @property
  def kilometers_per_hour(self) -> float:
    """Gets velocity in kilometers per hour."""
    return self.kmph

  def __add__(self, other: 'Velocity') -> 'Velocity':
    if isinstance(other, Velocity):
      return Velocity(mps=self.mps + other.mps)
    else:
      raise NotImplementedError(f'Cannot add Velocity and {type(other)}')

  def __sub__(self, other: 'Velocity') -> 'Velocity':
    if isinstance(other, Velocity):
      return Velocity(mps=self.mps - other.mps)
    else:
      raise NotImplementedError(f'Cannot subtract Velocity and {type(other)}')

  def __mul__(self, other: dt.timedelta) -> Distance:
    if isinstance(other, dt.timedelta):
      # distance = velocity * time (for constant velocity).
      return Distance(m=self.mps * other.total_seconds())
    else:
      raise NotImplementedError(f'Cannot multiply velocity with {type(other)}')

  def __rmul__(self, other: dt.timedelta) -> Distance:
    return self.__mul__(other)

  def __eq__(self, other: 'Velocity') -> bool:
    if isinstance(other, Velocity):
      # Note: we consider very similar velocities to be equal.
      return abs(self.mps - other.mps) < 1e-9
    else:
      raise ValueError(f'Cannot compare velocity and {type(other)}')

  def __str__(self) -> str:
    return f'{self.mps} m/s'


class Energy(object):
  """A compact energy class."""

  def __init__(self, *, watt_hours: float = 0.0):
    self._wh = watt_hours

  @property
  def watt_hours(self) -> float:
    return self._wh

  def __add__(self, other: 'Energy') -> 'Energy':
    if isinstance(other, Energy):
      return Energy(watt_hours=self.watt_hours + other.watt_hours)
    else:
      raise NotImplementedError(f'Cannot add Energy and {type(other)}')

  def __sub__(self, other: 'Energy') -> 'Energy':
    if isinstance(other, Energy):
      return Energy(watt_hours=self.watt_hours - other.watt_hours)
    else:
      raise NotImplementedError(f'Cannot subtract Energy and {type(other)}')

  def __truediv__(self, other: 'Energy') -> float:
    if isinstance(other, Energy):
      return self.watt_hours / other.watt_hours
    else:
      raise NotImplementedError(f'Cannot divide Energy and {type(other)}')

  def __mul__(self, other: float) -> 'Energy':
    if isinstance(other, (int, float)):
      return Energy(watt_hours=self.watt_hours * other)
    else:
      raise NotImplementedError(f'Cannot multiply Energy and {type(other)}')

  def __rmul__(self, other: float) -> 'Energy':
    return self.__mul__(other)

  def __gt__(self, other: 'Energy') -> bool:
    if isinstance(other, Energy):
      return self.watt_hours > other.watt_hours
    else:
      raise ValueError(f'Cannot compare Energy and {type(other)}')

  def __eq__(self, other: 'Energy') -> bool:
    if isinstance(other, Energy):
      return self.watt_hours == other.watt_hours
    else:
      raise ValueError(f'Cannot compare Energy and {type(other)}')

  def __ge__(self, other: 'Energy') -> bool:
    if isinstance(other, Energy):
      return self.watt_hours >= other.watt_hours
    else:
      return ValueError(f'Cannot compare Energy and {type(other)}')


class Power(object):
  """A compact power class."""

  def __init__(self, *, watts: float = 0.0):
    self._w = watts

  @property
  def watts(self) -> float:
    return self._w

  def __add__(self, other: 'Power') -> 'Power':
    if isinstance(other, Power):
      return Power(watts=self.watts + other.watts)
    else:
      raise NotImplementedError(f'Cannot add Power and {type(other)}')

  def __sub__(self, other: 'Power') -> 'Power':
    if isinstance(other, Power):
      return Power(watts=self.watts - other.watts)
    else:
      raise NotImplementedError(f'Cannot subtract Power and {type(other)}')

  def __mul__(self, other: dt.timedelta) -> Energy:
    if isinstance(other, dt.timedelta):
      return Energy(watt_hours=self.watts * timedelta_to_hours(other))
    else:
      raise NotImplementedError(f'Cannot multiply Power with {type(other)}')

  def __rmul__(self, other: dt.timedelta) -> Energy:
    return self.__mul__(other)

  def __gt__(self, other: 'Power') -> bool:
    if isinstance(other, Power):
      return self.watts > other.watts
    else:
      raise ValueError(f'Cannot compare Power and {type(other)}')

  def __eq__(self, other: 'Power') -> bool:
    if isinstance(other, Power):
      return self.watts == other.watts
    else:
      raise ValueError(f'Cannot compare Power and {type(other)}')


def distance_to_degrees(d: Distance) -> float:
  """Converts a distance to degrees. Only (sort of) valid near the equator."""

  # NOTE(scandido): 111 kilometers is about 1 degree latitude (anywhere) but
  # this only holds for longitude at the equator.
  return d.km / 111.0


def relative_distance(x: Distance, y: Distance) -> Distance:
  # Assumes x, y are relative to the target, so the distance is simply the norm.
  return Distance(m=np.linalg.norm([x.m, y.m], ord=2).item())


def seconds_to_hours(s: float) -> float:
  return s / 3600.0


def timedelta_to_hours(d: dt.timedelta) -> float:
  return seconds_to_hours(d.total_seconds())


def datetime(year: int,
             month: int,
             day: int,
             hour: int = 0,
             minute: int = 0,
             second: int = 0,
             microsecond: int = 0,
             tzinfo: dt.tzinfo = dt.timezone.utc,
             *,
             fold: int = 0) -> dt.datetime:
  """Creates a datetime with a default timezone of UTC.

  By default, a datetime uses "naive time", which is timezone-free.
  However, for the purposes of this simulation, this can cause errors
  when performing operations that rely on a UNIX timestamp (e.g. solar
  calculations). Therefore, this is the preferred way of constructing
  datetime objects within the codebase.

  Args:
    year: The year.
    month: The month (between 1 and 12 inclusive).
    day: The day (must be a valid day for a given month and year).
    hour: The hour (between 0 and 23 inclusive).
    minute: The minute (between 0 and 59 inclusive).
    second: The second (between 0 and 59 inclusive).
    microsecond: The microsecond (between 0 and 999_999).
    tzinfo: A timezone object. If None, it will default to UTC.
    fold: A value (0 or 1) to disambiguate duplicate times caused by time
      changes. For example, if the clocks go back an hour at 01:00 on a
      specific day, the time 00:30 will occur twice. If fold is 0, then the
      datetime referred to is the first occurance of the time. If it is 1,
      it is the second occurance of the time.

  Returns:
    A timezone object.
  """
  # pylint: disable=g-tzinfo-datetime
  return dt.datetime(
      year, month, day, hour, minute, second, microsecond, tzinfo, fold=fold)
  # pylint: enable=g-tzinfo-datetime


def datetime_from_timestamp(timestamp: int) -> dt.datetime:
  """Converts a given UTC timestamp into a datetime.

  The returned datetime includes timezone information.

  Args:
    timestamp: the timestamp (unix epoch; implicitly UTC).

  Returns:
    the corresponding datetime.
  """
  return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc)

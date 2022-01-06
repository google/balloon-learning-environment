# coding=utf-8
# Copyright 2022 The Balloon Learning Environment Authors.
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

"""Simulator for a stratospheric superpressure balloon.

This simulates a simplified model of stratospheric balloon flight for an
altitude controlled superpressure balloon. The state for the balloon exists
above a simplified Cartesian plane rather than a real coordinate on the globe.

We define the coordinate space with [x, y] as the position of the balloon
relative to the station keeping target. x is kilometers along the latitude line
and y is kilometers along the latitude line. pressure is the barometric pressure
(similar to altitude, but with a nontrivial relationship). time_elapsed is
time but relative to start of the simulation / wind field generation.

Typical usage:

  wind_field = ...
  balloon = Balloon(0., 0., 6000)  # At the station keeping target, 6000 Pa.
  stride = timedelta(minutes=3)
  horizon = timedelta(days=2)
  for _ in range(horizon / stride):
    command = np.random.choice([
      AltitudeControlCommand.UP,
      AltitudeControlCommand.DOWN,
      AltitudeControlCommand.STAY])
    balloon.simulate_step(wind_field, command, stride)
    print(balloon.x, balloon.y, balloon.pressure)
"""

import dataclasses
import datetime as dt
import enum
from typing import Any, Dict, Tuple

from balloon_learning_environment.env import wind_field
from balloon_learning_environment.env.balloon import acs
from balloon_learning_environment.env.balloon import altitude_safety
from balloon_learning_environment.env.balloon import control
from balloon_learning_environment.env.balloon import envelope_safety
from balloon_learning_environment.env.balloon import power_safety
from balloon_learning_environment.env.balloon import solar
from balloon_learning_environment.env.balloon import standard_atmosphere
from balloon_learning_environment.env.balloon import thermal
from balloon_learning_environment.utils import constants
from balloon_learning_environment.utils import spherical_geometry
from balloon_learning_environment.utils import units
import numpy as np

import s2sphere as s2


class BalloonStatus(enum.Enum):
  OK = 0
  OUT_OF_POWER = 1
  BURST = 2
  ZEROPRESSURE = 3


@dataclasses.dataclass
class BalloonState(object):
  """A dataclass containing variables relevant to the balloon state.

  Attributes:
    center_latlng: The latitude and longitude the simulation is centered around.
    date_time: The current time.
    time_elapsed: The time elapsed in simulation from the time the object was
        initialized.

    envelope_volume_base: The y-intercept for the balloon envelope volume model.
    envelope_volume_dv_pressure: The slope for the balloon envelope volume
        model.
    envelope_mass: Mass of the balloon envelope.
    envelope_max_superpressure: Maximum superpressure the balloon envelope can
       withstand before bursting.
    envelope_cod: Coefficient of drag for the balloon envelope. (To a first
       order approximation the coefficient of drag for the overall flight
       system.)
    payload_mass: The mass of the payload. The term payload here refers to all
       parts of the flight system other than the balloon envelope.
    nighttime_power_load: The "hotel" power load, i.e., what is required to keep
       the lights on, of the flight system at night.
    daytime_power_load: The "hotel" power load, i.e., what is required to keep
       the lights on, of the flight system during the day.
    acs_valve_hole_diameter: The size of the valve opening when venting air from
       the ballonet through the altitude control system.
    battery_capacity: The amount of energy that can be stored on our batteries.
       (Modeled as an ideal energy reservoir.)

    x: The balloon's current x coordinate, i.e., translation from the station's
        position in the W->E direction.
    y: The balloon's current y coordinate, i.e., translation from the station's
        position in the S->N direction.
    pressure: The ambient pressure around the balloon. This maps to altitude.
    ambient_temperature: The ambient temperature around the balloon.
    mols_lift_gas: Mols of helium within the balloon envelope.
    mols_air: Mols of dry air within the ballonet.
    internal_temperature: The temperature of the gas within the balloon
        envelope. Modeled (approximated) as a constant temperature throughout
        both the air and helium chambers.
    envelope_volume: The volume of the balloon envelope.
    superpressure: The excess pressure of the gas within the balloon envelope
        as compared to the ambient pressure, i.e., internal pressure - ambient
        pressure.
    acs_power: The power applied to the altitude control system compressor to
        pump air from outside the balloon into the ballonet.
    acs_mass_flow: The amount of air being pumped by the altitude control
        system.
    solar_charging: The amount of power entering the system via solar panels.
    power_load: The amount of power being used by the flight system.
    battery_charge: The amount of energy stored on the batteries.

    last_command: The previous command (up/down/stay) executed by the balloon.
        Set to stay at init.
    status: The current status of the balloon.
    power_safety_layer_enabled: if true the balloon will be prevented from
        draining its power at night by moving too much. Note: the balloon
        may still be able to run out of power, since it still may not
        receive enough light on its solar panels immediately after sunrise.
    navigation_is_paused: if true it signifies that the power safety layer
        paused navigation the previous time simulate_step
        was called.
    power_safety_layer: The power safety layer for the balloon state.
    envelope_safety_layer: The safety layer that attempts to prevent
        the envelope from bursting.
    altitude_safety_layer: The safety layer that attempts to prevent
        the balloon from navigating to unsafe altitudes.

    latlng: The current position of the balloon.
    battery_soc: The state of charge of the battery in [0, 1].
    excess_energy: Whether the balloon has excess energy.
    navigation_is_paused: Whether navigation was paused on the last timestep.

    upwelling_infrared: The upwelling infrared value.
    pressure_ratio: The pressure / superpressure ratio.
  """
  center_latlng: s2.LatLng

  date_time: dt.datetime
  time_elapsed: dt.timedelta = dt.timedelta()

  # Flight vehicle constants.
  envelope_volume_base: float = 1804  # [m^3]
  envelope_volume_dv_pressure: float = 0.0199  # [m^3/Pa]
  envelope_mass: float = 68.5  # [kg]
  envelope_max_superpressure: float = 2380  # [Pa]
  # NOTE(scandido): We simplify the coefficient of drag of the balloon to a
  # constant, which is more or less valid when fully inflated at altitude.
  envelope_cod: float = 0.25  # [.]

  # 78.5kg for the payload plus 14kg of ballast.
  payload_mass: float = 92.5  # [kg]

  nighttime_power_load: units.Power = units.Power(watts=183.7)
  daytime_power_load: units.Power = units.Power(watts=120.4)

  acs_valve_hole_diameter: units.Distance = units.Distance(m=0.04)

  battery_capacity: units.Energy = units.Energy(watt_hours=3058.56)

  # State of the system.
  x: units.Distance = units.Distance(m=0)
  y: units.Distance = units.Distance(m=0)

  pressure: float = 6000.0  # [Pa]

  ambient_temperature: float = 206.0  # [K]

  mols_lift_gas: float = 6830.0  # [mols]
  mols_air: float = 0  # [mols]
  internal_temperature: float = 206.0  # [K]

  envelope_volume: float = 1804.0  # [kg/s]
  superpressure: float = 0  # [Pa]

  acs_power: units.Power = units.Power(watts=0)
  acs_mass_flow: float = 0  # [kg/s]

  solar_charging: units.Power = units.Power(watts=0)
  power_load: units.Power = units.Power(watts=0)
  # battery_charge initialized to 95% capacity.
  battery_charge: units.Energy = units.Energy(watt_hours=2905.6)

  last_command: control.AltitudeControlCommand = (
      control.AltitudeControlCommand.STAY)
  status: BalloonStatus = BalloonStatus.OK
  power_safety_layer_enabled: bool = True
  power_safety_layer: power_safety.PowerSafetyLayer = dataclasses.field(
      init=False, compare=False)
  envelope_safety_layer: envelope_safety.EnvelopeSafetyLayer = (
      dataclasses.field(init=False, compare=False))
  altitude_safety_layer: altitude_safety.AltitudeSafetyLayer = (
      dataclasses.field(init=False, compare=False))

  upwelling_infrared: float = 250.0  # [W/m^2]

  def __post_init__(self):
    self.power_safety_layer = power_safety.PowerSafetyLayer(
        self.latlng, self.date_time)
    self.envelope_safety_layer = envelope_safety.EnvelopeSafetyLayer(
        self.envelope_max_superpressure)
    self.altitude_safety_layer = altitude_safety.AltitudeSafetyLayer()

  @property
  def latlng(self) -> s2.LatLng:
    return spherical_geometry.calculate_latlng_from_offset(
        self.center_latlng, self.x, self.y)

  @property
  def battery_soc(self) -> float:
    """Returns the battery state of charge (soc).

    Returns the battery soc, a number in the range [0, 1], with 0 corresponding
    to the batteries being empty and 1 the battery being full.
    """
    return self.battery_charge / self.battery_capacity

  @property
  def excess_energy(self) -> bool:
    """Returns whether the balloon has excess energy."""
    solar_elevation, _, _ = solar.solar_calculator(
        self.latlng, self.date_time)
    solar_power = solar.solar_power(solar_elevation, self.pressure)
    return (solar_power > self.daytime_power_load and
            self.battery_soc > 0.99)

  @property
  def navigation_is_paused(self) -> bool:
    # Navigation is paused when the balloon is not allowed to descend.
    return (self.power_safety_layer.navigation_is_paused
            or self.envelope_safety_layer.navigation_is_paused
            or self.altitude_safety_layer.navigation_is_paused)

  @property
  def pressure_ratio(self) -> float:
    superpressure = max(self.superpressure, 0.0)
    return (self.pressure + superpressure) / self.pressure


class Balloon:
  """A simulation of a stratospheric balloon.

  This class holds the system state vector and equations of motion
  (simulate_step) for simulating a stratospheric balloon.
  """

  def __init__(self, balloon_state: BalloonState):
    self.state = balloon_state

  def simulate_step(
      self,
      wind_vector: wind_field.WindVector,
      atmosphere: standard_atmosphere.Atmosphere,
      action: control.AltitudeControlCommand,
      time_delta: dt.timedelta,
      stride: dt.timedelta = dt.timedelta(seconds=10),
  ) -> None:
    """Steps forward the simulation.

    This moves the balloon's state forward according to the dynamics of motion
    for a stratospheric balloon.

    Args:
      wind_vector: A vector corresponding to the wind to apply to the balloon.
      atmosphere: The atmospheric conditions the balloon is flying in.
      action: An AltitudeControlCommand for the system to take during this
        simulation step, i.e., up/down/stay.
      time_delta: How much time is elapsing during this step. Must be a multiple
        of stride.
      stride: The step size for the simulation of the balloon physics.
    """

    self.state.last_command = action

    assert self.state.status == BalloonStatus.OK, (
        'Stepping balloon after a terminal event occured. '
        f'({self.state.status.name})')

    # The safety layers may prevent some actions from taking effect in
    # certain situations. While the correct interpretation
    # is, e.g., that even when a down action is commanded the altitude control
    # system may not be powered up, we simplify the code readability by
    # remapping the action that takes effect to, in this example, the stay
    # command.
    # The envelope/atitude safety layers trumps the power safety layer. This is
    # because, at worst, the envelope/altitude safety layers only recommend
    # ascending, which doesn't take any extra power.
    # Finally, the altitude safety layer trumps the envelope safety layer.
    # This is because ascending shouldn't be harmful to superpressure in most
    # situtations.
    effective_action = action
    if self.state.power_safety_layer_enabled:
      effective_action = self.state.power_safety_layer.get_action(
          effective_action, self.state.date_time,
          self.state.nighttime_power_load, self.state.battery_charge,
          self.state.battery_capacity)
    effective_action = self.state.envelope_safety_layer.get_action(
        effective_action, self.state.superpressure)
    effective_action = self.state.altitude_safety_layer.get_action(
        effective_action, atmosphere, self.state.pressure)

    outer_stride = int(time_delta.total_seconds())
    inner_stride = int(stride.total_seconds())
    assert outer_stride % inner_stride == 0, (
        f'The outer simulation stride (time_delta={time_delta}) must be a '
        f'multiple of the inner simulation stride (stride={stride})')

    for _ in range(outer_stride // inner_stride):
      state_changes = Balloon._simulate_step_internal(
          self.state, wind_vector, atmosphere, effective_action, stride)
      for k, v in state_changes.items():
        setattr(self.state, k, v)

      if self.state.status != BalloonStatus.OK:
        break

  @staticmethod
  def _simulate_step_internal(
      state: BalloonState,
      wind_vector: wind_field.WindVector,
      atmosphere: standard_atmosphere.Atmosphere,
      action: control.AltitudeControlCommand,
      stride: dt.timedelta,
  ) -> Dict[str, Any]:
    """Steps forward the simulation.

    This moves the balloon's state forward according to the dynamics of motion
    for a stratospheric balloon, and returns the attributes that need to be
    updated.

    Args:
      state: Current state.
      wind_vector: A vector corresponding to the wind to apply to the balloon.
      atmosphere: The atmospheric conditions the balloon is flying in.
      action: An AltitudeControlCommand for the system to take during this
        simulation step, i.e., up/down/stay.
      stride: The simulation stride, e.g., time delta elapsing in the
        simulation.

    Returns:
      The attributes that need to be updated.
    """

    state_changes = {}

    # NOTE(scandido): In the original simulator we modeled this using the
    # stochastic hybrid automaton to use a typical ODE solver. For the purposes
    # of this benchmark we have chosen to write the physics in a single, concise
    # "step forward in time" function so folks can more easily inspect the
    # model and assumptions.

    ## Step 1: The balloon moves with the wind 🌬.

    # NOTE(scandido): We should adjust this slightly to account for the altitude
    # of the balloon but it's a small effect we ignore.
    state_changes['x'] = state.x + (wind_vector.u * stride)
    state_changes['y'] = state.y + (wind_vector.v * stride)

    ## Step 2: The balloon moves up and down based on the buoyancy of the flight
    # system 🦆.

    # Compute the differential ascent rate.
    #
    # mg = Fb + Fd  (steady state)
    #    = rho Volume g - 1/2 rho (d^2h/dt^2) C_drag A
    #
    #                   rho Volume g - mg
    #  ==> d^2h/dt^2 = -----------------
    #                  1/2 rho C_drag A
    #
    # C_drag is via linear fit during the ascent model.
    # A ~ displacement^(2/3)

    rho_air = (state.pressure * constants.DRY_AIR_MOLAR_MASS) / (
        constants.UNIVERSAL_GAS_CONSTANT * state.ambient_temperature)

    drag = state.envelope_cod * state.envelope_volume**(2.0 / 3.0)

    total_flight_system_mass = (
        constants.HE_MOLAR_MASS * state.mols_lift_gas +
        constants.DRY_AIR_MOLAR_MASS * state.mols_air + state.envelope_mass +
        state.payload_mass)

    direction = (1.0 if rho_air * state.envelope_volume >=
                 total_flight_system_mass else -1.0)
    dh_dt = direction * np.sqrt(  # [m/s]
        np.abs(2 * (rho_air * state.envelope_volume -
                    total_flight_system_mass) * constants.GRAVITY /
               (rho_air * drag)))

    # We have the ascent rate in [m/s] but what we really care about is the
    # differential change in our state variable, pressure. Our pressure to
    # height map is a point-wise set of coordinates and we use a linear
    # interpolation. Thus, a local approximation of dp/dt is just a linear
    # factor away from dh/dt.
    #
    # Specifically:
    #
    #   p = m * h + b ==( chain rule ) ==> dp/dt = dp/dh * dh/dt = m * dh/dt
    dp = 1.0  # [Pa] A small pressure delta.
    height0 = atmosphere.at_pressure(state.pressure).height.meters
    height1 = atmosphere.at_pressure(state.pressure +
                                     direction * dp).height.meters
    dp_dh = direction * dp / (height1 - height0)
    dp_dt = dp_dh * dh_dt

    state_changes['pressure'] = state.pressure + dp_dt * stride.total_seconds()

    ## Step 3: Look up the ambient temperature, upwelling infrared radiation,
    # and solar radiation, and calculate the internal temperature of the
    # balloon 🌡.

    solar_elevation, _, solar_flux = solar.solar_calculator(
        state.latlng, state.date_time)

    # Use standard atmosphere for temperature.
    # A longer term goal would be to create a model with temperature lapse
    # variability.
    state_changes['ambient_temperature'] = atmosphere.at_pressure(
        state.pressure).temperature

    # TODO(scandido): Consider putting the main equations (dbtemp_dt) here to
    # match the rest of the file.
    d_internal_temp = thermal.d_balloon_temperature_dt(
        state.envelope_volume, state.envelope_mass, state.internal_temperature,
        state.ambient_temperature, state.pressure, solar_elevation,
        solar_flux, state.upwelling_infrared)
    state_changes['internal_temperature'] = (
        state.internal_temperature + d_internal_temp * stride.total_seconds())

    ## Step 4: Calculate superpressure and volume of the balloon 🎈.
    state_changes['envelope_volume'], state_changes['superpressure'] = (
        Balloon.calculate_superpressure_and_volume(
            state.mols_lift_gas, state.mols_air, state.internal_temperature,
            state.pressure, state.envelope_volume_base,
            state.envelope_volume_dv_pressure))

    if state_changes['superpressure'] > state.envelope_max_superpressure:
      state_changes['status'] = BalloonStatus.BURST
    if state_changes['superpressure'] <= 0.0:
      state_changes['status'] = BalloonStatus.ZEROPRESSURE

    ## Step 5: Calculate, based on desired action, whether we'll use the
    # altitude control system (ACS) ⚙️. Adjust power usage accordingly.

    if action == control.AltitudeControlCommand.UP:
      state_changes['acs_power'] = units.Power(watts=0.0)
      valve_area = np.pi * state.acs_valve_hole_diameter.meters**2 / 4.0
      # Coefficient of drag on the air passing through the ACS from the
      # aperture. A measured quantity.
      default_valve_hole_cd = 0.62  # [.]
      gas_density = (
          state.superpressure +
          state.pressure) * constants.DRY_AIR_MOLAR_MASS / (
              constants.UNIVERSAL_GAS_CONSTANT * state.internal_temperature)
      state_changes['acs_mass_flow'] = (
          -1 * default_valve_hole_cd * valve_area * np.sqrt(
              2.0 * state.superpressure * gas_density))
    elif action == control.AltitudeControlCommand.DOWN:
      # Run the ACS compressor at a power level that maximizes mols of air
      # pushed into the ballonet per watt of energy at the current pressure
      # ratio (backpressure the compressor is pushing against).
      state_changes['acs_power'] = acs.get_most_efficient_power(
          state.pressure_ratio)
      # Compute mass flow rate by first computing efficiency of air flow.
      efficiency = acs.get_fan_efficiency(state.pressure_ratio,
                                          state_changes['acs_power'])
      state_changes['acs_mass_flow'] = acs.get_mass_flow(
          state_changes['acs_power'], efficiency)
    else:  # action == control.AltitudeControlCommand.STAY.
      state_changes['acs_power'] = units.Power(watts=0.0)
      state_changes['acs_mass_flow'] = 0.0

    state_changes['mols_air'] = state.mols_air + (
        state_changes['acs_mass_flow'] /
        constants.DRY_AIR_MOLAR_MASS) * stride.total_seconds()
    state_changes['mols_air'] = np.clip(
        state_changes['mols_air'], a_min=0.0, a_max=None)

    ## Step 6: Calculate energy usage and collection, and move coulombs onto
    # and off of the battery as apppropriate. 🔋

    is_day = solar_elevation > solar.MIN_SOLAR_EL_DEG
    state_changes['solar_charging'] = (
        solar.solar_power(solar_elevation, state.pressure)
        if is_day else units.Power(watts=0.0))
    # TODO(scandido): Introduce a variable power load for cold upwelling IR?
    state_changes['power_load'] = (
        state.daytime_power_load if is_day else state.nighttime_power_load)
    state_changes['power_load'] += state_changes['acs_power']

    # We use a simplified model of a battery that is kept at a constant
    # temperature and acts like an ideal energy reservoir.
    state_changes['battery_charge'] = state.battery_charge + (
        state_changes['solar_charging'] - state_changes['power_load']) * stride
    state_changes['battery_charge'] = np.clip(
        state_changes['battery_charge'], units.Energy(watt_hours=0.0),
        state.battery_capacity)

    if state_changes['battery_charge'].watt_hours <= 0.0:
      state_changes['status'] = BalloonStatus.OUT_OF_POWER

    # This must be updated in the inner loop, since the safety layer and
    # solar calculations rely on the current time.
    state_changes['date_time'] = state.date_time + stride
    state_changes['time_elapsed'] = state.time_elapsed + stride

    return state_changes

  @staticmethod
  def calculate_superpressure_and_volume(
      mols_lift_gas: float,
      mols_air: float,
      internal_temperature: float,
      pressure: float,
      envelope_volume_base: float,
      envelope_volume_dv_pressure: float) -> Tuple[float, float]:
    """Calculates the current superpressure and volume of the balloon.

    Args:
      mols_lift_gas: Mols of helium within the balloon envelope [mols].
      mols_air: Mols of air within the ballonet [mols].
      internal_temperature: The temperature of the gas in the envelope.
      pressure: Ambient pressure of the balloon [Pa].
      envelope_volume_base: The y-intercept for the balloon envelope volume
        model [m^3].
      envelope_volume_dv_pressure: The slope for the balloon envelope volume
        model.

    Returns:
      An (envelope_volume, superpressure) tuple.
    """
    envelope_volume = 0.0
    superpressure = 0.0

    # Compute the unconstrained volume of the balloon which is
    # (n_gas + n_air) * R * T_gas / P_amb. This is the volume the balloon would
    # expand to if the material holding the lift gas didn't give any resistence,
    # e.g., to a first-order approximation a latex weather ballon.
    unconstrained_volume = (
        (mols_lift_gas + mols_air) *
        constants.UNIVERSAL_GAS_CONSTANT * internal_temperature /
        pressure)

    if unconstrained_volume <= envelope_volume_base:
      # Not fully inflated.
      envelope_volume = unconstrained_volume
      superpressure = 0.0
    else:
      # System of equations for a fully inflated balloon:
      #
      #  V = V0 + dv_dp * (P_gas - P_amb)
      #  P_gas = n * R * T_gas / V
      #
      # Solve the quadratic equation for volume:
      b = -(
          envelope_volume_base -
          envelope_volume_dv_pressure * pressure)
      c = -(
          envelope_volume_dv_pressure * unconstrained_volume *
          pressure)

      envelope_volume = 0.5 * (-b + np.sqrt(b * b - 4 * c))
      superpressure = (
          pressure * unconstrained_volume /
          envelope_volume - pressure)

    return envelope_volume, superpressure

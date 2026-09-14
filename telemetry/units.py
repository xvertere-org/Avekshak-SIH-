"""
Explicit SI and Engineering Unit Conversion Engine for SIH26054 Digital Twin.

Architectural Guarantees:
1. Dimensional Safety: Incompatible physical dimensions (e.g. pressure to temperature)
   strictly raise UnitConversionError.
2. Vibration Dimensionality: Acceleration (g, m/s²) and velocity (mm/s) are strictly distinct
   dimensions. Direct scalar conversion between g and mm/s is prohibited and raises UnitConversionError.
3. Fuel Density Provenance: Mass flow to volumetric flow requires an explicit fuel density
   parameter (rho_fuel). It is never hardcoded as an authoritative physical constant.
4. Deterministic Calculations: Conversions use exact, independently verifiable conversion constants.
"""

from enum import Enum
from typing import Dict, Tuple, Optional, Callable, Set
import math


class PhysicalDimension(str, Enum):
    """Physical quantity dimensions for aero-engine telemetry."""
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    VOLUMETRIC_FLOW = "VOLUMETRIC_FLOW"
    MASS_FLOW = "MASS_FLOW"
    ACCELERATION = "ACCELERATION"
    VELOCITY = "VELOCITY"
    LENGTH = "LENGTH"
    ROTATIONAL_SPEED = "ROTATIONAL_SPEED"
    VOLTAGE = "VOLTAGE"
    CURRENT = "CURRENT"
    FRACTION = "FRACTION"
    TIME = "TIME"


class UnitConversionError(ValueError):
    """Raised when an invalid or cross-dimensional unit conversion is attempted."""
    pass


# Exact conversion constants
STANDARD_GRAVITY = 9.80665       # m/s^2 per g (ISO 80000-3)
PA_PER_BAR = 100000.0             # Pa per bar (exact SI)
PA_PER_PSI = 6894.757293168361    # Pa per psi
PA_PER_INHG = 3386.3886666667     # Pa per inHg at 0 °C
METERS_PER_FOOT = 0.3048          # m per ft (exact international foot)
SECONDS_PER_HOUR = 3600.0         # s per hour
LITERS_PER_M3 = 1000.0            # L per m^3
LITERS_PER_US_GALLON = 3.785411784 # L per US liquid gallon
KG_PER_POUND = 0.45359237         # kg per lb (exact)


def _normalize_unit_str(unit_str: str) -> str:
    """Normalize unit string for case-insensitive lookup."""
    u = unit_str.strip().lower()
    u = u.replace("degc", "c").replace("°c", "c").replace("deg c", "c")
    u = u.replace("degf", "f").replace("°f", "f").replace("deg f", "f")
    u = u.replace("m/s^2", "m/s2").replace("l/hr", "l/h")
    return u


# Unit to Dimension mapping
UNIT_DIMENSIONS: Dict[str, PhysicalDimension] = {
    # Pressure (Canonical: bar)
    "bar": PhysicalDimension.PRESSURE,
    "pa": PhysicalDimension.PRESSURE,
    "hpa": PhysicalDimension.PRESSURE,
    "kpa": PhysicalDimension.PRESSURE,
    "mpa": PhysicalDimension.PRESSURE,
    "psi": PhysicalDimension.PRESSURE,
    "inhg": PhysicalDimension.PRESSURE,

    # Temperature (Canonical: degC)
    "c": PhysicalDimension.TEMPERATURE,
    "k": PhysicalDimension.TEMPERATURE,
    "f": PhysicalDimension.TEMPERATURE,

    # Volumetric Flow (Canonical: L/h)
    "l/h": PhysicalDimension.VOLUMETRIC_FLOW,
    "l/min": PhysicalDimension.VOLUMETRIC_FLOW,
    "m3/s": PhysicalDimension.VOLUMETRIC_FLOW,
    "gpm": PhysicalDimension.VOLUMETRIC_FLOW,

    # Mass Flow (Canonical: kg/s or kg/h)
    "kg/s": PhysicalDimension.MASS_FLOW,
    "kg/h": PhysicalDimension.MASS_FLOW,
    "g/s": PhysicalDimension.MASS_FLOW,
    "pph": PhysicalDimension.MASS_FLOW,

    # Acceleration (Canonical: g)
    "g": PhysicalDimension.ACCELERATION,
    "m/s2": PhysicalDimension.ACCELERATION,
    "cm/s2": PhysicalDimension.ACCELERATION,

    # Velocity (Canonical: mm/s)
    "mm/s": PhysicalDimension.VELOCITY,
    "m/s": PhysicalDimension.VELOCITY,
    "in/s": PhysicalDimension.VELOCITY,

    # Length / Altitude (Canonical: m)
    "m": PhysicalDimension.LENGTH,
    "ft": PhysicalDimension.LENGTH,
    "km": PhysicalDimension.LENGTH,
    "mile": PhysicalDimension.LENGTH,

    # Rotational Speed (Canonical: RPM)
    "rpm": PhysicalDimension.ROTATIONAL_SPEED,
    "rad/s": PhysicalDimension.ROTATIONAL_SPEED,
    "hz": PhysicalDimension.ROTATIONAL_SPEED,

    # Voltage (Canonical: V)
    "v": PhysicalDimension.VOLTAGE,
    "mv": PhysicalDimension.VOLTAGE,
    "kv": PhysicalDimension.VOLTAGE,

    # Current (Canonical: A)
    "a": PhysicalDimension.CURRENT,
    "ma": PhysicalDimension.CURRENT,

    # Dimensionless
    "%": PhysicalDimension.FRACTION,
    "percent": PhysicalDimension.FRACTION,
    "fraction": PhysicalDimension.FRACTION,

    # Time (Canonical: s)
    "s": PhysicalDimension.TIME,
    "sec": PhysicalDimension.TIME,
    "min": PhysicalDimension.TIME,
    "h": PhysicalDimension.TIME,
    "hour": PhysicalDimension.TIME,
}

CANONICAL_UNITS: Dict[PhysicalDimension, str] = {
    PhysicalDimension.PRESSURE: "bar",
    PhysicalDimension.TEMPERATURE: "degC",
    PhysicalDimension.VOLUMETRIC_FLOW: "L/h",
    PhysicalDimension.MASS_FLOW: "kg/s",
    PhysicalDimension.ACCELERATION: "g",
    PhysicalDimension.VELOCITY: "mm/s",
    PhysicalDimension.LENGTH: "m",
    PhysicalDimension.ROTATIONAL_SPEED: "RPM",
    PhysicalDimension.VOLTAGE: "V",
    PhysicalDimension.CURRENT: "A",
    PhysicalDimension.FRACTION: "%",
    PhysicalDimension.TIME: "s",
}


def get_unit_dimension(unit_str: str) -> PhysicalDimension:
    """Determine physical dimension of a unit string."""
    norm = _normalize_unit_str(unit_str)
    dim = UNIT_DIMENSIONS.get(norm)
    if dim is None:
        raise UnitConversionError(f"Unknown or unsupported unit: '{unit_str}'")
    return dim


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a numeric scalar value from one engineering unit to another within the same physical dimension.

    Raises:
        UnitConversionError: If units belong to different physical dimensions or are unknown.
    """
    if value is None or math.isnan(value):
        return value

    u_from = _normalize_unit_str(from_unit)
    u_to = _normalize_unit_str(to_unit)

    if u_from == u_to:
        return float(value)

    dim_from = get_unit_dimension(from_unit)
    dim_to = get_unit_dimension(to_unit)

    if dim_from != dim_to:
        # Explicit guard against acceleration to velocity conversion
        if (dim_from == PhysicalDimension.ACCELERATION and dim_to == PhysicalDimension.VELOCITY) or \
           (dim_from == PhysicalDimension.VELOCITY and dim_to == PhysicalDimension.ACCELERATION):
            raise UnitConversionError(
                f"Cannot directly convert between acceleration ({from_unit}) and velocity ({to_unit}) "
                "without spectral/frequency-domain integration or an explicit vibration harmonic model."
            )
        raise UnitConversionError(
            f"Cross-dimensional unit conversion is prohibited: '{from_unit}' ({dim_from.value}) to '{to_unit}' ({dim_to.value})."
        )

    val = float(value)

    # 1. PRESSURE (Base: Pa)
    if dim_from == PhysicalDimension.PRESSURE:
        # to Pa
        if u_from == "pa":
            pa = val
        elif u_from == "hpa":
            pa = val * 100.0
        elif u_from == "kpa":
            pa = val * 1000.0
        elif u_from == "mpa":
            pa = val * 1000000.0
        elif u_from == "bar":
            pa = val * PA_PER_BAR
        elif u_from == "psi":
            pa = val * PA_PER_PSI
        elif u_from == "inhg":
            pa = val * PA_PER_INHG
        else:
            raise UnitConversionError(f"Unsupported pressure unit: {from_unit}")

        # from Pa to target
        if u_to == "pa":
            return pa
        elif u_to == "hpa":
            return pa / 100.0
        elif u_to == "kpa":
            return pa / 1000.0
        elif u_to == "mpa":
            return pa / 1000000.0
        elif u_to == "bar":
            return pa / PA_PER_BAR
        elif u_to == "psi":
            return pa / PA_PER_PSI
        elif u_to == "inhg":
            return pa / PA_PER_INHG

    # 2. TEMPERATURE (Base: deg C)
    elif dim_from == PhysicalDimension.TEMPERATURE:
        # to deg C
        if u_from == "c":
            c = val
        elif u_from == "k":
            c = val - 273.15
        elif u_from == "f":
            c = (val - 32.0) * (5.0 / 9.0)
        else:
            raise UnitConversionError(f"Unsupported temperature unit: {from_unit}")

        # from deg C to target
        if u_to == "c":
            return c
        elif u_to == "k":
            return c + 273.15
        elif u_to == "f":
            return (c * 9.0 / 5.0) + 32.0

    # 3. ACCELERATION (Base: m/s^2)
    elif dim_from == PhysicalDimension.ACCELERATION:
        # to m/s^2
        if u_from == "m/s2":
            acc_ms2 = val
        elif u_from == "g":
            acc_ms2 = val * STANDARD_GRAVITY
        elif u_from == "cm/s2":
            acc_ms2 = val * 0.01
        else:
            raise UnitConversionError(f"Unsupported acceleration unit: {from_unit}")

        # from m/s^2 to target
        if u_to == "m/s2":
            return acc_ms2
        elif u_to == "g":
            return acc_ms2 / STANDARD_GRAVITY
        elif u_to == "cm/s2":
            return acc_ms2 * 100.0

    # 4. VOLUMETRIC FLOW (Base: L/h)
    elif dim_from == PhysicalDimension.VOLUMETRIC_FLOW:
        # to L/h
        if u_from == "l/h":
            lh = val
        elif u_from == "l/min":
            lh = val * 60.0
        elif u_from == "m3/s":
            lh = val * LITERS_PER_M3 * SECONDS_PER_HOUR
        elif u_from == "gpm":
            lh = val * LITERS_PER_US_GALLON * 60.0
        else:
            raise UnitConversionError(f"Unsupported volumetric flow unit: {from_unit}")

        # from L/h to target
        if u_to == "l/h":
            return lh
        elif u_to == "l/min":
            return lh / 60.0
        elif u_to == "m3/s":
            return lh / (LITERS_PER_M3 * SECONDS_PER_HOUR)
        elif u_to == "gpm":
            return lh / (LITERS_PER_US_GALLON * 60.0)

    # 5. MASS FLOW (Base: kg/s)
    elif dim_from == PhysicalDimension.MASS_FLOW:
        # to kg/s
        if u_from == "kg/s":
            kgs = val
        elif u_from == "kg/h":
            kgs = val / SECONDS_PER_HOUR
        elif u_from == "g/s":
            kgs = val * 0.001
        elif u_from == "pph":
            kgs = (val * KG_PER_POUND) / SECONDS_PER_HOUR
        else:
            raise UnitConversionError(f"Unsupported mass flow unit: {from_unit}")

        # from kg/s to target
        if u_to == "kg/s":
            return kgs
        elif u_to == "kg/h":
            return kgs * SECONDS_PER_HOUR
        elif u_to == "g/s":
            return kgs * 1000.0
        elif u_to == "pph":
            return (kgs * SECONDS_PER_HOUR) / KG_PER_POUND

    # 6. LENGTH (Base: m)
    elif dim_from == PhysicalDimension.LENGTH:
        # to m
        if u_from == "m":
            m = val
        elif u_from == "ft":
            m = val * METERS_PER_FOOT
        elif u_from == "km":
            m = val * 1000.0
        elif u_from == "mile":
            m = val * 1609.344
        else:
            raise UnitConversionError(f"Unsupported length unit: {from_unit}")

        # from m to target
        if u_to == "m":
            return m
        elif u_to == "ft":
            return m / METERS_PER_FOOT
        elif u_to == "km":
            return m / 1000.0
        elif u_to == "mile":
            return m / 1609.344

    # 7. ROTATIONAL SPEED (Base: RPM)
    elif dim_from == PhysicalDimension.ROTATIONAL_SPEED:
        # to RPM
        if u_from == "rpm":
            rpm = val
        elif u_from == "rad/s":
            rpm = val * (60.0 / (2.0 * math.pi))
        elif u_from == "hz":
            rpm = val * 60.0
        else:
            raise UnitConversionError(f"Unsupported rotational speed unit: {from_unit}")

        # from RPM to target
        if u_to == "rpm":
            return rpm
        elif u_to == "rad/s":
            return rpm * ((2.0 * math.pi) / 60.0)
        elif u_to == "hz":
            return rpm / 60.0

    # 8. VOLTAGE (Base: V)
    elif dim_from == PhysicalDimension.VOLTAGE:
        if u_from == "v":
            v = val
        elif u_from == "mv":
            v = val * 0.001
        elif u_from == "kv":
            v = val * 1000.0
        else:
            raise UnitConversionError(f"Unsupported voltage unit: {from_unit}")

        if u_to == "v":
            return v
        elif u_to == "mv":
            return v * 1000.0
        elif u_to == "kv":
            return v * 0.001

    # 9. CURRENT (Base: A)
    elif dim_from == PhysicalDimension.CURRENT:
        if u_from == "a":
            a = val
        elif u_from == "ma":
            a = val * 0.001
        else:
            raise UnitConversionError(f"Unsupported current unit: {from_unit}")

        if u_to == "a":
            return a
        elif u_to == "ma":
            return a * 1000.0

    # 10. FRACTION / PERCENT (Base: %)
    elif dim_from == PhysicalDimension.FRACTION:
        if u_from in ("%", "percent"):
            pct = val
        elif u_from == "fraction":
            pct = val * 100.0
        else:
            raise UnitConversionError(f"Unsupported fraction unit: {from_unit}")

        if u_to in ("%", "percent"):
            return pct
        elif u_to == "fraction":
            return pct * 0.01

    # 11. TIME (Base: s)
    elif dim_from == PhysicalDimension.TIME:
        if u_from in ("s", "sec"):
            s = val
        elif u_from == "min":
            s = val * 60.0
        elif u_from in ("h", "hour"):
            s = val * SECONDS_PER_HOUR
        else:
            raise UnitConversionError(f"Unsupported time unit: {from_unit}")

        if u_to in ("s", "sec"):
            return s
        elif u_to == "min":
            return s / 60.0
        elif u_to in ("h", "hour"):
            return s / SECONDS_PER_HOUR

    raise UnitConversionError(f"Unhandled unit conversion: '{from_unit}' to '{to_unit}'")


def convert_mass_to_volumetric_flow(
    mass_flow_val: float,
    from_unit: str = "kg/s",
    rho_fuel_kg_L: Optional[float] = None,
) -> Tuple[float, str, str]:
    """
    Convert mass fuel flow to canonical volumetric fuel flow (L/h).

    Mandatory Guardrail:
    - Fuel density is an operational assumption, NOT an authoritative physical constant.
    - Requires an explicit rho_fuel_kg_L parameter.
    - Returns (flow_L_h, "L/h", provenance_note).
    """
    if rho_fuel_kg_L is None or rho_fuel_kg_L <= 0.0:
        raise ValueError(
            "Fuel density 'rho_fuel_kg_L' must be explicitly provided and positive "
            "to convert mass flow to volumetric flow. Hardcoding is prohibited."
        )

    # First convert mass flow to kg/h
    kg_per_hour = convert_unit(mass_flow_val, from_unit, "kg/h")

    # Volume (L/h) = Mass (kg/h) / Density (kg/L)
    vol_flow_lh = kg_per_hour / float(rho_fuel_kg_L)

    provenance_note = f"DERIVED_FROM_MASS_FLOW: assumed rho_fuel={rho_fuel_kg_L:.4f} kg/L"
    return vol_flow_lh, "L/h", provenance_note

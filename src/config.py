"""
Configuration Module

Dataclasses, enums, and constants for network optimization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

EARTH_RADIUS_MILES = 3958.8


# =============================================================================
# ENUMS
# =============================================================================

class LoadStrategy(Enum):
    CONTAINER = "container"
    FLUID = "fluid"


class PathType(Enum):
    DIRECT_INJECTION = "direct_injection"
    OD_MM = "od_mm"  # O=D middle-mile: no linehaul, but injection + LM sort
    TWO_TOUCH = "2_touch"
    THREE_TOUCH = "3_touch"
    FOUR_TOUCH = "4_touch"
    FIVE_TOUCH = "5_touch"


class SortLevel(Enum):
    REGION = "region"
    MARKET = "market"
    SORT_GROUP = "sort_group"


class DestSortLevel(Enum):
    """Sort level at destination facility (for regional sort paths)."""
    REGION = "region"
    MARKET = "market"
    SORT_GROUP = "sort_group"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CostParameters:
    injection_sort_cost_per_pkg: float
    intermediate_sort_cost_per_pkg: float
    last_mile_sort_cost_per_pkg: float
    last_mile_delivery_cost_per_pkg: float
    container_handling_cost: float
    sort_points_per_destination: float
    sort_setup_cost_per_point: float = 0.0


@dataclass
class OptimizationConstants:
    MAX_SOLVER_TIME_SECONDS: int = 300
    NUM_SOLVER_WORKERS: int = 8
    CUBE_SCALE_FACTOR: int = 1000
    BIG_M: int = 10_000_000
    EPSILON: float = 1e-9


@dataclass
class ValidationTolerances:
    SHARE_SUM_TOLERANCE: float = 0.01
    COST_TOLERANCE: float = 0.001
    FILL_RATE_MAX: float = 1.0


# =============================================================================
# FILE TEMPLATES
# =============================================================================

OUTPUT_FILE_TEMPLATE = "network_opt_{scenario_id}_{strategy}.xlsx"


# =============================================================================
# FEASIBLE PATHS SCHEMA
# =============================================================================

FEASIBLE_PATHS_REQUIRED_COLUMNS = [
    "scenario_id",
    "origin",
    "dest",
    "node_1",
    "node_2",
    "path_type",
    "sort_level",
    "total_path_miles",
    "direct_miles",
    "tit_hours",
    "sla_met",
    "uses_only_active_arcs",
    "pkgs_mm",
    "pkgs_zs",
    "pkgs_di",
    "zone",
]

FEASIBLE_PATHS_OPTIONAL_COLUMNS = [
    "node_3",
    "node_4",
    "node_5",
    "dest_sort_level",
    "tit_sort_hours",
    "tit_crossdock_hours",
    "tit_transit_hours",
    "tit_dwell_hours",
]

VALID_PATH_TYPES = {"direct_injection", "od_mm", "2_touch", "3_touch", "4_touch", "5_touch"}
VALID_SORT_LEVELS = {"region", "market", "sort_group"}


# =============================================================================
# PARSING HELPERS
# =============================================================================

def parse_path_type(value: str) -> PathType:
    """Parse path_type string to enum."""
    mapping = {
        "direct_injection": PathType.DIRECT_INJECTION,
        "od_mm": PathType.OD_MM,
        "2_touch": PathType.TWO_TOUCH,
        "3_touch": PathType.THREE_TOUCH,
        "4_touch": PathType.FOUR_TOUCH,
        "5_touch": PathType.FIVE_TOUCH,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Invalid path_type: {value}. Must be one of {list(mapping.keys())}")
    return mapping[key]


def parse_sort_level(value: str) -> SortLevel:
    """Parse sort_level string to enum."""
    mapping = {
        "region": SortLevel.REGION,
        "market": SortLevel.MARKET,
        "sort_group": SortLevel.SORT_GROUP,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Invalid sort_level: {value}. Must be one of {list(mapping.keys())}")
    return
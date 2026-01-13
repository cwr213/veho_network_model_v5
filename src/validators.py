"""
Input Validation Module

Validates all input data before optimization.
"""

import pandas as pd
from typing import Dict, Set

from .config import (
    FEASIBLE_PATHS_REQUIRED_COLUMNS,
    VALID_PATH_TYPES,
    VALID_SORT_LEVELS,
)
from .io_loader import get_feasible_paths_scenario_ids


def validate_inputs(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Run all validations on input data.

    Raises ValueError with descriptive message on failure.
    """
    _validate_facilities(dfs["facilities"])
    _validate_mileage_bands(dfs["mileage_bands"])
    _validate_cost_params(dfs["cost_params"])
    _validate_container_params(dfs["container_params"])
    _validate_package_mix(dfs["package_mix"])
    _validate_scenarios(dfs["scenarios"])
    _validate_run_settings(dfs["run_settings"])
    _validate_feasible_paths(dfs["feasible_paths"])
    _validate_scenario_feasible_paths_match(dfs["scenarios"], dfs["feasible_paths"])
    _validate_referential_integrity(dfs)


def _validate_facilities(df: pd.DataFrame) -> None:
    """Validate facilities sheet."""
    required = ["facility_name", "type", "lat", "lon"]
    _check_required_columns(df, required, "facilities")

    if df["facility_name"].duplicated().any():
        dupes = df[df["facility_name"].duplicated()]["facility_name"].tolist()
        raise ValueError(f"Duplicate facility names: {dupes}")

    valid_types = {"hub", "hybrid", "launch"}
    invalid = set(df["type"].str.lower()) - valid_types
    if invalid:
        raise ValueError(f"Invalid facility types: {invalid}. Must be one of {valid_types}")

    if (df["lat"].abs() > 90).any():
        raise ValueError("Latitude must be between -90 and 90")
    if (df["lon"].abs() > 180).any():
        raise ValueError("Longitude must be between -180 and 180")


def _validate_mileage_bands(df: pd.DataFrame) -> None:
    """Validate mileage_bands sheet."""
    required = ["mileage_band_min", "mileage_band_max", "fixed_cost_per_truck", "variable_cost_per_mile", "circuity_factor", "mph"]
    _check_required_columns(df, required, "mileage_bands")

    if (df["mileage_band_min"] < 0).any():
        raise ValueError("mileage_band_min must be non-negative")
    if (df["mileage_band_max"] <= df["mileage_band_min"]).any():
        raise ValueError("mileage_band_max must be greater than mileage_band_min")
    if (df["circuity_factor"] < 1).any():
        raise ValueError("circuity_factor must be >= 1")
    if (df["mph"] <= 0).any():
        raise ValueError("mph must be positive")


def _validate_cost_params(df: pd.DataFrame) -> None:
    """Validate cost_params sheet."""
    required_keys = [
        "injection_sort_cost_per_pkg",
        "intermediate_sort_cost_per_pkg",
        "last_mile_sort_cost_per_pkg",
        "last_mile_delivery_cost_per_pkg",
        "container_handling_cost",
        "sort_points_per_destination",
    ]

    if "key" not in df.columns or "value" not in df.columns:
        raise ValueError("cost_params must have 'key' and 'value' columns")

    present_keys = set(df["key"].str.lower().str.strip())
    missing = set(k.lower() for k in required_keys) - present_keys
    if missing:
        raise ValueError(f"Missing required cost_params: {missing}")

    for _, row in df.iterrows():
        key = str(row["key"]).strip().lower()
        val = row["value"]
        if key in [k.lower() for k in required_keys]:
            if pd.isna(val) or not isinstance(val, (int, float)):
                raise ValueError(f"cost_params '{key}' must be numeric, got {val}")
            if val < 0:
                raise ValueError(f"cost_params '{key}' must be non-negative, got {val}")


def _validate_container_params(df: pd.DataFrame) -> None:
    """Validate container_params sheet."""
    required = ["container_type", "usable_cube_cuft", "pack_utilization_container"]
    _check_required_columns(df, required, "container_params")

    if not (df["container_type"].str.lower() == "gaylord").any():
        raise ValueError("container_params must include 'gaylord' container type")

    if (df["usable_cube_cuft"] <= 0).any():
        raise ValueError("usable_cube_cuft must be positive")
    if (df["pack_utilization_container"] <= 0).any() or (df["pack_utilization_container"] > 1).any():
        raise ValueError("pack_utilization_container must be between 0 and 1")


def _validate_package_mix(df: pd.DataFrame) -> None:
    """Validate package_mix sheet."""
    required = ["package_type", "avg_cube_cuft", "share_of_pkgs"]
    _check_required_columns(df, required, "package_mix")

    if (df["avg_cube_cuft"] <= 0).any():
        raise ValueError("avg_cube_cuft must be positive")
    if (df["share_of_pkgs"] < 0).any() or (df["share_of_pkgs"] > 1).any():
        raise ValueError("share_of_pkgs must be between 0 and 1")

    total_mix = df["share_of_pkgs"].sum()
    if abs(total_mix - 1.0) > 0.01:
        raise ValueError(f"share_of_pkgs must sum to 1.0, got {total_mix:.4f}")


def _validate_scenarios(df: pd.DataFrame) -> None:
    """Validate scenarios sheet."""
    required = ["scenario_id", "year", "day_type"]
    _check_required_columns(df, required, "scenarios")

    if df["scenario_id"].duplicated().any():
        dupes = df[df["scenario_id"].duplicated()]["scenario_id"].tolist()
        raise ValueError(f"Duplicate scenario_ids: {dupes}")

    valid_day_types = {"peak", "average", "trough"}
    invalid = set(df["day_type"].str.lower()) - valid_day_types
    if invalid:
        raise ValueError(f"Invalid day_types: {invalid}. Must be one of {valid_day_types}")


def _validate_run_settings(df: pd.DataFrame) -> None:
    """Validate run_settings sheet."""
    if "key" not in df.columns or "value" not in df.columns:
        raise ValueError("run_settings must have 'key' and 'value' columns")


def _validate_feasible_paths(df: pd.DataFrame) -> None:
    """
    Validate feasible_paths sheet from SLA model.

    This is the core input from the SLA model containing pre-enumerated paths.
    """
    _check_required_columns(df, FEASIBLE_PATHS_REQUIRED_COLUMNS, "feasible_paths")

    if df.empty:
        raise ValueError("feasible_paths cannot be empty")

    # Validate path_type values
    invalid_path_types = set(df["path_type"].str.lower().unique()) - VALID_PATH_TYPES
    if invalid_path_types:
        raise ValueError(
            f"Invalid path_type values in feasible_paths: {invalid_path_types}. "
            f"Must be one of {VALID_PATH_TYPES}"
        )

    # Validate sort_level values
    invalid_sort_levels = set(df["sort_level"].str.lower().unique()) - VALID_SORT_LEVELS
    if invalid_sort_levels:
        raise ValueError(
            f"Invalid sort_level values in feasible_paths: {invalid_sort_levels}. "
            f"Must be one of {VALID_SORT_LEVELS}"
        )

    # Validate numeric columns
    numeric_cols = ["total_path_miles", "direct_miles", "tit_hours", "pkgs_mm", "pkgs_zs", "pkgs_di"]
    for col in numeric_cols:
        if col in df.columns:
            if df[col].isna().all():
                raise ValueError(f"feasible_paths column '{col}' is all null")

    # Validate zone
    if "zone" in df.columns:
        if (df["zone"] < 0).any():
            raise ValueError("zone must be non-negative")

    # Validate node_1 equals origin
    if (df["node_1"] != df["origin"]).any():
        mismatches = df[df["node_1"] != df["origin"]][["origin", "node_1"]].head()
        raise ValueError(f"node_1 must equal origin. Mismatches: {mismatches.to_dict()}")


def _validate_scenario_feasible_paths_match(
    scenarios_df: pd.DataFrame,
    feasible_paths_df: pd.DataFrame
) -> None:
    """
    HARD ERROR: All scenario_ids must exist in feasible_paths.

    This ensures the sort model only runs on scenarios that have
    pre-enumerated paths from the SLA model.
    """
    scenario_ids = set(scenarios_df["scenario_id"].unique())
    feasible_path_ids = get_feasible_paths_scenario_ids(feasible_paths_df)

    missing = scenario_ids - feasible_path_ids
    if missing:
        raise ValueError(
            f"Scenarios not found in feasible_paths: {sorted(missing)}. "
            f"Available scenario_ids in feasible_paths: {sorted(feasible_path_ids)}"
        )


def _validate_referential_integrity(dfs: Dict[str, pd.DataFrame]) -> None:
    """Validate that feasible_paths reference valid facilities."""
    facilities = dfs["facilities"]
    feasible_paths = dfs["feasible_paths"]

    valid_facilities = set(facilities["facility_name"])

    # Check origin and dest
    for col in ["origin", "dest"]:
        invalid = set(feasible_paths[col].unique()) - valid_facilities
        if invalid:
            raise ValueError(
                f"feasible_paths '{col}' references unknown facilities: {sorted(invalid)[:10]}"
            )

    # Check node columns
    node_cols = ["node_1", "node_2", "node_3", "node_4", "node_5"]
    for col in node_cols:
        if col not in feasible_paths.columns:
            continue
        non_null = feasible_paths[col].dropna()
        if non_null.empty:
            continue
        invalid = set(non_null.unique()) - valid_facilities
        if invalid:
            raise ValueError(
                f"feasible_paths '{col}' references unknown facilities: {sorted(invalid)[:10]}"
            )


def _check_required_columns(df: pd.DataFrame, required: list, sheet_name: str) -> None:
    """Check that all required columns are present."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {sheet_name}: {sorted(missing)}")
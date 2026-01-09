"""
Feasible Paths Loader Module

Loads and filters pre-enumerated paths from SLA model output.
Handles conversion from node columns to path lists and splits
middle-mile vs direct injection flows.
"""

import pandas as pd
from typing import Tuple, List, Dict

from .config import (
    FEASIBLE_PATHS_REQUIRED_COLUMNS,
    VALID_PATH_TYPES,
    VALID_SORT_LEVELS,
)


def load_and_filter_feasible_paths(
        feasible_paths_df: pd.DataFrame,
        run_settings: Dict,
        scenario_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and filter feasible paths for a specific scenario.

    Filtering chain:
    1. scenario_id match
    2. sla_met = True
    3. uses_only_active_arcs = True (if allow_inactive_arcs = False)

    Args:
        feasible_paths_df: Raw feasible_paths from workbook
        run_settings: Dictionary of run settings
        scenario_id: Scenario to filter for

    Returns:
        Tuple of (middle_mile_candidates, direct_injection_summary)
        - middle_mile_candidates: DataFrame with pkgs_day, zone, path_nodes, etc.
        - direct_injection_summary: DataFrame with dest, dir_pkgs_day, zone=0
    """
    df = feasible_paths_df.copy()

    # Filter by scenario
    df = df[df["scenario_id"] == scenario_id].copy()
    if df.empty:
        raise ValueError(f"No paths found for scenario_id: {scenario_id}")

    initial_count = len(df)
    print(f"  Scenario {scenario_id}: {initial_count:,} total paths")

    # Filter by sla_met
    df = df[df["sla_met"] == True].copy()
    sla_count = len(df)
    print(f"  After sla_met=True: {sla_count:,} paths ({sla_count / initial_count * 100:.1f}%)")

    # Filter by uses_only_active_arcs (if required)
    allow_inactive = run_settings.get("allow_inactive_arcs", False)
    if not allow_inactive:
        df = df[df["uses_only_active_arcs"] == True].copy()
        active_count = len(df)
        print(f"  After active_arcs filter: {active_count:,} paths ({active_count / initial_count * 100:.1f}%)")

    if df.empty:
        raise ValueError(f"No paths remain after filtering for scenario: {scenario_id}")

    # Convert node columns to path_nodes list
    df = _convert_path_nodes(df)

    # Split into middle-mile and direct injection
    middle_mile, direct_injection = _split_flow_types(df)

    # Build direct injection summary
    direct_summary = _build_direct_injection_summary(direct_injection)

    return middle_mile, direct_summary


def _convert_path_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert node_1...node_5 columns to path_nodes list and path_str.

    Adds columns:
    - path_nodes: List of facility names in order
    - path_str: String representation (e.g., "A->B->C")
    """
    node_cols = ["node_1", "node_2", "node_3", "node_4", "node_5"]

    def extract_nodes(row):
        nodes = []
        for col in node_cols:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    nodes.append(str(val).strip())
        return nodes

    df["path_nodes"] = df.apply(extract_nodes, axis=1)
    df["path_str"] = df["path_nodes"].apply(lambda x: "->".join(x))

    return df


def _split_flow_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split paths into middle-mile and direct injection.

    Middle-mile: pkgs_mm + pkgs_zs (both use same path selection logic)
    Direct injection: pkgs_di (Zone 0, no linehaul)

    Returns:
        (middle_mile_df, direct_injection_df)
    """
    # Calculate combined middle-mile volume
    df["pkgs_day"] = df["pkgs_mm"].fillna(0) + df["pkgs_zs"].fillna(0)
    df["zone"] = df["zone_mm_zs"].fillna(0).astype(int)

    # Middle-mile: paths with pkgs_day > 0 and not direct_injection type
    # (direct_injection paths have their own handling)
    middle_mile = df[
        (df["pkgs_day"] > 0) &
        (df["path_type"].str.lower() != "direct_injection")
        ].copy()

    # Direct injection: either path_type is direct_injection OR pkgs_di > 0
    direct_injection = df[
        (df["path_type"].str.lower() == "direct_injection") |
        (df["pkgs_di"].fillna(0) > 0)
        ].copy()

    return middle_mile, direct_injection


def _build_direct_injection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate direct injection packages by destination.

    Direct injection is Zone 0 - packages delivered without middle-mile transport.
    """
    if df.empty:
        return pd.DataFrame(columns=["dest", "dir_pkgs_day", "zone"])

    # Aggregate by destination
    summary = df.groupby("dest").agg({
        "pkgs_di": "sum"
    }).reset_index()

    summary = summary.rename(columns={"pkgs_di": "dir_pkgs_day"})
    summary["zone"] = 0  # Direct injection is always Zone 0

    return summary[summary["dir_pkgs_day"] > 0].copy()


def validate_feasible_paths_columns(df: pd.DataFrame) -> None:
    """
    Validate that feasible_paths has required columns from SLA model.

    Called before main validation to catch structural issues early.
    """
    missing = set(FEASIBLE_PATHS_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"feasible_paths missing required columns: {sorted(missing)}. "
            f"Ensure SLA model output includes all required fields."
        )


def validate_path_structure(df: pd.DataFrame) -> None:
    """
    Validate path structure consistency.

    Checks:
    - path_nodes[0] == origin
    - path_nodes[-1] == dest
    - path_type matches node count
    """
    errors = []

    for idx, row in df.iterrows():
        nodes = row["path_nodes"]
        origin = row["origin"]
        dest = row["dest"]
        path_type = row["path_type"]

        # Check origin matches first node
        if nodes and nodes[0] != origin:
            errors.append(f"Row {idx}: path_nodes[0]={nodes[0]} != origin={origin}")

        # Check dest matches last node
        if nodes and nodes[-1] != dest:
            errors.append(f"Row {idx}: path_nodes[-1]={nodes[-1]} != dest={dest}")

        # Check path_type matches node count
        expected_counts = {
            "direct_injection": 1,
            "od_mm": 1,  # O=D middle-mile, single node
            "2_touch": 2,
            "3_touch": 3,
            "4_touch": 4,
            "5_touch": 5,
        }

        expected = expected_counts.get(path_type.lower())
        if expected and len(nodes) != expected:
            errors.append(
                f"Row {idx}: path_type={path_type} expects {expected} nodes, got {len(nodes)}"
            )

        if len(errors) >= 10:
            errors.append("... (additional errors truncated)")
            break

    if errors:
        raise ValueError(
            f"Path structure validation failed:\n" + "\n".join(errors)
        )


def get_unique_od_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique origin-destination pairs with total volume.

    Returns DataFrame with origin, dest, total_pkgs_day.
    """
    return df.groupby(["origin", "dest"]).agg({
        "pkgs_day": "sum"
    }).reset_index().rename(columns={"pkgs_day": "total_pkgs_day"})


def get_regional_sort_hub_constraints(
        df: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[Tuple[str, str], List[str]]:
    """
    Identify (hub, dest) pairs that require sort consistency constraints.

    For regional sort paths, tracks which dest_sort_levels are possible
    for each (regional_hub, dest) combination.

    Args:
        df: Filtered middle-mile candidates
        facilities: Facilities DataFrame with regional_sort_hub column

    Returns:
        Dict mapping (hub, dest) to list of possible dest_sort_levels
    """
    fac_lookup = facilities.set_index("facility_name")

    # Build dest -> regional_hub mapping
    dest_to_hub = {}
    for facility in fac_lookup.index:
        if "regional_sort_hub" in fac_lookup.columns:
            hub = fac_lookup.at[facility, "regional_sort_hub"]
            if pd.notna(hub) and str(hub).strip():
                dest_to_hub[facility] = str(hub).strip()
            else:
                dest_to_hub[facility] = facility
        else:
            dest_to_hub[facility] = facility

    # Find (hub, dest) pairs and their dest_sort_levels
    hub_dest_levels: Dict[Tuple[str, str], set] = {}

    for _, row in df.iterrows():
        dest = row["dest"]
        path_nodes = row["path_nodes"]
        sort_level = row["sort_level"]
        dest_sort_level = row.get("dest_sort_level")

        if dest not in dest_to_hub:
            continue

        regional_hub = dest_to_hub[dest]

        # Check if path routes through regional hub
        if regional_hub not in path_nodes:
            continue

        key = (regional_hub, dest)

        # Determine effective dest_sort_level
        if sort_level == "region":
            eff_dsl = dest_sort_level if dest_sort_level else "market"
        elif sort_level in ["market", "sort_group"]:
            eff_dsl = sort_level
        else:
            continue

        if key not in hub_dest_levels:
            hub_dest_levels[key] = set()
        hub_dest_levels[key].add(eff_dsl)

    # Convert sets to lists
    return {k: sorted(v) for k, v in hub_dest_levels.items()}


def summarize_candidate_paths(df: pd.DataFrame) -> str:
    """
    Generate summary statistics for candidate paths.

    Returns formatted string for logging.
    """
    lines = []

    total_paths = len(df)
    total_volume = df["pkgs_day"].sum()
    unique_ods = df.groupby(["origin", "dest"]).ngroups

    lines.append(f"  Candidate paths: {total_paths:,}")
    lines.append(f"  Unique OD pairs: {unique_ods:,}")
    lines.append(f"  Total volume: {total_volume:,.0f} packages")
    lines.append(f"  Avg paths per OD: {total_paths / max(unique_ods, 1):.1f}")

    # Path type distribution
    if "path_type" in df.columns:
        type_dist = df.groupby("path_type")["pkgs_day"].sum()
        lines.append("  Path types:")
        for pt, vol in type_dist.items():
            pct = vol / total_volume * 100 if total_volume > 0 else 0
            lines.append(f"    {pt}: {vol:,.0f} ({pct:.1f}%)")

    # Sort level distribution
    if "sort_level" in df.columns:
        sl_dist = df.groupby("sort_level")["pkgs_day"].sum()
        lines.append("  Sort levels:")
        for sl, vol in sl_dist.items():
            pct = vol / total_volume * 100 if total_volume > 0 else 0
            lines.append(f"    {sl}: {vol:,.0f} ({pct:.1f}%)")

    return "\n".join(lines)
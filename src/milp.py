"""
MILP Network Optimization Module

Solves minimum-cost network configuration using arc-pooled Mixed Integer Linear Programming.
Uses pre-enumerated feasible paths from SLA model.

Key Features:
- Paths come from feasible_paths (SLA model output)
- Sort level optimization (region/market/sort_group)
- Regional sort consistency constraint
- Container persistence through crossdocks
- Transport costs from mileage_bands

Sort Level Business Rules (from requirements):
1. Hub is regional hub for dest AND hub != dest → minimum market sort (can't do region)
2. Hub is the destination itself → sort_group required (no downstream to sort)
3. Hub is NOT regional hub for dest → region allowed (containers go to regional hub for re-sort)
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

from .config import CostParameters, LoadStrategy, OptimizationConstants


MAX_SAFE_INT = 2 ** 31 - 1


def safe_int_cost(value: float, context: str = "") -> int:
    """Convert float to int with overflow check."""
    int_val = int(round(value))
    if abs(int_val) > MAX_SAFE_INT:
        raise ValueError(f"Cost overflow in {context}: {value:,.0f}")
    return int_val


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return numerator / denominator if denominator != 0 else default


def solve_network_optimization(
    candidates: pd.DataFrame,
    facilities: pd.DataFrame,
    mileage_bands: pd.DataFrame,
    package_mix: pd.DataFrame,
    container_params: pd.DataFrame,
    cost_params: CostParameters,
    global_strategy: LoadStrategy,
    enable_sort_optimization: bool,
    scenario_row: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Optional[pd.DataFrame]]:
    """
    Solve network optimization using pre-enumerated paths.

    The model selects one path per OD pair, where each path has:
    - Physical routing (node sequence)
    - Sort level at origin (region/market/sort_group)
    - For region sort: dest_sort_level at regional hub (market/sort_group)

    Args:
        candidates: Filtered paths from load_feasible_paths
        facilities: Facility data with regional_sort_hub column
        mileage_bands: Transport cost rates
        package_mix: Package size distribution
        container_params: Container specifications
        cost_params: Processing cost parameters
        global_strategy: Load strategy (container/fluid)
        enable_sort_optimization: Whether to add sort constraints
        scenario_row: Scenario data for capacity overrides

    Returns:
        Tuple of (selected_paths, arc_summary, kpis, sort_summary)
    """
    cand = candidates.reset_index(drop=True).copy()
    print(f"    Strategy: {global_strategy.value}")
    print(f"    Sort optimization: {'ENABLED' if enable_sort_optimization else 'DISABLED'}")
    print(f"    Candidate paths: {len(cand):,}")

    if cand.empty:
        print("    ERROR: No candidate paths received by solver")
        return pd.DataFrame(), pd.DataFrame(), {"solver_status": "NO_CANDIDATES"}, pd.DataFrame()

    # Debug: show columns available
    print(f"    DEBUG: Candidate columns: {list(cand.columns)}")

    # Build facility lookup
    fac_lookup = facilities.set_index("facility_name")

    path_keys = list(cand.index)
    w_cube = _weighted_pkg_cube(package_mix)

    # Build arc structures
    arc_index_map, arc_meta, path_arcs, path_od_data = _build_arc_structures(
        cand, facilities, mileage_bands, path_keys
    )

    # Group paths by OD pair
    groups = cand.groupby(["origin", "dest"]).indices
    print(f"    Network: {len(arc_meta)} arcs, {len(path_keys)} paths, {len(groups)} OD pairs")

    # Debug: show sort level options per OD
    _debug_sort_level_options(cand, groups)

    # Build model
    model = cp_model.CpModel()
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    arc_pkgs = {a: model.NewIntVar(0, 10_000_000, f"arc_pkgs_{a}") for a in range(len(arc_meta))}
    arc_trucks = {a: model.NewIntVar(0, 10_000, f"arc_trucks_{a}") for a in range(len(arc_meta))}

    # Constraint: Exactly one path per OD
    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # Arc capacity constraints
    _add_arc_capacity_constraints(
        model, x, arc_pkgs, arc_trucks, arc_meta, path_arcs,
        path_od_data, path_keys, package_mix, container_params
    )

    # Sort constraints (if enabled)
    if enable_sort_optimization:
        sort_cap_count = _add_sort_capacity_constraints(
            model, x, path_keys, path_od_data, fac_lookup, cost_params, scenario_row
        )
        consistency_count = _add_regional_sort_consistency_constraints(
            model, x, path_keys, path_od_data, fac_lookup
        )

        # If infeasible, we'll retry without constraints to diagnose
    else:
        print("    Sort optimization DISABLED - no sort constraints added")

    # Objective: minimize total cost (transport + processing)
    cost_terms = []

    # Transport costs (per arc)
    for a_idx in range(len(arc_meta)):
        truck_cost = safe_int_cost(arc_meta[a_idx]["cost_per_truck"], f"arc_{a_idx}")
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # Processing costs (per path)
    for i in path_keys:
        path_data = path_od_data[i]
        volume = int(round(path_data['pkgs_day']))
        proc_cost = _calculate_processing_cost(
            path_data, fac_lookup, package_mix, container_params, cost_params
        )
        cost_terms.append(x[i] * safe_int_cost(proc_cost * volume, f"path_{i}"))

    model.Minimize(sum(cost_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = OptimizationConstants.MAX_SOLVER_TIME_SECONDS
    solver.parameters.num_search_workers = OptimizationConstants.NUM_SOLVER_WORKERS

    print(f"    Solving MILP...")
    status = solver.Solve(model)

    status_msg = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, f"Status_{status}")

    print(f"    Result: {status_msg}, Time: {solver.WallTime():.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n    DEBUG: Solver failed with status {status_msg}")
        print(f"    DEBUG: Total paths in model: {len(path_keys)}")
        print(f"    DEBUG: Total OD pairs: {len(groups)}")
        print(f"    DEBUG: Total arcs: {len(arc_meta)}")

        # Check if any OD has zero paths
        empty_ods = [(od, len(idxs)) for od, idxs in groups.items() if len(idxs) == 0]
        if empty_ods:
            print(f"    DEBUG: OD pairs with zero paths: {empty_ods[:5]}")

        # Sample some path data
        if path_keys:
            sample_idx = path_keys[0]
            print(f"    DEBUG: Sample path data: {path_od_data[sample_idx]}")

        # If sort optimization was enabled and we got INFEASIBLE, diagnose
        if enable_sort_optimization and status == cp_model.INFEASIBLE:
            print(f"\n    DIAGNOSTIC: Retrying WITHOUT sort constraints to identify cause...")

            # Rebuild model without sort constraints
            model2 = cp_model.CpModel()
            x2 = {i: model2.NewBoolVar(f"x_{i}") for i in path_keys}
            arc_pkgs2 = {a: model2.NewIntVar(0, 10_000_000, f"arc_pkgs_{a}") for a in range(len(arc_meta))}
            arc_trucks2 = {a: model2.NewIntVar(0, 10_000, f"arc_trucks_{a}") for a in range(len(arc_meta))}

            # One path per OD
            for group_name, idxs in groups.items():
                model2.Add(sum(x2[i] for i in idxs) == 1)

            # Arc capacity only
            _add_arc_capacity_constraints(
                model2, x2, arc_pkgs2, arc_trucks2, arc_meta, path_arcs,
                path_od_data, path_keys, package_mix, container_params
            )

            # Simple objective
            cost_terms2 = []
            for a_idx in range(len(arc_meta)):
                truck_cost = safe_int_cost(arc_meta[a_idx]["cost_per_truck"], f"arc_{a_idx}")
                cost_terms2.append(arc_trucks2[a_idx] * truck_cost)
            for i in path_keys:
                path_data = path_od_data[i]
                volume = int(round(path_data['pkgs_day']))
                proc_cost = _calculate_processing_cost(
                    path_data, fac_lookup, package_mix, container_params, cost_params
                )
                cost_terms2.append(x2[i] * safe_int_cost(proc_cost * volume, f"path_{i}"))
            model2.Minimize(sum(cost_terms2))

            solver2 = cp_model.CpSolver()
            solver2.parameters.max_time_in_seconds = 60
            status2 = solver2.Solve(model2)

            status2_msg = {
                cp_model.OPTIMAL: "OPTIMAL",
                cp_model.FEASIBLE: "FEASIBLE",
                cp_model.INFEASIBLE: "INFEASIBLE",
            }.get(status2, f"Status_{status2}")

            print(f"    DIAGNOSTIC: Without sort constraints: {status2_msg}")

            if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                print(f"    CONCLUSION: Sort constraints are causing infeasibility")
                print(f"    ACTION: Check max_sort_points_capacity in facilities or scenario")

                # Analyze what the unconstrained solution chose
                chosen_sort_levels = {}
                for i in path_keys:
                    if solver2.Value(x2[i]) == 1:
                        sl = path_od_data[i]['sort_level']
                        chosen_sort_levels[sl] = chosen_sort_levels.get(sl, 0) + 1
                print(f"    DIAGNOSTIC: Unconstrained solution sort levels: {chosen_sort_levels}")

                # Calculate sort points that would be used per facility (UNIQUE destinations)
                facility_sort_points = {}
                facility_dest_combos = {}  # facility -> set of (sort_target, eff_sort)
                pts_per_dest = cost_params.sort_points_per_destination

                for i in path_keys:
                    if solver2.Value(x2[i]) == 1:
                        origin = path_od_data[i]['origin']
                        dest = path_od_data[i]['dest']
                        sl = path_od_data[i]['sort_level']

                        # Determine effective sort target
                        if sl == 'region':
                            if dest in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
                                regional_hub = fac_lookup.at[dest, 'regional_sort_hub']
                                if pd.isna(regional_hub) or regional_hub == '':
                                    regional_hub = dest
                            else:
                                regional_hub = dest

                            if origin == regional_hub:
                                eff_sort = 'market'
                                sort_target = dest
                            else:
                                eff_sort = 'region'
                                sort_target = regional_hub
                        else:
                            eff_sort = sl
                            sort_target = dest

                        if origin not in facility_dest_combos:
                            facility_dest_combos[origin] = set()
                        facility_dest_combos[origin].add((sort_target, eff_sort))

                # Now calculate points from unique combos
                for origin, combos in facility_dest_combos.items():
                    total_pts = 0
                    for (sort_target, eff_sort) in combos:
                        if eff_sort == 'sort_group':
                            if sort_target in fac_lookup.index and 'last_mile_sort_groups_count' in fac_lookup.columns:
                                groups_count = fac_lookup.at[sort_target, 'last_mile_sort_groups_count']
                                pts = pts_per_dest * (int(groups_count) if pd.notna(groups_count) and groups_count > 0 else 1)
                            else:
                                pts = pts_per_dest
                        elif eff_sort == 'market':
                            pts = pts_per_dest
                        elif eff_sort == 'region':
                            pts = 1
                        else:
                            pts = pts_per_dest
                        total_pts += pts
                    facility_sort_points[origin] = total_pts

                # Compare to capacities
                print(f"\n    DIAGNOSTIC: Sort point usage vs capacity:")
                exceeded_facilities = []
                for facility in sorted(facility_sort_points.keys()):
                    used = facility_sort_points[facility]
                    cap = None
                    if facility in fac_lookup.index and 'max_sort_points_capacity' in fac_lookup.columns:
                        cap = fac_lookup.at[facility, 'max_sort_points_capacity']
                    cap_str = f"{int(cap)}" if pd.notna(cap) and cap > 0 else "unlimited"
                    exceeded = "*** EXCEEDED ***" if (pd.notna(cap) and cap > 0 and used > cap) else ""
                    if exceeded:
                        exceeded_facilities.append(facility)
                    print(f"      {facility}: {used:.0f} used / {cap_str} capacity {exceeded}")

                # Calculate minimum feasible sort points per facility
                print(f"\n    DIAGNOSTIC: Minimum feasible sort points per facility:")
                for facility in exceeded_facilities:
                    # Get facility capacity
                    fac_cap = None
                    if facility in fac_lookup.index and 'max_sort_points_capacity' in fac_lookup.columns:
                        fac_cap = fac_lookup.at[facility, 'max_sort_points_capacity']
                    fac_cap = int(fac_cap) if pd.notna(fac_cap) and fac_cap > 0 else 0

                    # Get all OD pairs from this facility
                    fac_paths = [i for i in path_keys if path_od_data[i]['origin'] == facility]
                    od_pairs = set((path_od_data[i]['origin'], path_od_data[i]['dest']) for i in fac_paths)

                    # DIAGNOSTIC: Count actual region paths
                    region_path_count = sum(1 for i in fac_paths if path_od_data[i]['sort_level'] == 'region')
                    region_dests = set(path_od_data[i]['dest'] for i in fac_paths if path_od_data[i]['sort_level'] == 'region')
                    print(f"\n      {facility}: DEBUG - {region_path_count} region paths to {len(region_dests)} destinations")

                    # For each dest, find minimum sort points available
                    min_pts_by_dest = {}
                    for i in fac_paths:
                        dest = path_od_data[i]['dest']
                        sl = path_od_data[i]['sort_level']

                        # Calculate sort points for this sort level
                        if sl == 'region':
                            # Get regional hub
                            if dest in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
                                hub = fac_lookup.at[dest, 'regional_sort_hub']
                                if pd.isna(hub) or hub == '':
                                    hub = dest
                            else:
                                hub = dest
                            # Region sorts share hub - will be counted later
                            pts = ('region', hub, 1)
                        elif sl == 'sort_group':
                            if dest in fac_lookup.index and 'last_mile_sort_groups_count' in fac_lookup.columns:
                                sort_groups_count = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                                sort_groups_count = int(sort_groups_count) if pd.notna(sort_groups_count) and sort_groups_count > 0 else 1
                            else:
                                sort_groups_count = 1
                            pts = ('sort_group', dest, cost_params.sort_points_per_destination * sort_groups_count)
                        else:  # market
                            pts = ('market', dest, cost_params.sort_points_per_destination)

                        # Track minimum per destination
                        if dest not in min_pts_by_dest:
                            min_pts_by_dest[dest] = []
                        min_pts_by_dest[dest].append(pts)

                    # Calculate theoretical minimum
                    total_min = 0
                    region_hubs = set()
                    non_region_dests = []

                    for dest, options in min_pts_by_dest.items():
                        # Check if region is available
                        region_opts = [o for o in options if o[0] == 'region']
                        if region_opts:
                            # Use region (just track the hub)
                            region_hubs.add(region_opts[0][1])
                        else:
                            # Must use market or sort_group - pick min
                            min_pts = min(o[2] for o in options)
                            non_region_dests.append((dest, min_pts))

                    # Region hubs cost 1 pt each
                    region_pts = len(region_hubs)
                    # Non-region dests cost their min
                    non_region_pts = sum(pts for _, pts in non_region_dests)
                    total_min = region_pts + non_region_pts

                    # Flag feasibility issue
                    feasibility_flag = ""
                    if total_min > fac_cap:
                        feasibility_flag = " *** INFEASIBLE: min > cap ***"
                    elif total_min == fac_cap:
                        feasibility_flag = " *** TIGHT: min == cap ***"

                    dests_with_region = len([d for d in min_pts_by_dest if any(o[0]=='region' for o in min_pts_by_dest[d])])
                    print(f"      {facility}: {len(od_pairs)} destinations, CAPACITY: {fac_cap}")
                    print(f"        - {dests_with_region} with region available → {region_pts} regional hubs")
                    print(f"        - {len(non_region_dests)} without region → {non_region_pts:.0f} pts (market min)")
                    print(f"        - MINIMUM FEASIBLE: {total_min:.0f} pts{feasibility_flag}")

                    # Show sample destinations WITH and WITHOUT region
                    if dests_with_region > 0:
                        sample_with = [d for d in min_pts_by_dest if any(o[0]=='region' for o in min_pts_by_dest[d])][:3]
                        print(f"        - Sample WITH region: {sample_with}")
                    if non_region_dests:
                        sample_without = [d for d, _ in non_region_dests[:3]]
                        print(f"        - Sample WITHOUT region: {sample_without}")

                # TEST: Try with ONLY sort capacity constraints (no regional consistency)
                print(f"\n    DIAGNOSTIC: Testing with ONLY sort capacity (no regional consistency)...")
                model3 = cp_model.CpModel()
                x3 = {i: model3.NewBoolVar(f"x_{i}") for i in path_keys}
                arc_pkgs3 = {a: model3.NewIntVar(0, 10_000_000, f"arc_pkgs_{a}") for a in range(len(arc_meta))}
                arc_trucks3 = {a: model3.NewIntVar(0, 10_000, f"arc_trucks_{a}") for a in range(len(arc_meta))}

                for group_name, idxs in groups.items():
                    model3.Add(sum(x3[i] for i in idxs) == 1)

                _add_arc_capacity_constraints(
                    model3, x3, arc_pkgs3, arc_trucks3, arc_meta, path_arcs,
                    path_od_data, path_keys, package_mix, container_params
                )

                # Add ONLY sort capacity constraints (no regional consistency)
                _add_sort_capacity_constraints(
                    model3, x3, path_keys, path_od_data, fac_lookup, cost_params, scenario_row
                )

                cost_terms3 = []
                for a_idx in range(len(arc_meta)):
                    truck_cost = safe_int_cost(arc_meta[a_idx]["cost_per_truck"], f"arc_{a_idx}")
                    cost_terms3.append(arc_trucks3[a_idx] * truck_cost)
                for i in path_keys:
                    path_data = path_od_data[i]
                    volume = int(round(path_data['pkgs_day']))
                    proc_cost = _calculate_processing_cost(
                        path_data, fac_lookup, package_mix, container_params, cost_params
                    )
                    cost_terms3.append(x3[i] * safe_int_cost(proc_cost * volume, f"path_{i}"))
                model3.Minimize(sum(cost_terms3))

                solver3 = cp_model.CpSolver()
                solver3.parameters.max_time_in_seconds = 120
                status3 = solver3.Solve(model3)

                status3_msg = {
                    cp_model.OPTIMAL: "OPTIMAL",
                    cp_model.FEASIBLE: "FEASIBLE",
                    cp_model.INFEASIBLE: "INFEASIBLE",
                }.get(status3, f"Status_{status3}")

                print(f"    DIAGNOSTIC: With ONLY sort capacity: {status3_msg}")

                if status3 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    print(f"    ROOT CAUSE: Regional sort consistency constraint is making model infeasible")

                    # Show what solution capacity-only found
                    cap_only_sort_levels = {}
                    for i in path_keys:
                        if solver3.Value(x3[i]) == 1:
                            sl = path_od_data[i]['sort_level']
                            cap_only_sort_levels[sl] = cap_only_sort_levels.get(sl, 0) + 1
                    print(f"    DIAGNOSTIC: Capacity-only solution: {cap_only_sort_levels}")
                else:
                    print(f"    ROOT CAUSE: Sort capacity constraints alone make model infeasible")
                    print(f"    This means minimum feasible calculation is wrong or capacity too low")
            else:
                print(f"    CONCLUSION: Problem is in base model (OD coverage or arc constraints)")

        return pd.DataFrame(), pd.DataFrame(), {"solver_status": status_msg}, pd.DataFrame()

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    print(f"    Selected {len(chosen_idx)} optimal paths")

    # Debug: show selected sort levels
    _debug_selected_sort_levels(chosen_idx, path_od_data)

    selected_paths = _build_selected_paths(
        chosen_idx, path_od_data, path_arcs, arc_meta,
        arc_trucks, arc_pkgs, cost_params, fac_lookup,
        package_mix, container_params, solver
    )

    raw_trailer_cube = _get_raw_trailer_cube(container_params)
    arc_summary = _build_arc_summary(
        arc_meta, arc_pkgs, arc_trucks, w_cube,
        raw_trailer_cube, container_params, solver
    )

    kpis = _calculate_kpis(selected_paths, arc_summary, raw_trailer_cube)
    sort_summary = _build_sort_summary(selected_paths, fac_lookup, cost_params) if enable_sort_optimization else pd.DataFrame()

    # Calculate and print sort point usage
    if enable_sort_optimization:
        sort_point_usage = _calculate_sort_point_usage(selected_paths, cand, fac_lookup, cost_params)
        if not sort_point_usage.empty:
            print(f"\n    Sort Point Usage:")
            for _, row in sort_point_usage.iterrows():
                min_str = f"min:{row['min_feasible']:.0f}" if pd.notna(row['min_feasible']) else ""
                print(f"      {row['facility']}: {row['actual_used']:.0f}/{row['capacity']} ({row['utilization']:.0%}) {min_str}")

    print(f"    Cost: ${selected_paths['total_cost'].sum():,.2f}")
    return selected_paths, arc_summary, kpis, sort_summary


def _debug_sort_level_options(cand: pd.DataFrame, groups: dict) -> None:
    """Debug: print sort level options for sample OD pairs."""
    print("\n    DEBUG: Sort level options (sample):")
    sample_count = 0
    for (origin, dest), idxs in groups.items():
        if sample_count >= 3:
            break
        sort_levels = cand.loc[list(idxs), 'sort_level'].unique()
        dest_sort_levels = cand.loc[list(idxs), 'dest_sort_level'].dropna().unique() if 'dest_sort_level' in cand.columns else []
        print(f"      {origin} → {dest}: sort_levels={list(sort_levels)}, dest_sort_levels={list(dest_sort_levels)}")
        sample_count += 1
    print()


def _debug_selected_sort_levels(chosen_idx: list, path_od_data: dict) -> None:
    """Debug: print selected sort levels."""
    sort_level_counts = {}
    for i in chosen_idx:
        sl = path_od_data[i]['sort_level']
        sort_level_counts[sl] = sort_level_counts.get(sl, 0) + 1

    print(f"    DEBUG: Selected sort levels: {sort_level_counts}")


def _build_arc_structures(cand, facilities, mileage_bands, path_keys):
    """Build arc index map and metadata from candidate paths."""
    arc_index_map = {}
    arc_meta = []
    path_arcs = {}
    path_od_data = {}

    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)

    for i in path_keys:
        r = cand.loc[i]
        nodes = r["path_nodes"]

        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'path_str': r.get("path_str", "->".join(nodes)),
            'path_type': r.get("path_type", "2_touch"),
            'path_nodes': nodes,
            'sort_level': str(r.get("sort_level", "market")).lower(),
            'dest_sort_level': str(r.get("dest_sort_level", "")).lower() if pd.notna(r.get("dest_sort_level")) else None,
            'zone': int(r.get("zone", -1)),
            'tit_hours': float(r.get("tit_hours", 0)),
            'total_path_miles': float(r.get("total_path_miles", 0)),
            'direct_miles': float(r.get("direct_miles", 0)),
        }

        arc_ids = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            if u == v:
                continue
            key = (u, v)
            if key not in arc_index_map:
                lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
                lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
                raw_dist = _haversine_miles(lat1, lon1, lat2, lon2)
                fixed, var, circuit, mph = _band_lookup(raw_dist, mileage_bands)
                dist = raw_dist * circuit
                arc_index_map[key] = len(arc_meta)
                arc_meta.append({
                    "from": u,
                    "to": v,
                    "distance_miles": dist,
                    "cost_per_truck": fixed + var * dist,
                    "mph": mph
                })
            arc_ids.append(arc_index_map[key])
        path_arcs[i] = arc_ids

    return arc_index_map, arc_meta, path_arcs, path_od_data


def _add_arc_capacity_constraints(model, x, arc_pkgs, arc_trucks, arc_meta, path_arcs,
                                   path_od_data, path_keys, package_mix, container_params):
    """Add constraints linking packages to truck requirements."""
    w_cube = _weighted_pkg_cube(package_mix)
    container_cap = _get_container_capacity(container_params) * _get_containers_per_truck(container_params)
    cap_scaled = int(container_cap * OptimizationConstants.CUBE_SCALE_FACTOR)
    w_scaled = int(w_cube * OptimizationConstants.CUBE_SCALE_FACTOR)

    for a_idx in range(len(arc_meta)):
        terms = [
            int(round(path_od_data[i]['pkgs_day'])) * x[i]
            for i in path_keys if a_idx in path_arcs[i]
        ]
        model.Add(arc_pkgs[a_idx] == sum(terms) if terms else 0)
        model.Add(arc_trucks[a_idx] * cap_scaled >= arc_pkgs[a_idx] * w_scaled)

        arc_has = model.NewBoolVar(f"arc_has_{a_idx}")
        model.Add(arc_pkgs[a_idx] <= OptimizationConstants.BIG_M * arc_has)
        model.Add(arc_trucks[a_idx] >= arc_has)


def _add_sort_capacity_constraints(model, x, path_keys, path_od_data, fac_lookup,
                                    cost_params, scenario_row):
    """
    Add sort capacity constraints at hub facilities.

    Sort points consumed = unique destinations × points per destination

    For each origin hub, count UNIQUE destinations (not paths) and their sort point requirements:
    - sort_group: pts_per_dest × last_mile_sort_groups_count for that dest
    - market: pts_per_dest (one market per dest)
    - region: 1 (one regional hub regardless of how many dests behind it)
    """
    pts_per_dest = cost_params.sort_points_per_destination

    # Check for scenario-level capacity override
    cap_override = None
    if scenario_row is not None and 'max_sort_points_capacity' in scenario_row.index:
        val = scenario_row['max_sort_points_capacity']
        if pd.notna(val) and val > 0:
            cap_override = int(val)

    # Group paths by origin facility
    origin_paths = {}
    for i in path_keys:
        origin = path_od_data[i]['origin']
        origin_paths.setdefault(origin, []).append(i)

    hub_types = {'hub', 'hybrid'}
    constraints_added = 0

    for facility in fac_lookup.index:
        if fac_lookup.at[facility, 'type'].lower() not in hub_types:
            continue
        if facility not in origin_paths:
            continue

        # Get capacity for this facility
        max_cap = cap_override
        if max_cap is None and 'max_sort_points_capacity' in fac_lookup.columns:
            max_cap = fac_lookup.at[facility, 'max_sort_points_capacity']

        if pd.isna(max_cap) or max_cap <= 0:
            continue
        max_cap = int(max_cap)

        # For sort point counting, we need to track unique destinations per sort level
        # When a path is selected, it "activates" that (destination, sort_level) combo

        # Group paths by (effective_sort_target, effective_sort_level)
        dest_sort_combos = {}  # (sort_target, eff_sort) -> list of path indices

        for i in origin_paths[facility]:
            dest = path_od_data[i]['dest']
            sl = path_od_data[i]['sort_level']

            # Determine effective sort level and target for sort point counting
            if sl == 'region':
                # For region sort, we're sorting to the regional hub
                if dest in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
                    regional_hub = fac_lookup.at[dest, 'regional_sort_hub']
                    if pd.isna(regional_hub) or regional_hub == '':
                        regional_hub = dest
                else:
                    regional_hub = dest

                if facility == regional_hub:
                    # Origin IS the regional hub - can't do region sort
                    eff_sort = 'market'
                    sort_target = dest
                else:
                    eff_sort = 'region'
                    sort_target = regional_hub
            else:
                eff_sort = sl
                sort_target = dest

            key = (sort_target, eff_sort)
            dest_sort_combos.setdefault(key, []).append(i)

        # Create binary variable for each unique (sort_target, sort_level) combo
        # This variable = 1 if we route to that target at that sort level
        combo_vars = {}
        for key, path_indices in dest_sort_combos.items():
            sort_target, eff_sort = key
            var_name = f"active_{facility}_{sort_target}_{eff_sort}"
            combo_var = model.NewBoolVar(var_name)
            combo_vars[key] = combo_var

            # combo_var = 1 iff at least one of these paths is selected
            # At most one path per OD is selected, so sum(x[i]) is 0 or 1 for each OD
            # But multiple ODs might share the same (sort_target, eff_sort) combo

            # combo_var >= x[i] for all i (if any path selected, combo is active)
            for i in path_indices:
                model.Add(combo_var >= x[i])

        # Calculate sort points: sum over active combos
        point_terms = []
        for (sort_target, eff_sort), combo_var in combo_vars.items():
            if eff_sort == 'sort_group':
                if sort_target in fac_lookup.index and 'last_mile_sort_groups_count' in fac_lookup.columns:
                    groups = fac_lookup.at[sort_target, 'last_mile_sort_groups_count']
                    pts = pts_per_dest * (int(groups) if pd.notna(groups) and groups > 0 else 1)
                else:
                    pts = pts_per_dest
            elif eff_sort == 'market':
                pts = pts_per_dest
            elif eff_sort == 'region':
                pts = 1
            else:
                pts = pts_per_dest

            point_terms.append(combo_var * int(pts))

        if point_terms:
            model.Add(sum(point_terms) <= max_cap)
            constraints_added += 1

            # Debug: show how many unique combos this facility has
            sg_count = sum(1 for (_, eff) in combo_vars.keys() if eff == 'sort_group')
            mkt_count = sum(1 for (_, eff) in combo_vars.keys() if eff == 'market')
            reg_count = sum(1 for (_, eff) in combo_vars.keys() if eff == 'region')
            print(f"      {facility}: {len(combo_vars)} combos (sg:{sg_count}, mkt:{mkt_count}, reg:{reg_count}), cap={max_cap}")

    if constraints_added > 0:
        print(f"    Sort capacity constraints: {constraints_added} facilities")

    return constraints_added


def _add_regional_sort_consistency_constraints(model, x, path_keys, path_od_data, fac_lookup):
    """
    Ensure consistent sort level on each arc from regional hub to destination.

    Business logic: On arc (hub → dest), all containers must be at the same
    sort granularity. For example:
    - Path A→B→C with region sort: B re-sorts to dest_sort_level for arc B→C
    - Path B→C with sort_group: arc B→C carries sort_group containers
    - If both exist, they must agree (B→C forces sort_group for A→B→C's DSL)

    O=D paths are excluded - they don't use any linehaul arc.
    Paths where hub==dest are excluded - no arc from hub to itself.
    """
    # Build dest -> regional_hub mapping
    dest_to_hub = {}
    for f in fac_lookup.index:
        if 'regional_sort_hub' in fac_lookup.columns:
            hub = fac_lookup.at[f, 'regional_sort_hub']
            dest_to_hub[f] = str(hub).strip() if pd.notna(hub) and str(hub).strip() else f
        else:
            dest_to_hub[f] = f

    # Find arcs (hub → dest) and group paths by effective sort level on that arc
    arc_sort_levels = {}  # (hub, dest) -> {effective_sort_level: [path_indices]}

    for i in path_keys:
        pd_ = path_od_data[i]
        origin = pd_['origin']
        dest = pd_['dest']
        nodes = pd_['path_nodes']
        sl = pd_['sort_level']
        dsl = pd_.get('dest_sort_level')

        # Skip O=D paths - they don't use any linehaul arc
        if origin == dest:
            continue

        if dest not in dest_to_hub:
            continue
        hub = dest_to_hub[dest]

        # Skip if hub not in path (path doesn't go through regional hub)
        if hub not in nodes:
            continue

        # Skip if hub == dest (no arc hub→dest exists; hub IS the dest)
        if hub == dest:
            continue

        # Determine the effective sort level on arc hub→dest
        if sl == 'region':
            # Region sort: hub re-sorts, dest_sort_level is the arc granularity
            eff_sort = dsl if dsl else 'market'
        elif sl == 'market':
            # Market sort at origin: containers at market level on all arcs
            eff_sort = 'market'
        elif sl == 'sort_group':
            # Sort_group at origin: containers at sort_group level on all arcs
            eff_sort = 'sort_group'
        else:
            eff_sort = 'market'

        key = (hub, dest)
        arc_sort_levels.setdefault(key, {}).setdefault(eff_sort, []).append(i)

    # Add consistency constraints for each arc
    constraints = 0
    for (hub, dest), sort_paths in arc_sort_levels.items():
        if len(sort_paths) <= 1:
            # Only one sort level option on this arc - no constraint needed
            continue

        # Create binary variable for each sort level option
        sort_vars = {sl: model.NewBoolVar(f"arc_{hub}_{dest}_{sl}") for sl in sort_paths}

        # Exactly one sort level must be chosen for this arc
        model.Add(sum(sort_vars.values()) == 1)

        # Link path selection to sort level choice
        for sl, indices in sort_paths.items():
            for i in indices:
                model.Add(x[i] <= sort_vars[sl])

        constraints += 1

    if constraints:
        print(f"    Regional sort consistency: {constraints} arc constraints")

    return constraints


def _get_regional_hub_for_dest(dest: str, fac_lookup: pd.DataFrame) -> str:
    """Get the regional sort hub for a destination facility."""
    if dest not in fac_lookup.index:
        return dest

    if 'regional_sort_hub' not in fac_lookup.columns:
        return dest

    hub = fac_lookup.at[dest, 'regional_sort_hub']
    if pd.notna(hub) and str(hub).strip():
        return str(hub).strip()

    return dest


def _calculate_processing_cost(path_data: dict, fac_lookup: pd.DataFrame,
                                package_mix: pd.DataFrame, container_params: pd.DataFrame,
                                cost_params: CostParameters) -> float:
    """
    Calculate per-package processing cost for a path.

    Cost components:
    1. Injection sort at origin (always)
    2. Intermediate handling:
       - Region sort: sort cost at regional hub, crossdock cost at others
       - Market/sort_group sort: crossdock cost only (containers persist)
    3. Last-mile sort at destination (unless sort_group at origin or dest_sort_level=sort_group)
    4. Delivery cost
    5. Container inefficiency penalty for sort_group (creates more partial containers)

    Container handling rules:
    - Containers are created at origin based on sort_level
    - Containers persist through crossdock operations (no break-down)
    - Containers are broken down at sort operations (regional hub for region sort)
    """
    nodes = path_data['path_nodes']
    origin = path_data['origin']
    dest = path_data['dest']
    sl = path_data.get('sort_level', 'market')
    dsl = path_data.get('dest_sort_level')  # Only relevant for region sort
    volume = path_data.get('pkgs_day', 0)

    # 1. Injection sort at origin (always incurred)
    cost = cost_params.injection_sort_cost_per_pkg

    # Handle O=D case (no linehaul)
    if origin == dest:
        # Still need last-mile sort and delivery
        return cost + cost_params.last_mile_sort_cost_per_pkg + cost_params.last_mile_delivery_cost_per_pkg

    # Get containers per package for crossdock cost calculation
    cpp = _calculate_containers_per_package(package_mix, container_params)

    # Get regional hub for destination
    regional_hub = _get_regional_hub_for_dest(dest, fac_lookup)

    # 2. Intermediate handling costs
    intermediates = nodes[1:-1] if len(nodes) > 2 else []

    for intermediate in intermediates:
        if sl == 'region' and intermediate == regional_hub:
            # This intermediate is the regional hub - incurs sort cost
            cost += cost_params.intermediate_sort_cost_per_pkg
        else:
            # This intermediate is a crossdock - incurs container handling only
            cost += cost_params.container_handling_cost * cpp

    # 3. Last-mile sort at destination
    # Not needed if packages are already sorted to sort_group level
    needs_last_mile_sort = True

    if sl == 'sort_group':
        # Origin sorted to sort_group - no last-mile sort needed
        needs_last_mile_sort = False
    elif sl == 'region' and dsl == 'sort_group':
        # Regional hub sorted to sort_group - no last-mile sort needed
        needs_last_mile_sort = False
    # market sort or region with dsl=market needs last-mile sort

    if needs_last_mile_sort:
        cost += cost_params.last_mile_sort_cost_per_pkg

    # 4. Delivery cost
    cost += cost_params.last_mile_delivery_cost_per_pkg

    # 5. Container inefficiency penalty for sort_group
    # sort_group creates one container per sort group (e.g., 4 containers)
    # market/region creates one container for the destination
    # The extra containers waste truck space when partially filled
    if sl == 'sort_group' and dest in fac_lookup.index:
        dest_sort_groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
        if pd.notna(dest_sort_groups) and dest_sort_groups > 1:
            dest_sort_groups = int(dest_sort_groups)

            # Calculate container capacity (packages per container)
            gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
            container_cube = float(gaylord["usable_cube_cuft"]) * float(gaylord["pack_utilization_container"])
            pkg_cube = _weighted_pkg_cube(package_mix)
            pkgs_per_container = container_cube / pkg_cube if pkg_cube > 0 else 500

            # Extra containers beyond what market sort would need
            market_containers = max(1, int(np.ceil(volume / pkgs_per_container)))
            sg_containers = dest_sort_groups  # One per sort group
            extra_containers = max(0, sg_containers - market_containers)

            # Only penalize if sort_group creates MORE containers than market would
            if extra_containers > 0:
                # The penalty is the cost of the extra containers
                # multiplied by the path length (more touches = more handling)
                num_arcs = max(1, len(nodes) - 1)
                # Penalty per extra container: handling cost at each intermediate + truck space cost
                # Using container_handling_cost as a proxy for the space/handling inefficiency
                container_penalty_per = cost_params.container_handling_cost * 0.5 * num_arcs
                cost += (extra_containers * container_penalty_per) / max(1, volume)

    return cost


def _build_selected_paths(chosen_idx, path_od_data, path_arcs, arc_meta,
                           arc_trucks, arc_pkgs, cost_params, fac_lookup,
                           package_mix, container_params, solver):
    """Build DataFrame of selected paths with costs."""
    data = []
    for i in chosen_idx:
        pd_ = path_od_data[i]

        # Calculate transport cost (allocated by volume share on each arc)
        transport = 0
        for a in path_arcs[i]:
            arc_vol = solver.Value(arc_pkgs[a])
            if arc_vol > 0:
                trucks = solver.Value(arc_trucks[a])
                share = pd_['pkgs_day'] / arc_vol
                transport += trucks * arc_meta[a]['cost_per_truck'] * share

        # Calculate processing cost
        proc = _calculate_processing_cost(
            pd_, fac_lookup, package_mix, container_params, cost_params
        ) * pd_['pkgs_day']

        total = transport + proc
        data.append({
            'origin': pd_['origin'],
            'dest': pd_['dest'],
            'pkgs_day': pd_['pkgs_day'],
            'path_str': pd_['path_str'],
            'path_type': pd_['path_type'],
            'path_nodes': pd_['path_nodes'],
            'chosen_sort_level': pd_['sort_level'],
            'dest_sort_level': pd_.get('dest_sort_level'),
            'zone': pd_['zone'],
            'tit_hours': pd_['tit_hours'],
            'total_path_miles': pd_['total_path_miles'],
            'direct_miles': pd_['direct_miles'],
            'total_cost': total,
            'linehaul_cost': transport,
            'processing_cost': proc,
            'cost_per_pkg': safe_divide(total, pd_['pkgs_day']),
        })

    return pd.DataFrame(data)


def _build_arc_summary(arc_meta, arc_pkgs, arc_trucks, w_cube, raw_trailer_cube, container_params, solver):
    """Build arc-level summary DataFrame."""
    gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    raw_cont = float(gaylord["usable_cube_cuft"])
    pack_util = float(gaylord["pack_utilization_container"])

    data = []
    for a_idx, arc in enumerate(arc_meta):
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            cube = pkgs * w_cube
            containers = max(1, int(np.ceil(cube / (raw_cont * pack_util))))
            data.append({
                "from_facility": arc["from"],
                "to_facility": arc["to"],
                "distance_miles": arc["distance_miles"],
                "pkgs_day": pkgs,
                "pkg_cube_cuft": cube,
                "trucks": trucks,
                "physical_containers": containers,
                "container_fill_rate": min(1.0, cube / (containers * raw_cont)),
                "truck_fill_rate": min(1.0, cube / (trucks * raw_trailer_cube)),
                "cost_per_truck": arc["cost_per_truck"],
                "total_cost": trucks * arc["cost_per_truck"],
                "CPP": safe_divide(trucks * arc["cost_per_truck"], pkgs),
            })

    return pd.DataFrame(data).sort_values(["from_facility", "to_facility"]).reset_index(drop=True)


def _calculate_kpis(selected_paths, arc_summary, raw_trailer_cube):
    """Calculate network-level KPIs."""
    if arc_summary.empty:
        return {
            "avg_truck_fill_rate": 0,
            "avg_container_fill_rate": 0,
            "total_cost": 0,
            "total_packages": 0,
        }

    total_cube = arc_summary['pkg_cube_cuft'].sum()
    total_cap = (arc_summary['trucks'] * raw_trailer_cube).sum()
    total_vol = arc_summary['pkgs_day'].sum()

    return {
        "avg_truck_fill_rate": safe_divide(total_cube, total_cap),
        "avg_container_fill_rate": (
            arc_summary['container_fill_rate'] * arc_summary['pkgs_day']
        ).sum() / total_vol if total_vol > 0 else 0,
        "total_cost": selected_paths['total_cost'].sum(),
        "total_packages": selected_paths['pkgs_day'].sum(),
    }


def _build_sort_summary(selected_paths, fac_lookup, cost_params=None):
    """Build sort decision summary DataFrame."""
    if selected_paths.empty:
        return pd.DataFrame()

    data = []

    for _, r in selected_paths.iterrows():
        o_hub = ''
        d_hub = ''

        if r['origin'] in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
            hub = fac_lookup.at[r['origin'], 'regional_sort_hub']
            o_hub = hub if pd.notna(hub) and hub != '' else r['origin']

        if r['dest'] in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
            hub = fac_lookup.at[r['dest'], 'regional_sort_hub']
            d_hub = hub if pd.notna(hub) and hub != '' else r['dest']

        data.append({
            'origin': r['origin'],
            'origin_region_hub': o_hub or r['origin'],
            'dest': r['dest'],
            'dest_region_hub': d_hub or r['dest'],
            'pkgs_day': r['pkgs_day'],
            'chosen_sort_level': r['chosen_sort_level'],
            'dest_sort_level': r.get('dest_sort_level'),
            'total_cost': r['total_cost'],
        })

    return pd.DataFrame(data)


def _calculate_sort_point_usage(selected_paths, all_paths_df, fac_lookup, cost_params):
    """
    Calculate sort point usage per facility: actual vs minimum feasible.

    Returns DataFrame with columns:
    - facility: facility name
    - capacity: max_sort_points_capacity
    - actual_used: sort points used by selected solution
    - min_feasible: theoretical minimum if all region-eligible use region
    - utilization: actual_used / capacity
    """
    if selected_paths.empty:
        return pd.DataFrame()

    pts_per_dest = cost_params.sort_points_per_destination
    hub_types = {'hub', 'hybrid'}

    results = []

    for facility in fac_lookup.index:
        if fac_lookup.at[facility, 'type'].lower() not in hub_types:
            continue

        # Get capacity
        cap = fac_lookup.at[facility, 'max_sort_points_capacity'] if 'max_sort_points_capacity' in fac_lookup.columns else None
        if pd.isna(cap) or cap <= 0:
            continue
        cap = int(cap)

        # Get selected paths from this facility
        fac_selected = selected_paths[selected_paths['origin'] == facility]
        if fac_selected.empty:
            continue

        # Calculate actual sort points used (unique destinations × sort level points)
        dest_sort_combos = {}  # (sort_target, eff_sort) -> True

        for _, row in fac_selected.iterrows():
            dest = row['dest']
            sl = row['chosen_sort_level']

            if sl == 'region':
                # Get regional hub
                if dest in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
                    hub = fac_lookup.at[dest, 'regional_sort_hub']
                    if pd.isna(hub) or hub == '':
                        hub = dest
                else:
                    hub = dest

                if facility == hub:
                    eff_sort = 'market'
                    sort_target = dest
                else:
                    eff_sort = 'region'
                    sort_target = hub
            else:
                eff_sort = sl
                sort_target = dest

            dest_sort_combos[(sort_target, eff_sort)] = True

        # Calculate actual points
        actual_pts = 0
        for (sort_target, eff_sort) in dest_sort_combos:
            if eff_sort == 'sort_group':
                if sort_target in fac_lookup.index and 'last_mile_sort_groups_count' in fac_lookup.columns:
                    groups = fac_lookup.at[sort_target, 'last_mile_sort_groups_count']
                    pts = pts_per_dest * (int(groups) if pd.notna(groups) and groups > 0 else 1)
                else:
                    pts = pts_per_dest
            elif eff_sort == 'market':
                pts = pts_per_dest
            elif eff_sort == 'region':
                pts = 1
            else:
                pts = pts_per_dest
            actual_pts += pts

        # Calculate minimum feasible (use all available region paths)
        fac_all = all_paths_df[all_paths_df['origin'] == facility] if all_paths_df is not None else None
        min_pts = None

        if fac_all is not None and not fac_all.empty:
            dest_options = fac_all.groupby('dest')['sort_level'].apply(lambda x: set(x.str.lower())).to_dict()

            region_hubs = set()
            non_region_pts = 0

            for dest, options in dest_options.items():
                if 'region' in options:
                    # Can use region
                    if dest in fac_lookup.index and 'regional_sort_hub' in fac_lookup.columns:
                        hub = fac_lookup.at[dest, 'regional_sort_hub']
                        if pd.notna(hub) and hub != '' and hub != facility:
                            region_hubs.add(hub)
                        else:
                            # Can't use region (facility IS the hub), use market
                            non_region_pts += pts_per_dest
                    else:
                        non_region_pts += pts_per_dest
                else:
                    # Must use market or sort_group - use market (cheaper)
                    non_region_pts += pts_per_dest

            min_pts = len(region_hubs) + non_region_pts

        results.append({
            'facility': facility,
            'capacity': cap,
            'actual_used': actual_pts,
            'min_feasible': min_pts,
            'utilization': actual_pts / cap if cap > 0 else 0,
            'headroom': cap - actual_pts,
        })

    return pd.DataFrame(results)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _weighted_pkg_cube(package_mix: pd.DataFrame) -> float:
    """Calculate weighted average package cube."""
    return (package_mix["avg_cube_cuft"] * package_mix["share_of_pkgs"]).sum()


def _get_container_capacity(container_params: pd.DataFrame) -> float:
    """Get effective container capacity in cuft."""
    gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    return float(gaylord["usable_cube_cuft"]) * float(gaylord["pack_utilization_container"])


def _get_containers_per_truck(container_params: pd.DataFrame) -> int:
    """Get number of containers per truck."""
    gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    if "containers_per_truck" in gaylord.index:
        return int(gaylord["containers_per_truck"])
    raise ValueError("container_params missing 'containers_per_truck' column")


def _get_raw_trailer_cube(container_params: pd.DataFrame) -> float:
    """Get raw trailer cube capacity."""
    gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    if "trailer_air_cube_cuft" in gaylord.index:
        return float(gaylord["trailer_air_cube_cuft"])
    raise ValueError("container_params missing 'trailer_air_cube_cuft' column")


def _calculate_containers_per_package(package_mix: pd.DataFrame, container_params: pd.DataFrame) -> float:
    """Calculate containers per package (inverse of packages per container)."""
    w_cube = _weighted_pkg_cube(package_mix)
    container_cap = _get_container_capacity(container_params)
    pkgs_per_container = container_cap / w_cube if w_cube > 0 else 50
    return 1.0 / pkgs_per_container


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in miles."""
    EARTH_RADIUS_MILES = 3958.8

    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_MILES * c


def _band_lookup(distance: float, mileage_bands: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Look up cost parameters for distance band."""
    for _, band in mileage_bands.iterrows():
        if band["mileage_band_min"] <= distance < band["mileage_band_max"]:
            return (
                float(band["fixed_cost_per_truck"]),
                float(band["variable_cost_per_mile"]),
                float(band["circuity_factor"]),
                float(band["mph"]),
            )

    last = mileage_bands.iloc[-1]
    return (
        float(last["fixed_cost_per_truck"]),
        float(last["variable_cost_per_mile"]),
        float(last["circuity_factor"]),
        float(last["mph"]),
    )
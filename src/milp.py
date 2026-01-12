"""
MILP Network Optimization Module

Solves minimum-cost network configuration using arc-pooled Mixed Integer Linear Programming.
Uses pre-enumerated feasible paths from SLA model.

Key Features:
- Paths come from feasible_paths (SLA model output)
- Regional sort consistency constraint
- Transport costs from mileage_bands
- TIT from feasible_paths (reporting only)
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

    Args:
        candidates: Filtered paths from load_feasible_paths
        facilities: Facility data
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

    path_keys = list(cand.index)
    w_cube = _weighted_pkg_cube(package_mix)

    arc_index_map, arc_meta, path_arcs, path_od_data = _build_arc_structures(
        cand, facilities, mileage_bands, path_keys
    )

    groups = cand.groupby(["origin", "dest"]).indices
    print(f"    Network: {len(arc_meta)} arcs, {len(path_keys)} paths, {len(groups)} OD pairs")

    # Build model
    model = cp_model.CpModel()
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    arc_pkgs = {a: model.NewIntVar(0, 10_000_000, f"arc_pkgs_{a}") for a in range(len(arc_meta))}
    arc_trucks = {a: model.NewIntVar(0, 10_000, f"arc_trucks_{a}") for a in range(len(arc_meta))}

    # Exactly one path per OD
    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # Arc capacity constraints
    _add_arc_capacity_constraints(
        model, x, arc_pkgs, arc_trucks, arc_meta, path_arcs,
        path_od_data, path_keys, package_mix, container_params
    )

    # Sort constraints
    if enable_sort_optimization:
        _add_sort_capacity_constraints(
            model, x, path_keys, path_od_data, facilities, cost_params, scenario_row
        )
        _add_regional_sort_consistency_constraints(
            model, x, path_keys, path_od_data, facilities
        )

    # Objective: minimize total cost
    cost_terms = []
    for a_idx in range(len(arc_meta)):
        truck_cost = safe_int_cost(arc_meta[a_idx]["cost_per_truck"], f"arc_{a_idx}")
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    for i in path_keys:
        path_data = path_od_data[i]
        volume = int(round(path_data['pkgs_day']))
        proc_cost = _calculate_processing_cost(
            path_data, facilities, package_mix, container_params, cost_params
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
        return pd.DataFrame(), pd.DataFrame(), {"solver_status": status_msg}, pd.DataFrame()

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    print(f"    Selected {len(chosen_idx)} optimal paths")

    selected_paths = _build_selected_paths(
        chosen_idx, path_od_data, path_arcs, arc_meta,
        arc_trucks, arc_pkgs, cost_params, facilities,
        package_mix, container_params, solver
    )

    raw_trailer_cube = _get_raw_trailer_cube(container_params)
    arc_summary = _build_arc_summary(
        arc_meta, arc_pkgs, arc_trucks, w_cube,
        raw_trailer_cube, container_params, solver
    )

    kpis = _calculate_kpis(selected_paths, arc_summary, raw_trailer_cube)
    sort_summary = _build_sort_summary(selected_paths, facilities) if enable_sort_optimization else pd.DataFrame()

    print(f"    Cost: ${selected_paths['total_cost'].sum():,.2f}")
    return selected_paths, arc_summary, kpis, sort_summary


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
            'sort_level': r.get("sort_level", "market"),
            'dest_sort_level': r.get("dest_sort_level"),
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


def _add_sort_capacity_constraints(model, x, path_keys, path_od_data, facilities,
                                    cost_params, scenario_row):
    """Add sort capacity constraints at hub facilities."""
    fac_lookup = facilities.set_index("facility_name")
    pts_per_dest = cost_params.sort_points_per_destination

    cap_override = None
    if scenario_row is not None and 'max_sort_points_capacity' in scenario_row.index:
        val = scenario_row['max_sort_points_capacity']
        if pd.notna(val) and val > 0:
            cap_override = int(val)

    origin_paths = {}
    for i in path_keys:
        origin = path_od_data[i]['origin']
        origin_paths.setdefault(origin, []).append(i)

    hub_types = {'hub', 'hybrid'}
    for _, fac_row in facilities.iterrows():
        facility = fac_row['facility_name']
        if fac_row['type'].lower() not in hub_types:
            continue
        if facility not in origin_paths:
            continue
        if facility not in fac_lookup.index:
            continue

        max_cap = cap_override
        if max_cap is None and 'max_sort_points_capacity' in fac_lookup.columns:
            max_cap = fac_lookup.at[facility, 'max_sort_points_capacity']

        if pd.isna(max_cap) or max_cap <= 0:
            continue
        max_cap = int(max_cap)

        point_terms = []
        for i in origin_paths[facility]:
            dest = path_od_data[i]['dest']
            sl = path_od_data[i]['sort_level']

            if sl == 'sort_group' and dest in fac_lookup.index:
                if 'last_mile_sort_groups_count' in fac_lookup.columns:
                    groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                    pts = pts_per_dest * (int(groups) if pd.notna(groups) and groups > 0 else 1)
                else:
                    pts = pts_per_dest
            else:
                pts = pts_per_dest

            point_terms.append(x[i] * int(pts))

        if point_terms:
            model.Add(sum(point_terms) <= max_cap)


def _add_regional_sort_consistency_constraints(model, x, path_keys, path_od_data, facilities):
    """
    Ensure consistent dest_sort_level at regional sort hubs.

    For hub H and destination D:
    - All paths through H to D must use the same dest_sort_level
    - This applies whether H is origin or intermediate
    """
    fac_lookup = facilities.set_index("facility_name")

    # Build dest -> regional_hub mapping
    dest_to_hub = {}
    for f in fac_lookup.index:
        if 'regional_sort_hub' in fac_lookup.columns:
            hub = fac_lookup.at[f, 'regional_sort_hub']
            dest_to_hub[f] = hub if pd.notna(hub) and hub != '' else f
        else:
            dest_to_hub[f] = f

    # Find (hub, dest) pairs and their dest_sort_levels
    hub_dest_paths = {}
    for i in path_keys:
        pd_ = path_od_data[i]
        dest = pd_['dest']
        nodes = pd_['path_nodes']
        sl = pd_['sort_level']
        dsl = pd_.get('dest_sort_level')

        if dest not in dest_to_hub:
            continue
        hub = dest_to_hub[dest]
        if hub not in nodes:
            continue

        eff_dsl = dsl if sl == 'region' else sl if sl in ['market', 'sort_group'] else 'market'
        key = (hub, dest)
        hub_dest_paths.setdefault(key, {}).setdefault(eff_dsl, []).append(i)

    # Add consistency constraints
    constraints = 0
    for (hub, dest), dsl_paths in hub_dest_paths.items():
        if len(dsl_paths) <= 1:
            continue

        dsl_vars = {dsl: model.NewBoolVar(f"dsl_{hub}_{dest}_{dsl}") for dsl in dsl_paths}
        model.Add(sum(dsl_vars.values()) == 1)

        for dsl, indices in dsl_paths.items():
            for i in indices:
                model.Add(x[i] <= dsl_vars[dsl])
        constraints += 1

    if constraints:
        print(f"    Regional sort consistency: {constraints} constraints")


def _calculate_processing_cost(path_data, facilities, package_mix, container_params, cost_params):
    """Calculate per-package processing cost for a path."""
    nodes = path_data['path_nodes']
    origin = path_data['origin']
    dest = path_data['dest']
    sl = path_data.get('sort_level', 'market')

    cost = cost_params.injection_sort_cost_per_pkg

    if origin == dest:
        return cost + cost_params.last_mile_sort_cost_per_pkg + cost_params.last_mile_delivery_cost_per_pkg

    intermediates = nodes[1:-1] if len(nodes) > 2 else []
    if intermediates:
        cpp = _calculate_containers_per_package(package_mix, container_params)
        if sl == 'region':
            cost += len(intermediates) * cost_params.intermediate_sort_cost_per_pkg
        else:
            cost += len(intermediates) * cost_params.container_handling_cost * cpp

    if sl != 'sort_group':
        cost += cost_params.last_mile_sort_cost_per_pkg

    return cost + cost_params.last_mile_delivery_cost_per_pkg


def _build_selected_paths(chosen_idx, path_od_data, path_arcs, arc_meta,
                           arc_trucks, arc_pkgs, cost_params, facilities,
                           package_mix, container_params, solver):
    """Build DataFrame of selected paths with costs."""
    data = []
    for i in chosen_idx:
        pd_ = path_od_data[i]

        transport = 0
        for a in path_arcs[i]:
            arc_vol = solver.Value(arc_pkgs[a])
            if arc_vol > 0:
                trucks = solver.Value(arc_trucks[a])
                share = pd_['pkgs_day'] / arc_vol
                transport += trucks * arc_meta[a]['cost_per_truck'] * share

        proc = _calculate_processing_cost(
            pd_, facilities, package_mix, container_params, cost_params
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


def _build_sort_summary(selected_paths, facilities):
    """Build sort decision summary DataFrame."""
    if selected_paths.empty:
        return pd.DataFrame()

    fac_lookup = facilities.set_index("facility_name")
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
    return 22


def _get_raw_trailer_cube(container_params: pd.DataFrame) -> float:
    """Get raw trailer cube capacity."""
    gaylord = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    if "trailer_cube_cuft" in gaylord.index:
        return float(gaylord["trailer_cube_cuft"])
    return 2700.0


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
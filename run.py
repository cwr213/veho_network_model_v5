"""
Main Execution Script

Orchestrates network optimization using pre-enumerated paths from SLA model.

Usage:
    python run.py --input input.xlsx --output_dir outputs/
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

from src.config import CostParameters, LoadStrategy, OUTPUT_FILE_TEMPLATE
from src.io_loader import load_workbook, params_to_dict
from src.validators import validate_inputs
from src.load_feasible_paths import (
    load_and_filter_feasible_paths,
    validate_feasible_paths_columns,
    validate_path_structure,
    summarize_candidate_paths
)
from src.milp import solve_network_optimization


def generate_run_id(scenarios_df: pd.DataFrame) -> str:
    """Generate unique run identifier."""
    years = scenarios_df['year'].unique()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year_str = f"{min(years)}" if len(years) == 1 else f"{min(years)}-{max(years)}"

    if 'max_sort_points_capacity' in scenarios_df.columns:
        caps = scenarios_df['max_sort_points_capacity'].dropna()
        if len(caps) > 0:
            cap_vals = sorted(caps.unique())
            cap_str = f"_cap{int(cap_vals[0])}" if len(cap_vals) == 1 else f"_cap{int(min(cap_vals))}-{int(max(cap_vals))}"
            return f"{year_str}_{timestamp}{cap_str}"

    return f"{year_str}_{timestamp}"


def add_direct_injection_zone_classification(direct_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure direct injection has zone=0."""
    if direct_df.empty:
        return direct_df
    df = direct_df.copy()
    df['zone'] = 0
    return df


def main(input_path: str, output_dir: str) -> int:
    """
    Main entry point for sort model optimization.

    Args:
        input_path: Path to input Excel file
        output_dir: Directory for output files

    Returns:
        Exit code (0 for success)
    """
    start_time = datetime.now()

    print("=" * 70)
    print("VEHO NETWORK OPTIMIZATION - SORT MODEL")
    print("=" * 70)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n Input: {input_path.name}")
    print(f" Output: {output_dir}")

    # Load and validate
    print(f"\n{'=' * 70}")
    print("LOADING INPUTS")
    print("=" * 70)

    try:
        dfs = load_workbook(input_path)
        print("Workbook loaded successfully")
    except Exception as e:
        print(f"\n ERROR: Could not load workbook: {e}")
        return 1

    try:
        validate_inputs(dfs)
        print("Input validation passed")
    except Exception as e:
        print(f"\n ERROR: Input validation failed: {e}")
        return 1

    try:
        validate_feasible_paths_columns(dfs["feasible_paths"])
        print("Feasible paths columns validated")
    except Exception as e:
        print(f"\n ERROR: Feasible paths validation failed: {e}")
        return 1

    # Parse parameters
    print(f"\n{'=' * 70}")
    print("PARSING PARAMETERS")
    print("=" * 70)

    cost_params_dict = params_to_dict(dfs["cost_params"])
    run_settings_dict = params_to_dict(dfs["run_settings"])

    cost_params = CostParameters(
        injection_sort_cost_per_pkg=float(cost_params_dict["injection_sort_cost_per_pkg"]),
        intermediate_sort_cost_per_pkg=float(cost_params_dict["intermediate_sort_cost_per_pkg"]),
        last_mile_sort_cost_per_pkg=float(cost_params_dict["last_mile_sort_cost_per_pkg"]),
        last_mile_delivery_cost_per_pkg=float(cost_params_dict["last_mile_delivery_cost_per_pkg"]),
        container_handling_cost=float(cost_params_dict["container_handling_cost"]),
        sort_points_per_destination=float(cost_params_dict["sort_points_per_destination"]),
    )

    global_strategy = LoadStrategy.CONTAINER
    enable_sort_opt = bool(run_settings_dict.get("enable_sort_optimization", False))

    user_run_id = run_settings_dict.get("run_id", None)
    run_id = str(user_run_id).strip() if user_run_id and str(user_run_id).strip() else generate_run_id(dfs["scenarios"])

    print(f"\nConfiguration:")
    print(f"  Strategy: Container")
    print(f"  Sort optimization: {'ENABLED' if enable_sort_opt else 'DISABLED'}")
    print(f"  Allow inactive arcs: {run_settings_dict.get('allow_inactive_arcs', False)}")
    print(f"  Run ID: {run_id}")

    all_results = []
    created_files = []

    # Process scenarios
    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(dfs['scenarios'])} SCENARIOS")
    print("=" * 70)

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        scenario_id = scenario_row["scenario_id"]
        year = int(scenario_row["year"])
        day_type = str(scenario_row["day_type"]).strip().lower()

        print(f"\n{'-' * 70}")
        print(f"SCENARIO {scenario_idx + 1}/{len(dfs['scenarios'])}: {scenario_id}")
        print(f"  Year: {year}, Day Type: {day_type}")
        print("-" * 70)

        try:
            # Load feasible paths
            print("\n1. Loading feasible paths...")

            middle_mile, direct_day = load_and_filter_feasible_paths(
                dfs["feasible_paths"],
                run_settings_dict,
                scenario_id
            )

            if middle_mile.empty:
                print(f"  ERROR: No middle-mile paths for scenario")
                continue

            validate_path_structure(middle_mile)
            print("  Path structure validated")

            middle_mile['scenario_id'] = scenario_id
            middle_mile['day_type'] = day_type

            print(summarize_candidate_paths(middle_mile))

            direct_day = add_direct_injection_zone_classification(direct_day)
            direct_pkgs_total = direct_day["dir_pkgs_day"].sum() if not direct_day.empty else 0
            print(f"  Direct injection (Zone 0): {direct_pkgs_total:,.0f} packages")

            # Run optimization
            print("\n2. Running MILP optimization...")

            od_selected, arc_summary, network_kpis, sort_summary = solve_network_optimization(
                candidates=middle_mile,
                facilities=dfs["facilities"],
                mileage_bands=dfs["mileage_bands"],
                package_mix=dfs["package_mix"],
                container_params=dfs["container_params"],
                cost_params=cost_params,
                global_strategy=global_strategy,
                enable_sort_optimization=enable_sort_opt,
                scenario_row=scenario_row
            )

            if od_selected.empty:
                print(f"  ERROR: Optimization returned no paths")
                continue

            print(f"  Selected {len(od_selected)} optimal paths")

            # Calculate summary
            total_cost = od_selected["total_cost"].sum()
            total_pkgs = od_selected["pkgs_day"].sum()
            cost_per_pkg = total_cost / max(total_pkgs, 1)

            print(f"\n3. Results:")
            print(f"  Total cost: ${total_cost:,.0f}")
            print(f"  Cost per package: ${cost_per_pkg:.3f}")
            print(f"  Middle-mile packages: {total_pkgs:,.0f}")
            print(f"  Direct injection packages: {direct_pkgs_total:,.0f}")

            # Sort level summary
            if 'chosen_sort_level' in od_selected.columns:
                sl_summary = od_selected.groupby('chosen_sort_level')['pkgs_day'].sum()
                print(f"\n  Sort level distribution:")
                for sl, vol in sl_summary.items():
                    pct = vol / total_pkgs * 100 if total_pkgs > 0 else 0
                    print(f"    {sl}: {vol:,.0f} ({pct:.1f}%)")

            # Write outputs
            output_filename = OUTPUT_FILE_TEMPLATE.format(
                scenario_id=scenario_id,
                strategy=global_strategy.value
            )
            output_path = output_dir / output_filename

            _write_scenario_output(
                output_path, scenario_id, year, day_type, global_strategy,
                od_selected, arc_summary, network_kpis, sort_summary,
                direct_pkgs_total
            )

            created_files.append(output_filename)
            print(f"\n  Wrote: {output_filename}")

            # Track results
            all_results.append({
                "scenario_id": scenario_id,
                "year": year,
                "day_type": day_type,
                "total_cost": total_cost,
                "cost_per_pkg": cost_per_pkg,
                "middle_mile_packages": total_pkgs,
                "direct_injection_packages": direct_pkgs_total,
                **network_kpis
            })

        except Exception as e:
            print(f"\n ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    elapsed = datetime.now() - start_time

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Elapsed time: {elapsed}")
    print(f"Processed: {len(all_results)} scenarios")
    print(f"Created: {len(created_files)} output files")

    if created_files:
        print(f"\nOutput files in {output_dir}:")
        for f in sorted(created_files):
            print(f"  {f}")

    # Write comparison if multiple scenarios
    if len(all_results) > 1:
        _write_comparison(output_dir, run_id, all_results)
        print(f"  comparison_{run_id}.xlsx")

    return 0


def _write_scenario_output(
    output_path: Path,
    scenario_id: str,
    year: int,
    day_type: str,
    strategy: LoadStrategy,
    od_selected: pd.DataFrame,
    arc_summary: pd.DataFrame,
    kpis: dict,
    sort_summary: pd.DataFrame,
    direct_pkgs: float
):
    """Write scenario output to Excel."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = [
            {"key": "scenario_id", "value": scenario_id},
            {"key": "year", "value": year},
            {"key": "day_type", "value": day_type},
            {"key": "strategy", "value": strategy.value},
            {"key": "total_cost", "value": od_selected["total_cost"].sum()},
            {"key": "cost_per_pkg", "value": od_selected["total_cost"].sum() / max(od_selected["pkgs_day"].sum(), 1)},
            {"key": "middle_mile_packages", "value": od_selected["pkgs_day"].sum()},
            {"key": "direct_injection_packages", "value": direct_pkgs},
        ]
        for k, v in kpis.items():
            summary_data.append({"key": k, "value": v})

        pd.DataFrame(summary_data).to_excel(writer, sheet_name="summary", index=False)

        # Selected paths
        od_out = od_selected.copy()
        if 'path_nodes' in od_out.columns:
            od_out['path_nodes'] = od_out['path_nodes'].apply(lambda x: '->'.join(x) if isinstance(x, list) else x)
        od_out.to_excel(writer, sheet_name="selected_paths", index=False)

        # Arc summary
        if not arc_summary.empty:
            arc_summary.to_excel(writer, sheet_name="arc_summary", index=False)

        # Sort summary
        if not sort_summary.empty:
            sort_summary.to_excel(writer, sheet_name="sort_summary", index=False)


def _write_comparison(output_dir: Path, run_id: str, results: list):
    """Write multi-scenario comparison."""
    compare_path = output_dir / f"comparison_{run_id}.xlsx"

    df = pd.DataFrame(results)
    with pd.ExcelWriter(compare_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="comparison", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veho Network Optimization - Sort Model")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")

    args = parser.parse_args()

    try:
        exit_code = main(args.input, args.output_dir)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# Sort Model - SLA Integration

Network optimization using pre-enumerated paths from SLA model.

## Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `config.py` | Dataclasses, enums, constants |
| `io_loader.py` | Excel workbook loading |
| `validators.py` | Input validation |
| `load_feasible_paths.py` | SLA model output parsing and filtering |
| `milp.py` | Mixed Integer Linear Programming solver |
| `run.py` | Main execution script |

## Usage

```bash
# As module
python -m sort_model.run --input input.xlsx --output_dir outputs/

# Or import
from sort_model import main
main("input.xlsx", "outputs/")
```

## Input File Structure

### Required Sheets

| Sheet | Description |
|-------|-------------|
| `feasible_paths` | Pre-enumerated paths from SLA model |
| `mileage_bands` | Transport cost rates by distance |
| `cost_params` | Processing cost parameters |
| `container_params` | Container specifications |
| `package_mix` | Package size distribution |
| `facilities` | Facility locations and attributes |
| `scenarios` | Scenarios to optimize |
| `run_settings` | Runtime configuration |

### feasible_paths Columns (from SLA model)

Required:
- `scenario_id`, `origin`, `dest`
- `node_1`, `node_2` (node_3/4/5 optional)
- `path_type`: direct_injection, 2_touch, 3_touch, 4_touch, 5_touch
- `sort_level`: region, market, sort_group
- `total_path_miles`, `direct_miles`, `tit_hours`
- `sla_met`, `uses_only_active_arcs` (boolean)
- `pkgs_mm`, `pkgs_zs`, `pkgs_di`
- `zone_mm_zs`

Optional:
- `dest_sort_level` (for regional sort paths)
- `tit_sort_hours`, `tit_crossdock_hours`, etc.

### cost_params Keys

```
injection_sort_cost_per_pkg      0.50
intermediate_sort_cost_per_pkg   0.40
container_handling_cost          11.00
last_mile_sort_cost_per_pkg      0.25
last_mile_delivery_cost_per_pkg  0.00
sort_points_per_destination      2
```

### run_settings Keys

```
enable_sort_optimization         TRUE/FALSE
allow_inactive_arcs              TRUE/FALSE
fluid_opportunity_fill_threshold 0.75
run_id                           (optional)
```

## Key Constraints

### Regional Sort Consistency

For regional sort hub H and destination D, all paths through H to D must use the same `dest_sort_level`. This ensures sorting decisions are consistent whether H is origin or intermediate.

### Path Selection

Exactly one path selected per origin-destination pair. Path cost includes:
- Transport: truck cost allocated by volume share
- Processing: injection sort + intermediate handling + last-mile sort

## Flow Types

| Column | Flow Type | Treatment |
|--------|-----------|-----------|
| `pkgs_mm` | middle_mile | Optimize path selection |
| `pkgs_zs` | zone_skip | Same as middle_mile |
| `pkgs_di` | direct_injection | Zone 0, no linehaul |

Combined middle-mile volume: `pkgs_day = pkgs_mm + pkgs_zs`

## Validation Rules

**HARD ERROR**: All `scenario_id` values in scenarios sheet must exist in feasible_paths. Model only runs on scenarios with pre-enumerated paths.

## Output Files

Per scenario:
- `network_opt_{scenario_id}_container.xlsx`
  - summary: KPIs and configuration
  - selected_paths: Chosen path for each OD
  - arc_summary: Arc-level utilization
  - sort_summary: Sort decisions (if enabled)

Multi-scenario:
- `comparison_{run_id}.xlsx`: Side-by-side scenario comparison
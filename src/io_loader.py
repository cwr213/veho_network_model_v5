"""
Input/Output Loader Module

Handles loading Excel workbooks and parsing parameters.
"""

import pandas as pd
from pathlib import Path
from typing import Dict

REQUIRED_SHEETS = [
    "feasible_paths",
    "mileage_bands",
    "cost_params",
    "container_params",
    "package_mix",
    "facilities",
    "scenarios",
    "run_settings",
]


def load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all required sheets from Excel workbook.

    Args:
        path: Path to Excel file

    Returns:
        Dictionary mapping sheet names to DataFrames
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    xlsx = pd.ExcelFile(path)
    available_sheets = set(xlsx.sheet_names)

    missing = set(REQUIRED_SHEETS) - available_sheets
    if missing:
        raise ValueError(f"Missing required sheets: {sorted(missing)}")

    dfs = {}
    for sheet in REQUIRED_SHEETS:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        df = _clean_column_names(df)
        df = _convert_boolean_columns(df, sheet)
        dfs[sheet] = df

    return dfs


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _convert_boolean_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Convert boolean columns from Excel TRUE/FALSE strings."""
    bool_columns = {
        "feasible_paths": ["sla_met", "uses_only_active_arcs"],
        "run_settings": [],  # Handled in params_to_dict
    }

    columns_to_convert = bool_columns.get(sheet_name, [])

    for col in columns_to_convert:
        if col in df.columns:
            df[col] = df[col].apply(_parse_bool)

    return df


def _parse_bool(value) -> bool:
    """Parse boolean from various Excel formats."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().upper() in ("TRUE", "YES", "1", "T", "Y")
    return False


def params_to_dict(df: pd.DataFrame) -> Dict:
    """
    Convert parameter sheet (key/value format) to dictionary.

    Expected columns: key, value
    """
    if "key" not in df.columns or "value" not in df.columns:
        raise ValueError("Parameter sheet must have 'key' and 'value' columns")

    result = {}
    for _, row in df.iterrows():
        key = str(row["key"]).strip().lower()
        value = row["value"]

        # Parse booleans
        if isinstance(value, str) and value.strip().upper() in ("TRUE", "FALSE", "YES", "NO"):
            value = value.strip().upper() in ("TRUE", "YES")

        result[key] = value

    return result


def get_feasible_paths_scenario_ids(df: pd.DataFrame) -> set:
    """Extract unique scenario_ids from feasible_paths."""
    if "scenario_id" not in df.columns:
        raise ValueError("feasible_paths must have 'scenario_id' column")
    return set(df["scenario_id"].dropna().unique())
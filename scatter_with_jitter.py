#!/usr/bin/env python3
"""
Scatter with Jitter for DevSecOps/CI-CD Taxonomy Scores

Reads a CSV file with at least the columns:
    Thema, Avg_Relevanz, Avg_Kompetenz, Avg_Motivation

The script automatically searches for these column headers in the first n rows
of the CSV file (configurable via --header-line-search), allowing you to have
metadata or description rows at the top of your CSV file.

Plots: X=Relevance, Y=Competence, point size ~ Motivation, jitter to reduce overplotting.

Usage:
    python scatter_with_jitter.py /path/to/taxonomy_scores.csv
    python scatter_with_jitter.py /path/to/taxonomy_scores.csv --out /path/to/output.png
    python scatter_with_jitter.py data.csv --annotate-top 10 --jitter 0.06
    python scatter_with_jitter.py data.csv --header-row-search 30
    python scatter_with_jitter.py data.csv --label-all  # Label all bubbles (requires adjustText)
    python scatter_with_jitter.py data.csv --x-max 4 --y-max 4  # Custom axis limits
    python scatter_with_jitter.py --config config.yaml  # Load parameters from YAML
    python scatter_with_jitter.py --sheet-id 1q5L801RbVIpyqmlPKE7xIKxqMkPnsClB0LodUto7VLk  # Download from Google Sheets

Notes:
- Bubble size is proportional to Avg_Motivation values
- Axis limits automatically adjust to data range (no longer fixed at 1-5)
- Use --x-min/--x-max/--y-min/--y-max to manually control axis limits
- No seaborn used, only matplotlib
- No explicit colors specified
- Properly handles CSV files with multi-line quoted cells
- For --label-all option: install adjustText with `pip install adjusttext`
- Can load configuration from YAML file using --config
- Can download CSV directly from Google Sheets using --sheet-id
"""

import argparse
import csv
import os
import sys
import tempfile
import textwrap
from typing import Any

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Optional dependency for label adjustment
from adjustText import adjust_text


def short_label(s: str, width: int = 22) -> str:
    s = str(s) if s is not None else ""
    return textwrap.shorten(s, width=width, placeholder="…")


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse YAML config: {e}", file=sys.stderr)
        sys.exit(1)


def download_google_sheet(sheet_id: str) -> str:
    """Download CSV from Google Sheets and return the content as a string."""
    url = f"https://docs.google.com/spreadsheet/ccc?key={sheet_id}&output=csv"
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as e:
        print(f"ERROR: Failed to download Google Sheet: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scatter with Jitter for taxonomy scores (Relevance/Competence/Motivation)."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help="Path to CSV with columns: Thema, Avg_Relevanz, Avg_Kompetenz, Avg_Motivation. "
        "The script will automatically search for the header row. "
        "Optional if --sheet-id is provided.",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file. CLI arguments override config file values.",
    )
    parser.add_argument(
        "--sheet-id",
        help="Google Sheets ID to download CSV from. "
        "URL format: docs.google.com/spreadsheet/ccc?key=SHEET_ID&output=csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default: <csv_dir>/<csv_stem>_scatter_jitter.png",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.05,
        help="Stddev of jitter added to X/Y to reduce overlap. Default: 0.05",
    )
    parser.add_argument(
        "--annotate-top",
        type=int,
        default=8,
        help="Annotate top-N points by combined score. Default: 8. Use 0 to disable. Ignored if --label-all is set.",
    )
    parser.add_argument(
        "--label-all",
        action="store_true",
        help="Label all data points (requires adjustText package: pip install adjusttext). "
        "Automatically adjusts label positions to minimize overlaps.",
    )
    parser.add_argument(
        "--title",
        default="Taxonomie: Streudiagramm mit Jitter (Größe = Motivation)",
        help="Chart title.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible jitter."
    )
    parser.add_argument(
        "--header-row-search",
        type=int,
        default=20,
        help="Number of initial CSV rows to search for Avg_* column headers. Default: 20",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Minimum value for X-axis (Relevanz). Default: auto-detect from data",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Maximum value for X-axis (Relevanz). Default: auto-detect from data with padding",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Minimum value for Y-axis (Kompetenz). Default: auto-detect from data",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Maximum value for Y-axis (Kompetenz). Default: auto-detect from data with padding",
    )
    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config(args.config)

    # Merge config values with CLI args (CLI takes precedence)
    # For each argument, use CLI value if explicitly set, otherwise use config value
    cli_args = vars(args)
    defaults = vars(parser.parse_args([]))  # Get default values

    for key, value in config.items():
        # Convert YAML keys with underscores to hyphens to match CLI args
        arg_key = key.replace("-", "_")
        if arg_key in cli_args:
            # Only use config value if CLI arg wasn't explicitly set (i.e., it's still the default)
            if cli_args[arg_key] == defaults.get(arg_key):
                cli_args[arg_key] = value

    # Handle sheet-id from config
    if "sheet_id" in cli_args and cli_args["sheet_id"] is None:
        # Check if config has sheet_id (with hyphen or underscore)
        if "sheet-id" in config:
            cli_args["sheet_id"] = config["sheet-id"]
        elif "sheet_id" in config:
            cli_args["sheet_id"] = config["sheet_id"]

    # Determine CSV source
    csv_path = args.csv
    temp_file = None

    if args.sheet_id:
        # Download from Google Sheets
        csv_content = download_google_sheet(args.sheet_id)

        if csv_path:
            # User provided explicit CSV path, write to that location
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(csv_content)
        else:
            # No CSV path provided, use temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            )
            temp_file.write(csv_content)
            temp_file.close()
            csv_path = temp_file.name
    elif not csv_path:
        print(
            "ERROR: Either provide a CSV file path or use --sheet-id to download from Google Sheets",
            file=sys.stderr,
        )
        return 1

    try:
        return _process_csv(csv_path, args)
    finally:
        # Clean up temporary file if created
        if temp_file:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Ignore cleanup errors


def _process_csv(csv_path: str, args: argparse.Namespace) -> int:
    """Process the CSV file and generate the scatter plot."""
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # Find header row by using proper CSV parsing (to handle multi-line cells correctly)
    # Search for the row containing both "Thema" and "Avg_*" columns
    header_row_thema = None
    header_row_avg = None
    required_avg_cols = ["Avg_Relevanz", "Avg_Kompetenz", "Avg_Motivation"]

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                if row_num >= args.header_row_search:
                    break

                # Look for row containing "Thema"
                if header_row_thema is None and "Thema" in row:
                    header_row_thema = row_num

                # Look for row containing all Avg_* columns
                if all(col in row for col in required_avg_cols):
                    header_row_avg = row_num
                    break

    except Exception as e:
        print(f"ERROR: Failed to search for header row: {e}", file=sys.stderr)
        return 1

    # Determine which header row to use
    if header_row_avg is None:
        print(
            f"ERROR: Could not find header row with {required_avg_cols} "
            f"in first {args.header_row_search} rows.",
            file=sys.stderr,
        )
        return 1

    # Check if "Thema" and "Avg_*" are on the same row
    if header_row_thema == header_row_avg:
        # Single-row header - easy case
        header_row = header_row_thema
        use_multi_header = False
    elif header_row_thema is not None and header_row_avg == header_row_thema + 1:
        # Multi-row header: Thema on one row, Avg_* on next row
        header_row = header_row_thema
        use_multi_header = True
    else:
        # Use the Avg row as header
        header_row = header_row_avg
        use_multi_header = False

    # Read the CSV
    try:
        if use_multi_header:
            # Read with multi-row header
            df = pd.read_csv(csv_path, skiprows=range(header_row), header=[0, 1])
            # Flatten multi-index columns
            new_columns = []
            for col in df.columns:
                # col is a tuple like ('Thema', '') or ('Fritz', 'Relevanz')
                if isinstance(col, tuple):
                    # Use the second level if it's not empty/unnamed, otherwise first level
                    if (
                        col[1]
                        and str(col[1]).strip()
                        and not str(col[1]).startswith("Unnamed")
                    ):
                        new_columns.append(col[1])
                    else:
                        new_columns.append(col[0])
                else:
                    new_columns.append(col)
            df.columns = new_columns
        else:
            # Single-row header
            df = pd.read_csv(csv_path, skiprows=range(header_row))

    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        return 1

    # Normalize and validate columns
    required = ["Thema", "Avg_Relevanz", "Avg_Kompetenz", "Avg_Motivation"]
    for col in required:
        if col not in df.columns:
            print(
                f"ERROR: Missing required column '{col}'. Found columns: {list(df.columns)}",
                file=sys.stderr,
            )
            return 1

    # Ensure numeric
    for col in ["Avg_Relevanz", "Avg_Kompetenz", "Avg_Motivation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Avg_Relevanz", "Avg_Kompetenz", "Avg_Motivation"]).copy()

    if df.empty:
        print(
            "ERROR: No valid numeric rows after parsing Relevance/Competence/Motivation.",
            file=sys.stderr,
        )
        return 1

    # Derive output path
    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(csv_path)), f"{stem}_scatter_jitter.png"
        )

    # Jitter
    rng = np.random.default_rng(args.seed)
    jx = rng.normal(0, args.jitter, size=len(df))
    jy = rng.normal(0, args.jitter, size=len(df))

    x = df["Avg_Relevanz"].to_numpy() + jx
    y = df["Avg_Kompetenz"].to_numpy() + jy

    # Bubble size scaled by Motivation
    mot = df["Avg_Motivation"].to_numpy()
    max_mot = np.nanmax(mot) if len(mot) else 1.0
    sizes = 100.0 * (mot / max_mot) + 30.0  # base size + scaling

    # Calculate axis limits
    # Auto-detect from data with padding if not specified
    x_data_min = np.nanmin(df["Avg_Relevanz"])
    x_data_max = np.nanmax(df["Avg_Relevanz"])
    y_data_min = np.nanmin(df["Avg_Kompetenz"])
    y_data_max = np.nanmax(df["Avg_Kompetenz"])

    # Use specified values or calculate from data
    x_min = (
        args.x_min if args.x_min is not None else max(1.0, np.floor(x_data_min - 0.5))
    )
    x_max = args.x_max if args.x_max is not None else np.ceil(x_data_max + 0.5)
    y_min = (
        args.y_min if args.y_min is not None else max(1.0, np.floor(y_data_min - 0.5))
    )
    y_max = args.y_max if args.y_max is not None else np.ceil(y_data_max + 0.5)

    # Generate tick marks (use 0.5 or 1.0 increments depending on range)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_step = 0.5 if x_range <= 4 else 1.0
    y_step = 0.5 if y_range <= 4 else 1.0

    x_ticks = np.arange(x_min, x_max + x_step / 2, x_step)
    y_ticks = np.arange(y_min, y_max + y_step / 2, y_step)

    # Plot
    plt.figure(figsize=(7, 7))
    # No explicit colors or styles
    plt.scatter(x, y, s=sizes, alpha=0.6, edgecolors="none")

    plt.xlabel("Relevanz")
    plt.ylabel("Kompetenz")
    plt.title(args.title)
    plt.xlim(x_min - 0.1, x_max + 0.1)  # Add small margin
    plt.ylim(y_min - 0.1, y_max + 0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.grid(True, linewidth=0.5, alpha=0.3)

    # Annotate points
    if args.label_all:
        # Create annotations for all points
        texts = []
        for idx in df.index:
            i = df.index.get_loc(idx)
            # Use full label text without trimming
            label = (
                str(df.loc[idx, "Thema"]) if df.loc[idx, "Thema"] is not None else ""
            )
            text = plt.annotate(label, (x[i], y[i]), fontsize=5)
            texts.append(text)

        # Adjust text positions to minimize overlaps
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
            expand_points=(1.2, 1.2),
            force_text=(0.5, 0.5),
            force_points=(0.2, 0.2),
        )
    elif args.annotate_top and args.annotate_top > 0:
        # Annotate top-N by combined score (Relevance + Competence + Motivation)
        df["_score"] = df["Avg_Relevanz"] + df["Avg_Kompetenz"] + df["Avg_Motivation"]
        top = df.nlargest(args.annotate_top, "_score")
        # map indices to coords
        for idx in top.index:
            # find position in x,y arrays (same order as df)
            i = df.index.get_loc(idx)
            label = short_label(df.loc[idx, "Thema"], 24)
            plt.annotate(
                label,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

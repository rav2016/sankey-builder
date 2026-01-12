# Sankey Builder – User Guide

## Overview
Sankey Builder is a local web utility for creating, validating, and exporting Sankey diagrams using CSV-based flow data. It runs entirely on your machine.

## Start the app
```bash
conda activate sankey312
streamlit run app.py
```
Open: `http://localhost:8501`

## Flow data (flows.csv)
Each row is a flow:

Required:
- `source` (string)
- `target` (string)
- `value` (number, non-negative)

Optional:
- `link_label` (string hover text)
- `link_color` (e.g., `rgba(0,0,0,0.25)`)

**Tip:** If any text field contains commas, wrap it in double quotes in the CSV.

## Node layout (nodes.csv) – optional
Use this to control layout when fixed layout is enabled.

Columns:
- `node`
- `level` (0..N, left→right)
- `order` (top→bottom within a level)

Example:
```csv
node,level,order
Input,0,0
Process A,1,0
Process B,1,1
Output,2,0
```

## Main UI areas

### Datasets (sidebar)
Manage multiple charts:
- Add new
- Duplicate
- Rename
- Delete
- Switch active dataset

### Flow table
Edit flows directly in the table:
- add/remove rows
- update values
- chart updates automatically

### Validation
Checks for:
- missing fields
- non-numeric values
- negative values
- zero values
- duplicates
- nodes with only incoming or outgoing flows

### Aggregate duplicate edges
When enabled, repeated `(source, target)` rows are summed for rendering.

### Fixed layout (levels)
When enabled:
- assign node `level` and `order`
- use “Auto levels” for a quick starting layout
- import `nodes.csv` for consistent formatting

## Import / Export
- Import flows CSV
- Import node layout CSV
- Export flows CSV
- Export node layout CSV
- Export chart as HTML
- Export chart as PNG
- Export project JSON (reload later)

## Known warnings
- `missing ScriptRunContext`: Streamlit internal warning; safe to ignore.

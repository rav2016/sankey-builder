<<<<<<< HEAD
# sankey-builder
A local web utility to build Sankey diagrams from CSV.
=======
# Sankey Builder

Sankey Builder is a local web-based utility for creating, validating, and exporting Sankey diagrams using simple CSV data.

It supports interactive editing, duplicate edge aggregation, fixed layout control (node levels), and export to PNG, HTML, and project files.

📘 See [User Guide](docs/USER_GUIDE.md)  
🗺️ See [Roadmap](ROADMAP.md)

## Features
- Editable table-based flow input
- CSV import / export
- Duplicate edge aggregation
- Fixed layout using node levels and ordering
- Live validation
- PNG and HTML export
- Project save/load (`.json`)
- Multiple datasets per session
- Node-layout CSV import/export (`nodes.csv`)

## Requirements
- Python 3.12+
- Conda or virtualenv recommended

## Installation (Conda)
```bash
conda create -n sankey312 python=3.12 -y
conda activate sankey312
pip install -r requirements.txt
```

## Run
```bash
conda activate sankey312
streamlit run app.py
```

Then open:
```
http://localhost:8501
```

## CSV Format

### Flows CSV
Required columns:
- `source`
- `target`
- `value`

Optional columns:
- `link_label`
- `link_color`

If a field contains commas, wrap it in quotes.

### Node Layout CSV (optional)
Columns:
- `node`
- `level`
- `order`

Example:
```csv
node,level,order
Input,0,0
Process A,1,0
Process B,1,1
Output,2,0
```

## Export Formats
- CSV (flows)
- CSV (node layout)
- HTML (interactive)
- PNG (static image)
- JSON (project)

## Examples
Sample CSV files are available in the [examples](examples/) folder.

## License
MIT
>>>>>>> 0302d58 (Add node-layout CSV import and repo structure)

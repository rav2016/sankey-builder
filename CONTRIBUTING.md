# Contributing to Sankey Builder

Thank you for your interest in contributing to Sankey Builder.

This project aims to stay:
- simple
- predictable
- dependency-light
- usable by non-developers

Please keep those principles in mind when contributing.

## Getting Started

### Requirements
- Python 3.12+
- Conda or virtualenv
- Git

### Setup
```bash
git clone https://github.com/rav2016/sankey-builder.git
cd sankey-builder
conda create -n sankey312 python=3.12 -y
conda activate sankey312
pip install -r requirements.txt
streamlit run app.py
```

## How to Contribute

1. Fork the repository and create a feature branch:
```bash
git checkout -b feature/my-change
```

2. Make changes:
- keep them focused and minimal
- avoid heavy dependencies
- keep UI and docs generic

3. Test manually:
- CSV import/export (flows and nodes)
- aggregation toggle
- fixed layout

4. Submit a Pull Request:
- clear title
- explain why the change is needed
- include screenshots only if UI changes

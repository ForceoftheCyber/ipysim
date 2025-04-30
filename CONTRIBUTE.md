# Contributing

## Initial setup and installing dependecies

Start by making and activating a Python virtual environment:

Windows:
```bash
python -m venv ".venv"
.\.venv\Scripts\activate
```

*For Linux or Mac, replace `.\.venv\Scripts\activate` with `source .venv/Scripts/activate`.*

Install the requirements:
```bash
python -m pip install -r requirements.txt
```

## Publish to PyPi
Update version number in [pyproject.toml](./pyproject.toml)

Build:
```bash
python -m pip install --upgrade build
python -m build
```

Upload:
```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Then provide API token.

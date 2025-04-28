# Contributing

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

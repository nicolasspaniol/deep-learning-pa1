test:
    PYTHONPATH=src uv run pytest

run $SCRIPT:
    PYTHONPATH=src uv run scripts/$SCRIPT.py

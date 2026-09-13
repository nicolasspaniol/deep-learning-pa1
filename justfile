test:
    PYTHONPATH=src uv run pytest

run $SCRIPT *args:
    PYTHONPATH=src uv run scripts/$SCRIPT.py {{args}}

typecheck:
    uvx ty check

test:
    uv run pytest

run $SCRIPT *args:
    uv run scripts/$SCRIPT.py {{args}}

typecheck:
    uvx ty check

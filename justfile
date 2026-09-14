test:
    uv run pytest


run $SCRIPT *args:
    uv run scripts/$SCRIPT.py {{args}}


typecheck:
    uvx ty check


download_data $STAGE:
    curl -L -o ./data/stage1_$STAGE.zip https://data.broadinstitute.org/bbbc/BBBC038/stage1_$STAGE.zip
    unzip -q data/stage1_$STAGE.zip -d data/stage1_$STAGE
    rm ./data/stage1_$STAGE.zip

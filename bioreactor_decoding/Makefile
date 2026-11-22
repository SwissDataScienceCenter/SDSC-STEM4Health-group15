#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = bioreactor_decoding
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Rebuild every out-of-date model, metric and submission (DVC pipeline)
.PHONY: pipeline
pipeline:
	PATH="$(CURDIR)/.venv/bin:$$PATH" dvc repro

## Show CV and holdout MAPE of every model in the pipeline
.PHONY: metrics
metrics:
	PATH="$(CURDIR)/.venv/bin:$$PATH" dvc metrics show


MAMBA_ENV = /home/renku/work/.micromamba/envs/bioreactor-presentation

## Install the presentation group (Manim), built against presentation/environment.yml
.PHONY: presentation-env
presentation-env:
	PATH="$(MAMBA_ENV)/bin:$$PATH" PKG_CONFIG_PATH="$(MAMBA_ENV)/lib/pkgconfig" \
	LDFLAGS="-Wl,-rpath,$(MAMBA_ENV)/lib" uv sync --group presentation

## Render the presentation videos (holdout traces of three models) to presentation/media/
.PHONY: videos
videos:
	.venv/bin/python presentation/export_traces.py
	cd presentation && for m in mechanistic_lite ml_sensors hybrid_blend; do \
		MODEL=$$m ../.venv/bin/manim -qh --media_dir media -o $$m holdout_traces.py HoldoutTraces; \
	done


## Render the Quarto report reports/report.html from the DVC-tracked metrics and figures
.PHONY: report
report:
	cd reports && QUARTO_PYTHON="$(CURDIR)/.venv/bin/python" ../.venv/bin/quarto render report.qmd


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) bioreactor_decoding/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

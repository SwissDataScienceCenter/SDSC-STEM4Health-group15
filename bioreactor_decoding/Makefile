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
	.venv/bin/ruff format --check
	.venv/bin/ruff check

## Format source code with ruff
.PHONY: format
format:
	.venv/bin/ruff check --fix
	.venv/bin/ruff format



## Run tests
.PHONY: test
test:
	.venv/bin/python -m pytest tests


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
	cd presentation && for m in mechanistic_lite ml_sensors option_a; do \
		MODEL=$$m ../.venv/bin/manim -qh --media_dir media -o $$m holdout_traces.py HoldoutTraces; \
	done


## Render the Quarto report reports/report.html from the DVC-tracked metrics and figures
.PHONY: report
report:
	cd reports && QUARTO_PYTHON="$(CURDIR)/.venv/bin/python" ../.venv/bin/quarto render report.qmd


## Render the percent-error videos (one approach added per video) to presentation/media/
.PHONY: error-videos
error-videos:
	.venv/bin/python presentation/export_traces.py
	cd presentation && for st in 1 2 3; do \
		STAGE=$$st ../.venv/bin/manim -qh --media_dir media -o errors_$$st holdout_errors.py ErrorTraces; \
	done


## Render the hero video (presentation/hero.py) and a still of its end card
.PHONY: hero-video
hero-video:
	.venv/bin/python presentation/export_traces.py
	cd presentation && ../.venv/bin/manim -qh --media_dir media -o hero hero.py Hero \
		&& ../.venv/bin/manim -qh -s --media_dir media -o hero_end_card hero.py Hero


## Build the 5-minute deck presentation/slides/presentation.pptx (after make hero-video)
.PHONY: slides
slides:
	.venv/bin/python presentation/slides/build_slides.py


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

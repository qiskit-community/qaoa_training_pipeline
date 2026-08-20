OS := $(shell uname -s)

ifeq ($(OS), Linux)
  NPROCS := $(shell grep -c ^processor /proc/cpuinfo)
else ifeq ($(OS), Darwin)
  NPROCS := 2
else
  NPROCS := 0
endif # $(OS)

ifeq ($(NPROCS), 2)
	CONCURRENCY := 2
else ifeq ($(NPROCS), 1)
	CONCURRENCY := 1
else ifeq ($(NPROCS), 3)
	CONCURRENCY := 3
else ifeq ($(NPROCS), 0)
	CONCURRENCY := 0
else
	CONCURRENCY := $(shell echo "$(NPROCS) 2" | awk '{printf "%.0f", $$1 / $$2}')
endif

.PHONY: lint style black test test_ci coverage clean

all_check: style lint

lint:
	python -m ruff check qaoa_training_pipeline tests

black:
	python -m black qaoa_training_pipeline tests

style:
	python -m black --check qaoa_training_pipeline tests
	python -m ruff check qaoa_training_pipeline tests

test:
	python -m unittest discover -v tests

test_ci:
	echo "Detected $(NPROCS) CPUs running with $(CONCURRENCY) workers"
	python -m stestr run --concurrency $(CONCURRENCY)

coverage:
	python -m coverage3 run --source qaoa_training_pipeline -m unittest discover -s tests -q
	python -m coverage3 report

coverage_erase:
	python -m coverage erase

clean: coverage_erase;

clean:
    rm -rf build
    rm -rf dist
    rm -rf treemendous.egg-info
    rm -rf setup.py

build:
    poetry build

test:
    poetry run pytest

test-perf: build
    python tests/performance/boundry_vs_avl.py


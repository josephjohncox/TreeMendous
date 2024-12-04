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

version:
    @poetry version -s

release: build
    @poetry version -s | xargs -I {} gh release create v{} --generate-notes

delete-release: delete-tag
    @poetry version -s | xargs -I {} gh release delete v{}

delete-tag:
    @poetry version -s | xargs -I {} git tag -d v{}
    @poetry version -s | xargs -I {} git push origin :refs/tags/v{}


bump-patch:
    poetry version patch
    
bump-minor:
    poetry version minor

bump-major:
    poetry version major


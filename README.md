# stistools

[![Documentation Status](https://readthedocs.org/projects/stistools/badge/?version=latest)](https://stistools.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ye-spencer/stistools-sdetfp/graph/badge.svg)](https://codecov.io/gh/ye-spencer/stistools-sdetfp)

[![Coverage sunburst](https://codecov.io/gh/ye-spencer/stistools-sdetfp/graphs/sunburst.svg)](https://codecov.io/gh/ye-spencer/stistools-sdetfp)

## For SDETFP

### Running Blackbox and Whitebox tests locally

```bash
# First-time setup
pip install -e ".[test]"

# Run tests with statement and branch coverage
python -m pytest --cov=stistools --cov-branch --cov-report=html --cov-report=term-missing

# Then open `htmlcov/index.html` locally to view coverage results for your branch. 
```

### Running Mutating tests locally
```bash
# First-time setup
pip install -e ".[test]"

# If you get a permissions error when running mutmut, fix it first:
chmod -R u+w tests/ tests_sdet/
rm -rf .mutmut-cache mutants/

# To run mutation tests on a specific file, edit the [tool.mutmut] section of pyproject.toml:
toml[tool.mutmut]
paths_to_mutate = ["stistools/<file_being_tested.py>"]
tests_dir = ["tests_sdet"]
also_copy = ["tests_sdet"]
runner = "python -m pytest -o addopts= -x -q tests_sdet/<test_file.py>"
do_not_mutate = ["stistools/version.py", "stistools/__init__.py"]
debug = true
# If pyproject.toml config is not picked up, create a setup.cfg file instead:
[mutmut]
paths_to_mutate = stistools/<file_being_tested.py>
tests_dir = tests_sdet
also_copy = tests_sdet
runner = python -m pytest -o addopts= -x -q tests_sdet/<test_file.py>
do_not_mutate = stistools/version.py,stistools/__init__.py
debug = true

# Then run:
bashmutmut run

# View summary score
mutmut results

# See all surviving mutants
mutmut results --all True

# Inspect a specific surviving mutant
mutmut show <mutant_id>

# Start a completely fresh run
rm -f .mutmut-cache
mutmut run
```

### Running Documentation tests locally
```bash
# First-time setup
pip install -e ".[test]"

# Check documentation coverage (fail if below 75%)
interrogate -vv stistools/ --fail-under 75

# Check docstring style
pydocstyle --count stistools/
```

## Original Readme

Tools for HST/STIS.

Code Contribution Guide:

- For new additions to stistools, a new branch off the main repository is encouraged.  Use initials for the beginning of the branch title. It is also acceptable to put a PR in from a fork of stistools (necessary for an external contributor).

- Each PR should have at least one approved review by at least one STIS team member AND one DATB/SCSB member (this could be either Sara or Robert).

- After approved reviews, test your new content with readthedocs.  You can do this by pushing to the doc_updates_rtd branch.  This branch is setup to re-build the https://stistools.readthedocs.io/en/doc_updates_rtd/ page after any new commits.

## Documentation
Minimum requirement to have at least some inline numpy style API docstrings.  PR author should also review narrative docs to make sure those are appropriately updated. Any new tasks will need a new rst file for sphinx/rtd to pick up the new docs.

## Testing
New functions and or new functionality should have appropriate unit tests.  Tests that use any input and or truth files, will need to use artifactory for hosting test files.

## Pep 8
Try to adhere to pep 8 standards when reasonable.  Code comments are heartily encouraged!



# Running blackbox and whitebox tests locally

## First-time setup
pip install -e ".[test]"

## Run tests with statement AND branch coverage
python -m pytest --cov=stistools --cov-branch --cov-report=html --cov-report=term-missing

## Then open htmlcov/index.html to view full coverage breakdown

# Running mutation tests locally

## First-time setup (already done above)

## Run mutation testing on the package (scope to source, not tests)
mutmut run stistools/

## View summary score
mutmut results

## See every surviving mutant
mutmut results --all True

## Inspect a specific surviving mutant (substitute the ID from results)
mutmut show <mutant_id>

## To start a fresh run and clear cached results
rm -f .mutmut-cache
mutmut run stistools/

# Running documentation tests locally

## Check documentation coverage (fail if below 75%)
interrogate -vv stistools/ --fail-under 75

## Check docstring style
pydocstyle --count stistools/

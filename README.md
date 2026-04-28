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

# Run mutating tests 
mutmut run

# View summary score
mutmut results

# See every surviving mutant
mutmut results --all True

# Inspect a specific surviving mutant (substitute the ID from results)
mutmut show <mutant_id>

# To start a fresh run and clear cached results
rm -f .mutmut-cache
mutmut run stistools/
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

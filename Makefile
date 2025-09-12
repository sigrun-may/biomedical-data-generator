# Copyright (c) 2022 Sigrun May,
# Ostfalia Hochschule für angewandte Wissenschaften
#
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MITsrc := biomedical_data_generator

other-src := examples

# check the code
check:
	pydocstyle --count $(src) $(other-src) $(test-src)
	black $(src) $(other-src) $(test-src) --check --diff
	flake8 $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(other-src) $(test-src)
	pylint $(src) $(other-src)

# format the code
format:
	black $(src) $(other-src) $(test-src)
	isort $(src) $(other-src) $(test-src)
	mdformat *.md

install:
	poetry lock && poetry install --all-extras

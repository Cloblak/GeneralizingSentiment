install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black scripts/*.py data/*.py

lint:
	python -m pylint --disable=R,C scripts/*.py --extension-pkg-whitelist=cv2

clean: format lint
all: install format lint

init:
	pip install -r requirements.txt
test:
	python -m unittest discover -s tests

.PHONY: init test
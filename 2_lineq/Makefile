PYTHON = python3

.PHONY: all
all : out-A.txt out-B.txt

out-A.txt : main-A.py
	$(PYTHON) main-A.py > out-A.txt

out-B.txt : main-B.py
	$(PYTHON) main-B.py > out-B.txt

.PHONY: clean
clean:
	rm -rf out-A.txt out-B.txt __pycache__

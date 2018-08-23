PYTHON = python3

.PHONY : all
all : out-A.txt out-B.txt out-C.txt

out-A.txt : main-A.py
	$(PYTHON) main-A.py > out-A.txt

out-B.txt : main-B.py
	$(PYTHON) main-B.py > out-B.txt

out-C.txt : main-C.py
	$(PYTHON) main-C.py > out-C.txt

.PHONY: clean
clean :
	rm -rf *.txt __pycache__

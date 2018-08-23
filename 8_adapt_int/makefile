PYTHON = python3

.PHONY: all
all: out.txt

out.txt : main.py
	$(PYTHON) main.py > out.txt

.PHONY: clean
clean:
	rm -rf *.txt __pycache__

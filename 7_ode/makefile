PYTHON = python3

.PHONY: all
all: out.txt
	gnuplot plot.gpi

out.txt : main.py
	$(PYTHON) main.py > out.txt

.PHONY: clean
clean:
	rm -rf *.txt *.svg __pycache__

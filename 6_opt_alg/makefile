PYTHON = python3

.PHONY: all
all: out.txt plotdata.txt
	gnuplot plot.gpi

plotdata.txt : printdata.py
	$(PYTHON) printdata.py > plotdata.txt

out.txt : main.py
	$(PYTHON) main.py > out.txt

.PHONY: clean
clean:
	rm -rf *.txt *.svg __pycache__

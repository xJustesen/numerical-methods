PYTHON = python3

.PHONY: all
all : out-A.txt out-B.txt times.txt

out-A.txt : main-A.py
	$(PYTHON) main-A.py > out-A.txt

out-B.txt : main-B.py
	$(PYTHON) main-B.py > out-B.txt

times.txt : timer.py
	str='Number of CPU-seconds used when doing cyclic sweeps:';\time -ao $@ -f "$$str %U" $(PYTHON) timer.py sweeps $$str > /dev/null
	str='Number of CPU-seconds used when doing value-by-value:';\time -ao $@ -f "$$str %U" $(PYTHON) timer.py values $$str > /dev/null

.PHONY: clean
clean:
	rm -rf *.txt __pycache__

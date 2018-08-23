PYTHON = python3

.PHONY : all
all : out_A.txt out_B.txt out_C.txt

out_A.txt : main_A.py
	$(PYTHON) $< > $@

out_B.txt : main_B.py
	$(PYTHON) $< > $@

out_C.txt : main_C.py
	$(PYTHON) $< > $@

.PHONY: clean
clean :
	rm -rf *.txt __pycache__

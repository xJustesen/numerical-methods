.PHONY: A
A : out-A.txt out-B.txt
	gnuplot plot-A.gpi
	gnuplot plot-B.gpi

out-A.txt : main-A.py
	python3 main-A.py > out-A.txt

out-B.txt : main-B.py
	python3 main-B.py > out-B.txt

.PHONY: clean
clean:
	rm -f out-A.txt out-B.txt plot-A.svg plot-B.svg

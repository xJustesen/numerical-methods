.PHONEY: A
A : out.A.txt

out.A.txt : main-A.py
	python3 main-A.py > out.A.txt

.PHONEY: B
B : out.B.txt

out.B.txt : main-B.py
	python3 main-B.py > out.B.txt

.PHONEY: clean
clean:
	rm -f main-A out.A.txt out.B.txt correct.txt

check : out.A.txt correct.txt
	@echo "Checking target A ..."
	@diff --brief out.A.txt correct.txt
	@echo "target A seems to be ok ..."
	@echo "Checking target B ..."
	@diff --brief out.B.txt correct.txt
	@echo "target B seems to be ok ..."

correct.txt :
	echo "Hello, $$(whoami)" > correct.txt

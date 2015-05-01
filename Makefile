report.pdf:untitled.txt
	pdflatex report

untitled.txt:
	python submit.py

.PHONY: clean
all:report.pdf
clean:
	rm -f submit.pdf

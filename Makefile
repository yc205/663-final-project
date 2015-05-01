report.pdf:submit.py
	pdflatex report

submit.py:
	python submit.py

.PHONY: clean
all:report.pdf
clean:
	rm -f submit.pdf

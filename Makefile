submit.pdf: submit.py
	ipython nbconvert --to latex --post PDF submit.py


.PHONY: clean

clean:
	rm -f submit.pdf

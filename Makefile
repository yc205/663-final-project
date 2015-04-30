
submit.pdf: submit.ipynb
	ipython nbconvert --to latex --post PDF submit.ipynb

.PHONY: clean

clean:
	rm -f submit.pdf


Yu_Chen_Project.pdf: submit.ipynb
	ipython nbconvert --to latex --post PDF submit.ipynb

.PHONY: clean

clean:
	rm -f Yu_Chen_Project.pdf

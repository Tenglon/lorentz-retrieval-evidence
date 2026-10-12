PYTHON ?= python3

.PHONY: test assets slides reproduce clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) scripts/generate_assets.py --check

assets:
	$(PYTHON) scripts/generate_assets.py

slides: assets
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex

reproduce: test slides

clean:
	rm -f slides/*.aux slides/*.bbl slides/*.blg slides/*.log slides/*.nav \
		slides/*.out slides/*.snm slides/*.toc slides/*.vrb slides/main.pdf

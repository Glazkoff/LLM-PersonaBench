#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
(
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  for pass in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex; done
)
(
  cd supplement
  for doc in cached-search prospective-diagnostic acquisition; do
    pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex"
    pdflatex -interaction=nonstopmode -halt-on-error "$doc.tex"
  done
  pdfunite controls.pdf robustness.pdf extended.pdf external.pdf historical.pdf cached-search.pdf prospective-diagnostic.pdf acquisition.pdf ../supplement.pdf
)

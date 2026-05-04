# Project Report Template

This directory contains a LaTeX template for a workshop- or conference-style
project report.

## Files

- `report.tex`: main report template.
- `references.bib`: starter bibliography file.

## Build

From this directory:

```bash
latexmk -pdf report.tex
```

If `latexmk` is unavailable:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

## Reproducibility Requirements

The final project should include:

- Python 3.11 or a clearly documented Python 3 version.
- PyTorch, unless there is a documented reason not to use it.
- A `requirements.txt` file with all pip dependencies.
- Clear documentation for non-pip dependencies, if any.
- A one-line Python command that reproduces the experiments, also documented in
  the project `README.md`.

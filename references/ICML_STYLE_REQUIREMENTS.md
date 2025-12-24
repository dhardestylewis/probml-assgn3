# ICML 2024 Style Requirements
**Created:** 2025-12-23
**Source:** Based on ICML formatting standards

---

## Download Location

**Official ICML 2024 Style Package:**
https://media.icml.cc/Conferences/ICML2024/Styles/icml2024.zip

**Note:** Direct download failed in current environment. User will need to download manually and place in project directory.

## Required Files

1. `icml2024.sty` - Main style file
2. `icml2024.bst` - Bibliography style file
3. Example paper template

## Format Specifications

### Paper Length
- **Main paper:** 8 pages maximum (including figures, tables)
- **References:** Unlimited additional pages
- **Appendix:** Unlimited additional pages

### Column Format
- **Two-column layout** (standard for ICML)
- Column width and spacing defined by style file

### Font and Size
- **Body text:** 10pt
- **Title:** Large, bold
- **Section headers:** Bold, numbered
- **Math:** Computer Modern or Times

### Title and Authors
Use ICML macros:
- `\icmltitle{Your Title Here}`
- `\icmlauthor{Author Name}{Affiliation}`
- `\icmladdress{Full Address}`
- `\icmlcorrespondingauthor{Name}{email@domain.edu}`

### Abstract
- Single paragraph recommended
- No citations in abstract
- 150-250 words typical

### Sections
- Numbered sections: Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion
- Unnumbered: Abstract, References, Appendix

### Figures and Tables
- Place in column or span both columns (`figure*`, `table*`)
- Captions below figures, above tables
- Use `\caption{...}` and `\label{...}`

### References
- Use `icml2024.bst` for BibTeX
- Author-year citation style (e.g., "Murphy, 2012")
- Alphabetical order

### Equations
- Number important equations
- Use `align` or `equation` environments
- Define all symbols on first use

### Code and Algorithms
- Use `algorithm` package with `algorithmic`
- Pseudocode should be readable

## LaTeX Preamble Example

```latex
\documentclass{article}
\usepackage[accepted]{icml2024}
% Option: [accepted] for camera-ready, omit for submission

\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}

\icmltitlerunning{Short Title for Header}

\begin{document}

\icmltitle{Full Paper Title Here}

\icmlauthor{Your Name}{Your Institution}
\icmladdress{Department, Institution, City, Country}
\icmlcorrespondingauthor{Your Name}{email@domain.edu}

\icmlkeywords{Machine Learning, ICML, probabilistic models}

\vskip 0.3in
```

## Comparison to Current Report

### Current Report (report.tex)
- Format: Single-column article (`\documentclass[11pt]{article}`)
- Margins: 1 inch all around (`\geometry{margin=1in}`)
- Title: Centered text block
- Length: 10 pages

### Required Changes for ICML
1. **Document class:** Change to `\documentclass{article}` with `\usepackage{icml2024}`
2. **Remove custom geometry:** ICML style controls layout
3. **Title block:** Use `\icmltitle` and `\icmlauthor` macros
4. **References:** Convert to author-year format if not already
5. **Length check:** Ensure main content ≤ 8 pages (excluding references)

## Action Items

1. **Download icml2024.zip** from official ICML link (manual download required)
2. **Extract to project directory:** Place `icml2024.sty` and `icml2024.bst` in `hws/hw3.d/`
3. **Create ICML version:** Save `report.tex` as `report_icml.tex`
4. **Update preamble:** Replace geometry with `\usepackage[accepted]{icml2024}`
5. **Convert title/authors:** Use ICML macros
6. **Compile:** Run `pdflatex` and `bibtex` multiple times
7. **Verify:** Check page count, formatting, references

## Verification Commands

```bash
# After manual download and extraction
cd /home/user/probml-assgn3/hws/hw3.d/
ls -la icml2024.sty icml2024.bst

# Compile ICML version
pdflatex report_icml.tex
bibtex report_icml
pdflatex report_icml.tex
pdflatex report_icml.tex

# Check page count
pdfinfo report_icml.pdf | grep Pages
```

## Resources

- **ICML 2024 Downloads:** https://icml.cc/Downloads
- **Author Instructions:** https://icml.cc/Conferences/2024/AuthorInstructions
- **Example Paper:** https://media.icml.cc/Conferences/ICML2024/Styles/example_paper.pdf
- **Overleaf Template:** Available for online editing

## Notes

- ICML style file download requires manual intervention in current environment
- Style file not critical for course submission, but good practice for final version
- Can continue with current single-column format and apply ICML style later
- Main priority: Content accuracy, symbol removal, critique resolution

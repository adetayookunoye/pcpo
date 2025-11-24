# Paper Formatting Guide: From AISTAT to Other Venues

## Quick Reference Table

| Venue | Format | Modifications | Status |
|-------|--------|---------------|--------|
| **AISTAT 2024** | Current | None needed | ✅ Ready |
| **NeurIPS 2024** | Anonymous | 1 package change | ⚡ 5 min |
| **ICML 2024** | Anonymous | 1 package change | ⚡ 5 min |
| **arXiv** | Preprint | Optional tweaks | ✅ Ready |
| **SIAM Review** | Journal | 3-4 changes | ⏳ 15 min |
| **IEEE TPAMI** | Journal | 2-3 changes | ⏳ 15 min |
| **Computational Methods** | Journal | Minor edits | ⏳ 15 min |

## AISTAT Format (Current)

Your paper is ready in AISTAT format. No changes needed.

**Key features**:
- 2-column layout
- Anonymous submission format
- 8-12 page limit (+ references)
- Technical track

**To submit**:
1. Compile: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
2. Create PDF: `main.pdf`
3. Upload to conference system

## NeurIPS 2024

### Changes Required

**Step 1**: Replace line 29 in `main.tex`:
```latex
% OLD:
\usepackage[accepted]{aistats2024}

% NEW:
\usepackage{neurips_2024}
```

**Step 2**: Update preamble for NeurIPS format (~line 2):
```latex
% For NeurIPS, keep:
\documentclass[12pt]{article}
\usepackage{neurips_2024}

% Remove/comment out AISTAT-specific packages
% (most overlap with NeurIPS packages anyway)
```

**Step 3**: Remove author names for anonymous review (line 15-19):
```latex
% BEFORE:
\author{
  Adetayo Okunoye \\
  Department of Computer Science \\
  \texttt{adetayo@example.edu}
}

% AFTER (anonymous):
\author{Anonymous submission}
```

**Step 4**: Add header for NeurIPS (after \begin{document}):
```latex
\begin{document}

% For NeurIPS camera-ready, add:
\def\year{2024}
\def\confname{NeurIPS}
```

### Compile

```bash
# Download NeurIPS template (if needed)
# From: https://github.com/NICTA/neurips_template

# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Submission Checklist for NeurIPS
- [ ] Remove author information
- [ ] No identifying information (institution names, prior work links)
- [ ] PDF is anonymized
- [ ] Page count: ≤ 8 pages + references
- [ ] Create supplementary.pdf for appendices
- [ ] .zip file contains: main.pdf, supplement.pdf, references.bib

## ICML 2024

### Changes Required

Almost identical to NeurIPS:

**Step 1**: Replace line 29:
```latex
\usepackage{icml2024}
```

**Step 2**: Update document class if needed:
```latex
\documentclass{article}
\usepackage{icml2024}
```

**Step 3**: Anonymize (same as NeurIPS):
```latex
\author{Anonymous submission}
```

### Key Differences from NeurIPS
- ICML is slightly more restrictive on page count (typically 8 pages)
- References can exceed page limit
- Supplement submissions separate

## arXiv Submission

Your paper is already compatible with arXiv.

### Steps

1. **Compile locally** (ensure no errors):
   ```bash
   pdflatex -interaction=nonstopmode main.tex
   bibtex main
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex
   ```

2. **Create tarball** with source files:
   ```bash
   tar -czf paper.tar.gz main.tex references.bib *.sty
   # OR include all:
   tar -czf paper.tar.gz main.tex references.bib
   ```

3. **Upload to arXiv**:
   - Go to https://arxiv.org/submit
   - Select "Artificial Intelligence"
   - Upload tarball
   - arXiv auto-compiles

### arXiv Optimization

To make paper look best on arXiv:

**Add metadata** (in preamble before \begin{document}):
```latex
% For arXiv submission
\usepackage{hyperref}
\hypersetup{
  pdftitle={Stream Function Neural Operators with Probabilistic Inference},
  pdfauthor={Adetayo Okunoye},
  pdfkeywords={neural operators, constraints, uncertainty quantification}
}
```

**Recommended arXiv category**: `cs.LG` (Machine Learning)
or `math.NA` (Numerical Analysis)

## Journal Format: SIAM Review

### Key Differences

SIAM journals have different:
- Page layout (single column)
- Title page format
- Section numbering
- Appendix handling

### Changes Required

**Step 1**: Replace document class (line 2):
```latex
% OLD:
\documentclass[12pt]{article}

% NEW:
\documentclass[11pt,onecolumn]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx}
```

**Step 2**: Remove AISTAT/conference package (line 29):
```latex
% OLD:
\usepackage[accepted]{aistats2024}

% NEW:
% (delete this line - using standard article class)
```

**Step 3**: Update title page (lines 15-19):
```latex
\title{Stream Function Neural Operators with Probabilistic Inference:\\
Guaranteed Physical Constraints and Multi-Scale Learning}

\author{
  Adetayo Okunoye\affil{1}
}

\affiliation{
  \affil{1} Your Institution, Department, Email
}

\date{\today}
```

**Step 4**: Add SIAM-style abstract (after \maketitle):
```latex
\begin{abstract}
\noindent
[Your abstract - same content]
\end{abstract}

\keywords{neural operators, constraints, PDE surrogates}
```

**Step 5**: Reduce font size and adjust spacing:
```latex
% In preamble:
\usepackage{setspace}
\onehalfspacing  % SIAM typically wants 1.5 spacing
```

### Submission Process for SIAM

1. Create account at https://www.siam.org/publications/journals/
2. Choose journal (e.g., "SIAM Review")
3. Upload files:
   - main.tex
   - references.bib
   - main.pdf (compiled)
   - Any figures separately

## Journal Format: IEEE TPAMI

### Key Differences

IEEE TPAMI is more restrictive:
- Single column
- Specific header/footer format
- Different reference style

### Changes Required

**Step 1**: Replace class (line 2):
```latex
\documentclass[10pt,twocolumn,twoside]{IEEEtran}
```

**Step 2**: Remove conference package:
```latex
% Remove: \usepackage[accepted]{aistats2024}
```

**Step 3**: Add IEEE-specific commands:
```latex
\usepackage{cite}  % For IEEE-style citations

% Before \begin{document}:
\IEEEpubid{0000--0000/00\$00.00~\copyright~2024~IEEE}
```

**Step 4**: Change title/author format:
```latex
\title{Stream Function Neural Operators with Probabilistic Inference}

\author{
  \IEEEauthorblockN{Adetayo Okunoye}
  \IEEEauthorblockA{
    Institution Name\\
    City, Country\\
    Email: \texttt{email@institution.edu}
  }
}
```

## Creating Multiple Versions

### Recommended Structure

```bash
# In your project
pcpo/analysis/latex/
├── main.tex                    # Master version
├── main_aistat.pdf             # AISTAT ready
├── main_neurips.tex            # NeurIPS version
├── main_neurips.pdf
├── main_journal_siam.tex       # Journal version
├── main_journal_siam.pdf
├── README.md
├── QUICK_START.md
└── FORMAT_GUIDE.md             # This file
```

### Makefile for Multiple Versions

Create `Makefile` in latex/ directory:

```makefile
.PHONY: all aistat neurips icml siam arxiv clean

all: aistat neurips icml siam

aistat:
	cp main.tex main_aistat.tex
	pdflatex -interaction=nonstopmode main_aistat.tex
	bibtex main_aistat
	pdflatex -interaction=nonstopmode main_aistat.tex
	pdflatex -interaction=nonstopmode main_aistat.tex

neurips:
	cp main.tex main_neurips.tex
	# Add modifications for NeurIPS
	sed -i 's/\[accepted\]{aistats2024}/neurips_2024/g' main_neurips.tex
	pdflatex -interaction=nonstopmode main_neurips.tex
	bibtex main_neurips
	pdflatex -interaction=nonstopmode main_neurips.tex
	pdflatex -interaction=nonstopmode main_neurips.tex

icml:
	cp main.tex main_icml.tex
	sed -i 's/\[accepted\]{aistats2024}/icml2024/g' main_icml.tex
	pdflatex -interaction=nonstopmode main_icml.tex
	bibtex main_icml
	pdflatex -interaction=nonstopmode main_icml.tex
	pdflatex -interaction=nonstopmode main_icml.tex

siam:
	cp main.tex main_siam.tex
	# Add SIAM modifications
	pdflatex -interaction=nonstopmode main_siam.tex
	bibtex main_siam
	pdflatex -interaction=nonstopmode main_siam.tex
	pdflatex -interaction=nonstopmode main_siam.tex

arxiv:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex
	tar -czf paper_arxiv.tar.gz main.tex references.bib

clean:
	rm -f *.pdf *.aux *.bbl *.blg *.out *.log *_main* main_*.tex
```

Then compile any version with:
```bash
make aistat    # or make neurips, make icml, make siam
make clean     # remove all compiled files
```

## Page Count Limits

| Venue | Limit | Flexibility | 
|-------|-------|------------|
| AISTAT | 8 + refs | Strict |
| NeurIPS | 8 + refs | Strict |
| ICML | 8 + refs | Strict |
| arXiv | Unlimited | N/A |
| SIAM Review | 30-35 | Flexible |
| IEEE TPAMI | 15-20 | Flexible |

**Current paper**: ~8,000 words (~16 pages including appendices)

**To reduce for conferences** (to 8 pages):
1. Move appendices to supplementary material
2. Reduce related work to 1.5 pages
3. Condense methods section
4. Combine some results tables

**To expand for journals**:
1. Add detailed related work (3-4 pages)
2. Expand experimental section
3. Add more ablations
4. Include extended proofs

## Citation Style Guide

Current: `\bibliographystyle{apalike}` (APA)

### To change citation style:

**NeurIPS style** (numbered):
```latex
\bibliographystyle{plainnat}
% Then citations appear as [1], [2], etc.
```

**IEEE style**:
```latex
\bibliographystyle{ieeetr}
```

**SIAM style**:
```latex
\bibliographystyle{siam}
```

## Anonymous Review Mode

To create anonymous version for review:

1. **Remove author information**:
```latex
\author{Anonymous submission}
```

2. **Remove citations to your own work** or change to:
```latex
% Instead of: ...as shown in \cite{okunoye2024}...
% Use: ...as shown in concurrent work...
```

3. **Remove institution names** from references

4. **Remove identifying code/URLs**:
```latex
% Don't mention: "Our code at github.com/yourname/..."
% Instead: "Code available upon request"
```

## Useful Commands by Venue

### For Conference Versions
```latex
% Add to preamble for space saving:
\usepackage[compact]{titlesec}  % Reduce spacing around sections
\setlength{\parskip}{0pt}        % No space between paragraphs
\setlength{\parindent}{12pt}     % Standard indent
```

### For Journal Versions
```latex
% Add for journal formatting:
\onehalfspacing           % 1.5 line spacing
\usepackage{fancyhdr}     % Fancy headers/footers
\pagestyle{fancy}
\lhead{Stream Function Neural Operators}
\rhead{\thepage}
```

## Final Submission Checklist

### For All Venues
- [ ] Compile with no errors
- [ ] Check PDF looks correct
- [ ] All figures visible and labeled
- [ ] All citations resolve
- [ ] Spelling/grammar check
- [ ] Page count within limits

### For Conferences (AISTAT/NeurIPS/ICML)
- [ ] Anonymous (no author names)
- [ ] No identifying URLs
- [ ] Supplementary material separate
- [ ] Title describes novelty
- [ ] Abstract ≤ 200 words

### For Journals
- [ ] Author names and affiliations
- [ ] Keywords section
- [ ] Abstract ≤ 250 words
- [ ] Longer related work section
- [ ] More comprehensive experiments

### For arXiv
- [ ] Source files included
- [ ] All figures embedded
- [ ] References compile
- [ ] Keywords in metadata
- [ ] Category selected

## Quick Fixes

### Paper too long?
1. Remove an ablation from appendix
2. Shorten literature review
3. Combine tables
4. Use shorter captions

### Paper too short?
1. Expand methods with more detail
2. Add related work
3. Include all ablations
4. Add discussion of limitations

### Figures look pixelated?
1. Use vector formats (.pdf instead of .png)
2. Compile with: `pdflatex -dpi=300 main.tex`
3. Check figure source resolution

## Support Resources

- **AISTAT**: https://www.aistats.org/
- **NeurIPS**: https://nips.cc/
- **ICML**: https://icml.cc/
- **SIAM**: https://www.siam.org/
- **IEEE**: https://www.ieee.org/
- **arXiv**: https://arxiv.org/
- **LaTeX Help**: https://www.overleaf.com/learn

---

**Next Steps**:
1. Choose your target venue
2. Follow the modifications above
3. Compile test version
4. Review for formatting issues
5. Submit when ready!

Good luck with your submission! 🎓

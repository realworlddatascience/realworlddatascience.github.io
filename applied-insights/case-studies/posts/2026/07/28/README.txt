RWDS article — drop-in package
==============================

FILES
  report.qmd        Main Quarto source (local image references)
  references.bib    24 references, keyed @ref1..@ref24
  images/           10 PNGs (header + Figures 1-4)

TO RENDER
  1. Copy report.qmd, references.bib and images/ into your cloned copy of the
     RWDS template repo (the one that already contains rwds.scss, rwds.css and
     chicago.csl).
  2. From that folder:  quarto render report.qmd
     (or: quarto preview report.qmd)

CITATION STYLE — IMPORTANT
  The YAML uses  csl: chicago.csl  (ships with the template = Chicago author-date).
  That renders your citations as (Author Year) and alphabetises the reference list.
  Your article was written in NUMBERED Vancouver style. To keep [1]..[24]:
    - download vancouver.csl from https://github.com/citation-style-language/styles
    - place it next to report.qmd
    - in report.qmd swap:   csl: chicago.csl   ->   csl: vancouver.csl

#!/bin/bash
# Rebuild clean manuscript, tracked-changes version, response letter and source bundle.
set -e
for i in 1 2 3; do pdflatex -interaction=nonstopmode paper_revised.tex > build.out 2>&1 || true; done
for i in 1 2; do pdflatex -interaction=nonstopmode response_to_reviewers.tex > /dev/null 2>&1 || true; done
latexdiff -t CFONT --flatten --exclude-textcmd="section,subsection,paragraph" --config "PICTUREENV=(?:picture|DIFnomarkup|table|figure)[\w\d*@]*" paper_submitted.tex paper_revised.tex > paper_tracked_changes.tex 2>/dev/null
sed -i 's/\\ref{fig:wass}/4/g; s/\\ref{tab:crossmodal}/7/g; s/\\ref{tab:wesad_pen}/6/g; s/\\citep{wesaduci}/(Schmidt and Reiss, 2018)/g' paper_tracked_changes.tex
python3 - <<'P'
p='paper_tracked_changes.tex'; s=open(p).read(); o="\\end{frontmatter}"
s=s.replace(o,o+"\n\n\\noindent\\fbox{\\parbox{0.97\\textwidth}{\\small\\textbf{Tracked changes.} {\\color{blue}Blue indicates additions}; {\\color{red}\\footnotesize red indicates deletions}. Tables and figures are shown in their revised form only.}}\n\\medskip\n",1)
open(p,'w').write(s)
P
for i in 1 2; do pdflatex -interaction=nonstopmode paper_tracked_changes.tex > build_td.out 2>&1 || true; done
rm -f SMHL_revision_source.zip
zip -qr SMHL_revision_source.zip paper_revised.tex paper_tracked_changes.tex response_to_reviewers.tex make_figures.py joint_horizon_boot.py build_all.sh figures res/results/rev/*.csv juno
grep -cE "^!|undefined" paper_revised.log paper_tracked_changes.log response_to_reviewers.log || true
grep -cE "Overfull \\\\hbox \(([3-9][0-9]|[0-9]{3})" paper_revised.log || true
for f in paper_revised paper_tracked_changes response_to_reviewers; do pdfinfo $f.pdf | grep Pages; done

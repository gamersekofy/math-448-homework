# HW3 (Chapter 4: Classification) — Work Report

## Instructions received

Implement MATH 448 homework 3 (`hw3-ch4/questions.md`, problems 3, 6, 13, 15) in `D:\projects\SFSU\fa26\math448\math-448-homework\hw3-ch4\`, differing from the two previous notebook-based homeworks:

- Write Python script(s) (not a Jupyter notebook).
- Use `typst` (in PATH) to compile a clean PDF that answers all questions, including plots/images and code snippets.
- Install the ISLP package (from the ISLP_labs repo). Ignore Nix files. ARM64 Windows, `pwsh` shell.
- Past homeworks (`hw1-ch2`, `hw2-ch3`) are the style reference (title block: author "Uzair Hamed Mohammed", date; question text + written answers → `solutions.pdf`).

User's choices via questions: install ISLP into **system Python**; PDF date **10/16/2026**; code in PDF = **inline snippets + full `solutions.py` appendix**.

## Accomplished

1. **Environment** — `pip install ISLP` succeeded (ISLP 0.4.1; it downgraded pandas 3.0.1 → 2.3.3 and built statsmodels 0.15.0). Cleaned the corrupt leftover `~andas` site-packages dir from the interrupted first install. Verified: `ISLP.load_data("Weekly")` loads (1089 × 9).

2. **`solutions.py`** — complete, runs end-to-end (~40 s), writes `results.json` + 7 figures in `figs/`:
   - **Q3**: proof-support figure `q3_qda_boundary.png` (equal vs. unequal-variance discriminants; boundaries at exact roots −4.918 / −0.082, verified to machine precision — a normalization-term bug in the first version was found and fixed).
   - **Q6**: (a) p = 0.3775; (b) h = 50 (both numerically verified).
   - **Q13**: (a) describe table, correlations, boxplots, volume plot, corr heatmap; (b) full logistic regression (only **Lag2** significant, p = 0.0296); (c) training confusion matrix [[54,430],[48,557]], 56.11% correct; (d)–(h) test-split 1990–2008 → 2009–2010 on `Lag2`: Logit/LDA 62.5%, QDA 58.65% (predicts all Up), KNN(K=1) 49.04%, NB 58.65%; (j) systematic 315-fit experiment (15 predictor sets incl. squares/interactions/log-Volume × 4 methods + 17 K values for KNN). Best: **QDA on Lag1+Lag2+Lag1²+Lag2² → 64.42%** ([[20,23],[14,47]]), plus a K-sweep curve (best K=47 on Lag2: 61.5%).
   - **Q15**: `Power`, `Power2`, `Power3`, `PlotPower` implemented; outputs 8, 6561, 1000 / 8¹⁷ / 131³; figures `q15_power3.png` (linear + log-log) and `q15_plotpower.png`.

3. **`solutions.typ`** — written (all problems + tables pulled from `results.json`, figures embedded, confusion-matrix helper, title block with date 10/16/2026, outline, questions as callout blocks, code snippets for Q15). Compile errors encountered were fixed one at a time (`set.text` → `set text`, `calc.power` → `calc.pow`, `pct(x, 1)` → `d: 1`, `$bold(K)^*$` → `$bold(K)$`).

## Where I got stuck / Pending

1. **`typst compile solutions.typ` still failing at line 221 (part (c) prose)** — the last reported error was `error: unexpected argument` pointing at `#pct(cmc.at("down_correct_rate"), 1))`. I had just fixed that pair of calls to use `d: 1` but the recompile was interrupted, so the PDF **does not yet exist / is not verified**. Possible remaining issues in that same region: an `#pct(...)` with a numeric 1 (or a similar call earlier on that line — line 221 is one long paragraph) that the grep-based fix missed. Long single-paragraph lines make error pinpointing harder — reformat questionably long lines before debugging.
2. Not yet verified visually: the compiled PDF layout (title, outline, tables, figure sizing), Typst float→string edge cases in `fmt` (e.g. `Year` column, `-0` artifacts), and the `str(r.at("K"))` assuming the key is `K` (verify its actual spelling in `results.json`).
3. **Pending follow-ups after first successful compile:**
   - `hw3-ch4/README.md` (chapter + problem list, matching other folders).
   - Check that the appendix (`#raw(read("solutions.py")…)`, 7.4 pt) doesn't overflow monospace lines.
   - Cross-check every hard-coded prose number against `results.json` once more.
   - Optional polish: figure caption wording, page-break balance.
4. Note: `solutions.py` contains a stray unused comment `# ruff: noqa` at top-line convention from the repo—harmless. `results.json` includes key `cm.at("frac")` paths used in `.typ`; keys were created consistently.

**Next step:** re-run `typst compile solutions.typ solutions.pdf` in `hw3-ch4/` with a longer timeout, fix line 221 region, then do a visual pass of the PDF, add README.md, and recompile.

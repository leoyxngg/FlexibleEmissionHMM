---
name: ipynb_plot_style
description: Style and review plots created in Jupyter notebooks, including heatmaps, financial return charts, axis units, captions, and preview exports. Use when creating or updating .ipynb plots or when the user requests a notebook plot preview.
---

# Notebook plot style

## Labels and units

- Use the units the user requested. For plots of daily returns or differences of mean daily returns, display values as percentages (`%`) rather than abbreviating them as `pp` when the user requests percent labels. A difference of 0.028% in two mean returns is an **absolute difference of percentage-valued returns**, not a relative improvement of 0.028%; say this clearly in the caption when needed.
- Keep the calculation unchanged when changing only its display unit. Convert decimal returns to displayed percent with `100 * value` exactly once, and check the colorbar, annotations, title, and explanation agree.
- Give each column or axis a concise comparison label. Say which period and data series the returns represent, and whether a plotted position applies to the following day's return.

## Heatmaps and previews

- Put the signed return difference directly in each heatmap cell, with `+`/`−` signs and the requested unit. Do not place sample counts (`n=...`) inside cells unless requested. If sample sizes are needed for rigor, report them separately from the preview or in accompanying analysis.
- Use a diverging color map centered on zero with symmetric limits for signed differences. Verify readable text contrast and avoid clipped titles, axes, ticks, colorbars, and notes by inspecting the rendered image.
- If the user asks to preview first, export a clearly named preview under the experiment's `artifacts/` directory and show it in chat; do not add it to the notebook or treat the design as approved until the user says so.
- Do not change source checkpoints, refit models, or replace saved experimental signals while producing a plot from saved results.

## Notebook conventions

- In `.ipynb` Markdown use `$...$` for inline math and `$$...$$` for display math, not `\\(...\\)` or `\\[...\\]`.
- Use notebook-relative or project-relative output paths instead of hard-coded machine-specific paths.
- Keep conclusions descriptive: a heatmap of conditional means does not establish statistical significance, causal profit, or a fair held-out model-selection result.

# Notebook Markdown

- In `.ipynb` Markdown cells, delimit inline LaTeX with `$...$` and display LaTeX with `$$...$$` on separate lines.
- Do not use `\\(...\\)` or `\\[...\\]` as math delimiters in notebook Markdown.

# Python Style

- Follow PEP 8 - Style Guide for Python Code (`peps.python.org/pep-0008/`) in all Python files: 4-space indentation, snake_case for functions/variables, UPPER_CASE for module constants, imports grouped stdlib / third-party / local, one blank line between top-level definitions, line length up to 88 characters where practical, and docstrings (`"""..."""`) on public functions.
- Follow PEP 257 - Docstring Conventions (`peps.python.org/pep-0257/`): imperative-mood one-line summary ending with a period, a blank line after it when more follows, and the closing `"""` on its own line for multi-line docstrings. Keep every docstring human readable and under 100 words.

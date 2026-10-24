# Misc math notebooks

## The Bizarre Geometry of High-Dimensional Space

A self-contained notebook exploring concentration of measure: where a ball's
volume lives, spiky hypercubes, near-orthogonal random vectors, and the
Gaussian shell. Only needs `numpy` and `matplotlib`.

### Run it

```bash
jupyter lab "The Bizarre Geometry of High-Dimensional Space.ipynb"
```

Then run the cells top to bottom. (Or just upload the `.ipynb` to Google Colab,
where numpy and matplotlib are preinstalled.)

`jupyter lab` opens the notebook in a browser. Don't confuse it with
`jupyter nbconvert --execute` below, which is headless — it re-runs the notebook
and saves the outputs but never opens a window.

### Local setup (uv)

Jupyter here is managed by [uv](https://docs.astral.sh/uv/), not Homebrew or the
system Python — it lives in its own isolated environment so it can't break when
Homebrew bumps its Python. `uv`/`uvx` are on `PATH` via `~/.local/bin`, and the
`jupyter` command is on `PATH` too.

```bash
# one-time install (if setting up a new machine)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install jupyter-core --with jupyterlab --with nbconvert \
  --with ipykernel --with numpy --with matplotlib

# upgrade later
uv tool upgrade jupyter-core

# re-render the notebook in place (embeds the plot outputs)
jupyter nbconvert --to notebook --execute --inplace \
  "The Bizarre Geometry of High-Dimensional Space.ipynb"
```

# Build makemore: MLP

A character-level language model built with PyTorch. This educational project follows the MLP portion of Andrej Karpathy's [makemore series](https://github.com/karpathy/makemore): it learns patterns from a names dataset and samples new, name-like strings.

> **Project status:** Learning project / notebook implementation. It is not packaged as a production-ready library or command-line application.

## What you'll learn

- Convert characters into integer tokens and build context/next-character examples.
- Train a multi-layer perceptron (MLP) with learned character embeddings.
- Track cross-entropy loss and evaluate on development and test splits.
- Generate samples from a trained model and visualize character embeddings.

The notebook uses a context of five characters, 20-dimensional embeddings, and a 600-unit `tanh` hidden layer. It trains on a random 80% of the names and reserves 10% each for development and testing.

## Quick start

### Option 1: Google Colab

1. Open [`build_makemore_mlp.ipynb`](build_makemore_mlp.ipynb) in Google Colab.
2. Run the cells from top to bottom. The notebook downloads `names.txt` from the makemore repository.

### Option 2: Run locally

You need Python 3 and Jupyter. A GPU is optional; training can take longer on a CPU.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook build_makemore_mlp.ipynb
```

Run the notebook cells in order. The training loop runs for 500,000 iterations; lower this value in the notebook for a quicker trial.

## Repository contents

```text
.
├── build_makemore_mlp.ipynb  # Data preparation, training, evaluation, and sampling
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignores local environments and generated files
└── CONTRIBUTING.md           # Notes for people improving the project
```

## Dataset and attribution

The notebook downloads [`names.txt`](https://github.com/karpathy/makemore) from the makemore project at runtime. Please refer to the upstream project for dataset provenance and applicable terms. The dataset is not stored in this repository.

## Possible next steps

- Refactor reusable model and training code into Python modules.
- Add command-line training and name-generation entry points.
- Save and load model checkpoints.
- Add automated tests and a reproducible environment/CI workflow.

## License

No license has been added yet. Until a license is chosen, do not assume others have permission to reuse, modify, or distribute this project's code. See GitHub's guide to [licensing a repository](https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) before selecting one.

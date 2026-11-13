# Build makemore: MLP

A small character-level language model built in PyTorch. This project follows the MLP portion of Andrej Karpathy's **makemore** series: it learns patterns in a names dataset and uses them to generate new, name-like strings.

## What's inside

The project is currently a Jupyter notebook: [`build_makemore_mlp.ipynb`](build_makemore_mlp.ipynb). It walks through:

- Loading the names dataset and mapping characters to integer IDs.
- Building training examples from a rolling context of five characters.
- Splitting the examples into training, development, and test sets (80/10/10).
- Training a character-level MLP with learned character embeddings.
- Measuring cross-entropy loss, sampling generated names, and visualizing the learned embeddings.

The model embeds each of the five context characters into a 20-dimensional vector, concatenates those embeddings, and passes them through a 600-unit `tanh` hidden layer to predict the next character. The vocabulary contains the 26 letters plus a special end-of-name token.

## Run the notebook

Open `build_makemore_mlp.ipynb` in Google Colab or a local Jupyter environment. The notebook downloads `names.txt` from the [makemore repository](https://github.com/karpathy/makemore) when it runs.

For a local environment, install the notebook's Python dependencies:

```bash
python -m pip install torch matplotlib jupyter
```

Then start Jupyter and open the notebook:

```bash
jupyter notebook build_makemore_mlp.ipynb
```

A GPU is optional; the notebook can run on a CPU, though training may take longer. The training loop runs for 500,000 iterations, so reduce that count if you want a quicker experiment.

## Data and attribution

The notebook uses `names.txt`, downloaded at runtime from [karpathy/makemore](https://github.com/karpathy/makemore). See that project for information about the dataset and its use.

## Learning goals

This is an educational implementation intended to make the steps of building and training a simple language model easy to inspect and experiment with. Try changing the context length, embedding size, hidden-layer size, or training settings and compare the results.

Run script with `train_nn_on_drifts.py`, see `quick_test.sh` to see how to generate data and train on it.

The premise of this is the following : we are interested in the surrogate models for simulating the following equation:

$$dx_t = ( b * \mu_t)(x_t) dt + dW_t, \mu_t = \text{Law}(X_t)$$

where $*$ means convolution and $W_t$ is a noise. It is known that this equation can be approximated with a an interacting particle system, given with

$$N \in \mathbb{N}\;\;\; dx^i = \frac{1}{N} \sum_{j=1}^N b(x^i_t - x^j_t) dt + dW_t$$

In this project we aim at training a neural network that will approximate the map

$$(t, x) \mapsto (b * \mu_t)(x_t)$$

by being trained to predict the map

$$(t, x, b, \mu_0) \mapsto \frac{1}{N} \sum_{j=1}^N b(x-x^j_t) dt$$

where $\mu_0$ is an initial distribution and $b$ is either parameterised or expressed in some basis.

______

The core of the project is `train_nn_on_drifts.py`, one can run `tests.py` or `quick_test.sh`, which are typically run by me on every commit (CI shell script is not here)

You can visualise experiments in `visualise_trained_model.ipynb`.

C++ scripts for data generation are in `cpp_simul`.

(this description is intentionally left vague)
# epidemics-suppression

This repository contains a algorithm written in Python that implements a mathematical model, introduced in the paper <paper_ref>, describing the suppression of an epidemics due to measures that identify and isolate infected individuals. The output of the algorithm is the time evolution of the relative suppression of the effective reproduction number due to these measures.

In section 4 of the paper we described the application of the model to the COVID-19 epidemic, and we reported the results of some computations that were made running the code contained in this repository.

We refer to the paper for details about the model, the interpretation of the parameters appearing here, and the sources of the epidemiological features of COVID-19 that are used as an input of the model. Note that in this repository, like in the computations made in the paper, the "default" (i.e. without measures) effective reproduction number is taken constantly equal to 1, as we are interested in studying the relative suppression only.


## Set-up and usage

To use the algorithm you should clone the repository by running
```sh
git clone https://github.com/MarcoMene/epidemics-suppression.git
cd epidemics-suppression
```
The dependencies of the library are described in the Pipfile. To install them, you first need to install Pipenv:
```sh
pip install pipenv
```
Then you run
```sh
pipenv update
```

## Examples

The folder `examples` contains some functions that, when executed, run the algorithm with certain specific choice of the input parameters, generating the plots appearing in the paper. There are also some scripts running only some small pieces of the algorithm to illustrate how they work.

1. `examples/R_suppression_examples.py` contains some examples of computations of the suppressed effective reproduction number, given the distribution of the time of positive testing for an infected individual.

2. `examples/epidemic_data_examples.py` contains some examples plotting the distributions of epidemic data used as inputs of the model.

3. `examples/time_evolution_examples.py` contains some examples running the complete algorithm in certain scenarios, and plotting the results.

4. `examples/reproduction_number_time_evolution_span_parameters.py` contains a function that runs the algorithm several time, each with a different choice of the input parameters.

## How to cite

TBD

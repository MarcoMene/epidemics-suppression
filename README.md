# epidemics-suppression

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


This repository implements a mathematical model, introduced in the paper <paper_ref>, 
describing the suppression of an epidemic due to measures that identify and
isolate infected individuals, possibly with the help of a mobile app.
The output of the algorithm is the time evolution
of the relative suppression of the effective reproduction number due to these measures.

In section 4 of the paper we described the application of the model to the COVID-19 
epidemic, and we reported the results of some computations that were made running the 
code contained in the package `examples` of this repository.

We refer to the paper for details about the model, the interpretation of the parameters
appearing here, and the sources of the epidemiological features of COVID-19 that 
are used as an input of the model. Note that in this repository, like in the 
computations made in the paper, the "default" (i.e. without measures) effective 
reproduction number is taken constantly equal to 1, as we are interested in studying 
the relative suppression only.


## Set-up and usage

To use the algorithm you should clone the repository by running
```sh
git clone https://github.com/MarcoMene/epidemics-suppression.git
cd epidemics-suppression
```
The library runs on Python 3.8. The dependencies of the library are provided in the
file `pyproject.toml`. To install them, you first need to
install [Poetry](https://python-poetry.org/docs/), and then run
```sh
poetry install
```



## Examples

The directory `examples` contains some functions that, when executed, run the algorithm
with certain specific choice of the input parameters, printing the results
and generating the plots appearing in the paper.
There are also some scripts running only some small pieces of the algorithm to
illustrate how they work, or displaying the epidemic data used by the algorithm.

## How to cite

TBD

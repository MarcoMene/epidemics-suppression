# epidemics-suppression

This repository supports this paper: <paper_ref>.
It provides a python implementation of a mathematical model that conveniently parametrizes the effectiveness of isolation measures in containing the 
COVID-19 epidemics, giving as output the time-evolution of the suppression of the effective reproduction number.
The study takes some epidemiological characteristics of COVID-19 from scientific literature.

Refer to the paper for the details of the model, the interpretation of parameters etc.


## Set-up the project on your machine

```sh
git clone https://github.com/MarcoMene/epidemics-suppression.git
cd epidemics-suppression
pipenv install
```

# Main scripts

Given a set of initial parameters, **computes the time evolution of the suppression** induced over R by isolation measures.

```
scripts/reproduction_number_time_evolution.py
```

## how to cite

<paper_ref>
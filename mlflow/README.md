## 1. About

This repository contains MLFlow integration toolchain for [ODAHU platform](https://github.com/odahu/odahu-flow).

## 2. What does it do?

Odahu-flow's toolchain for MLFlow is responsible for running process of training MLFlow models. It produces model as ZIP archive in Odahu-flow's General Python Prediction Interface.

## 3. Requirements

This toolchain requires MLFlow tracking server to be installed (MLFlow tracking server is bundled in HELM Chart).

## 4. Implementation details

The toolchain Docker file must have two conda environments:
* `base` contains this MLFlow toolchain package and its dependencies.
* `odahu_model` contains model dependencies. It is also available as value of ODAHU_CONDA_ENV_NAME environment variable.


This toolchain provides two entrypoints:
* `odahu-flow-mlflow-runner` operates inside the `base` conda environment.
It prepares MLFlow training process and launch `odahu-flow-mlflow-wrapper`.
* `odahu-flow-mlflow-wrapper` launchs MLFlow training inside `odahu_model` conda environment.

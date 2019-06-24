# 1. About

This repository contains MLFlow integration toolchain for [Legion platform](https://github.com/legion-platform/legion).

## 1.1. What does it do?

Legion's toolchain for MLFlow is responsible for running process of training MLFlow models. It produces Legion wrapper (`entrypoint.py`) module according `Legion's Python Integration Protocol`, that is required for using this built models (with wrapper) for packing models to target systems (such as SageMaker, Docker, Spark and etc.). For dependencies declaration this toolchain supports two types: `conda` and `docker`. Type `conda` could be reused in multiple packaging cases, `docker` could be used only for wrapping model to Docker Image.

## 1.2. Legion's Python Integration Protocol

This protocol requires Python's module named `entrypoint` (e.g. file `entrypoint.py`) in models directory. This file should be importable Python module, that contains next function:

* `init` - is being invoked during bootup, can be used for loading/parsing configuration and etc.
* `predict` - (IS NOT USED IF `predict_matrix` is defined) is being invoked during invocating prediction method. It supports making one prediction (e.g. prediction for one image), based on input data vector and it outputs JSON serializable object, e.g. dict, int, list, tuple and etc. (WARNING: pandas types should be converted to native types to be JSON-serializable).
* `predict_matrix` - is being invoked for prediction on prepared matrix.
* `info` - OPTIONAL method. Might be used for getting detail information about model inputs and outputs.
* `columns_order` - OPTIONAL method. Is used for ordering values for `predict_vector`. Otherwise values is being passed as dict unordered.
* `get_json_encoder` - OPTIONAL method. Might be used for providing custom JSON serializable object.

### 1.2.1 Init function
Arguments: ---

Returns: None

### 1.2.2 Predict function
Arguments: dict if info function is not provided, instance of `pydantic.BaseModel` otherwise.

Returns: JSON serializable object, e.g. dict / int / float / etc.

### 1.2.3 Info function (optional)
Arguments: ---

Returns: type (inherited from `pydantic.BaseModel`) or OpenAPI JSON format as dict.

### 1.2.4 Providing custom JSON encoder (optional)
Arguments: ---

Returns: type (to be used as `cls` for `json.dump`)

# 2. Requirements

This toolchain requires MLFlow tracking server to be installed.
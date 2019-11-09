SHELL := /bin/bash
TAG=latest

.EXPORT_ALL_VARIABLES:

## docker-build-mlflow-toolchain: Build MLFlow toolchain docker image
docker-build-mlflow-toolchain:
	docker build -t odahuflow/mlflow-toolchain:${TAG} -f containers/mlflow-toolchain/Dockerfile .

## docker-build-mlflow-toolchain-gpu: Build MLFlow gpu toolchain docker image
docker-build-mlflow-toolchain-gpu:
	docker build -t odahuflow/mlflow-toolchain-gpu:${TAG} -f containers/mlflow-toolchain-gpu/Dockerfile .

## docker-build-mlflow-tracking-server: Build MLFlow docker image
docker-build-mlflow-tracking-server:
	docker build -t odahuflow/mlflow-server:${TAG} -f containers/mlflow-tracking-server/Dockerfile .

## docker-build-pipine-agent: Build pipeline agent docker image
docker-build-pipine-agent:
	docker build -t odahuflow/pipine-agent:${TAG} -f containers/mlflow-pipine-agent/Dockerfile .

## docker-build-resource-applier: Build resource applier docker image
docker-build-resource-applier:
	docker build -t odahuflow/odahuflow-trainer-resource-applier:${TAG} -f containers/resource-applier/Dockerfile .

## install-mflow-runner: Start MLFLow server in Docker
install-mflow-runner:
	cd mlflow && \
		rm -rf build dist *.egg-info && \
		pip3 install ${BUILD_PARAMS} -e . && \
		python setup.py sdist && \
		python setup.py bdist_wheel

## install-mflow-runner-tests: Start MLFLow server in Docker
install-mflow-runner-tests:
	cd mlflow && pip install -e ".[testing]"

## lint-mflow-runner: Start linting of mlflow runner
lint-mflow-runner:
	pylint mlflow/odahuflow
	pylint mlflow/tests

## test-mflow-runner: Start unit tests of mlflow runner
test-mflow-runner:
	cd mlflow && pytest tests --disable-warnings

## run-mflow-server: Start MLFLow server in Docker
run-mflow-server:
	docker run -ti --rm \
	  -p 5555:5000 \
	  -v `pwd`/server-mlruns:/mlruns \
	  odahuflow/k8s-mlflow-server

## run-mflow-sandbox: Start MLFLow sandbox
run-mflow-sandbox:
	docker run -ti --rm \
	  -e MLFLOW_TRACKING_URI=http://`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`:5000 \
	  -v `pwd`:/ex \
	  -v `pwd`:/legion-cmd \
	  -w /ex \
	  odahuflow/mlflow-toolchain:${TAG} bash

## helm-delete-mflow: Delete mlflow helm release
helm-delete-mflow:
	helm delete --purge odahuflow-mlflow || true

## helm-install-mflow: Install mlflow helm chart
helm-install-mflow: helm-delete-mflow
	helm install helms/odahuflow-mlflow --name odahuflow-mlflow --namespace odahuflow --debug -f helms/values.yaml --atomic --wait --timeout 120

SHELL := /bin/bash
TAG=latest

PIP_EXTRA_INDEX_URL=

-include .env
.EXPORT_ALL_VARIABLES:

## docker-build-mlflow-training-integration: Build MLFlow training integration docker image
docker-build-mlflow-training-integration:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t odahu/odahu-flow-mlflow-training-integration:${TAG} -f containers/mlflow-training-integration/Dockerfile .

## docker-build-mlflow-tracking-server: Build MLFlow docker image
docker-build-mlflow-tracking-server:
	docker build -t odahu/mlflow-server:${TAG} -f containers/mlflow-tracking-server/Dockerfile .

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
	  odahu/k8s-mlflow-server

## run-mflow-sandbox: Start MLFLow sandbox
run-mflow-sandbox:
	docker run -ti --rm \
	  -e MLFLOW_TRACKING_URI=http://`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`:5000 \
	  -v `pwd`:/ex \
	  -v `pwd`:/legion-cmd \
	  -w /ex \
	  odahu/mlflow-training-integration:${TAG} bash

## helm-delete-mflow: Delete mlflow helm release
helm-delete-mflow:
	helm delete --purge odahu-flow-mlflow || true

## helm-install-mflow: Install mlflow helm chart
helm-install-mflow: helm-delete-mflow
	helm install helms/odahu-flow-mlflow --name odahu-flow-mlflow --namespace odahu-flow --debug -f helms/values.yaml --atomic --wait --timeout 120

SHELL := /bin/bash
TAG=latest

-include .env

.EXPORT_ALL_VARIABLES:

.PHONY: help


help: Makefile
	@echo "Choose a command run in "$(PROJECTNAME)":"
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'
	@echo

## docker-build-mlflow-toolchain: Build MLFlow toolchain docker image
docker-build-mlflow-toolchain:
	docker build -t legion/mlflow-toolchain:${TAG} -f containers/mlflow-toolchain/Dockerfile .

## docker-build-mlflow-toolchain-gpu: Build MLFlow gpu toolchain docker image
docker-build-mlflow-toolchain-gpu:
	docker build -t legion/mlflow-toolchain-gpu:${TAG} -f containers/mlflow-toolchain-gpu/Dockerfile .

## docker-build-mlflow-tracking-server: Build MLFlow docker image
docker-build-mlflow-tracking-server:
	docker build -t legion/k8s-mlflow-server:${TAG} -f containers/mlflow-tracking-server/Dockerfile .

## docker-build-pipine-agent: Build pipeline agent docker image
docker-build-pipine-agent:
	docker build -t legion/pipine-agent:${TAG} -f containers/mlflow-pipine-agent/Dockerfile .

## docker-build-resource-applier: Build resource applier docker image
docker-build-resource-applier:
	docker build -t legion/resource-applier:${TAG} -f containers/resource-applier/Dockerfile .

build-temp-model:
	cd final2 && \
	docker build -t legion-temp/final-model:${TAG} .

run-temp-model:
	docker run -ti --rm \
	  -p 5001:5000 \
	  -v `pwd`/runner.py:/legion-cmd/runner.py \
	  -v `pwd`/entrypoint.py:/legion-cmd/entrypoint.py \
	  legion-temp/final-model:${TAG}

## run-server: Start MLFLow server in Docker
run-server:
	docker run -ti --rm \
	  -p 5555:5000 \
	  -v `pwd`/server-mlruns:/mlruns \
	  legion/k8s-mlflow-server

## run-sandbox: Start MLFLow sandbox
run-sandbox:
	docker run -ti --rm \
	  -e MLFLOW_TRACKING_URI=http://`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`:5000 \
	  -v `pwd`:/ex \
	  -v `pwd`:/legion-cmd \
	  -w /ex \
	  legion/mlflow-toolchain:${TAG} bash

## helm-delete: Delete mlflow helm release
helm-delete:
	helm delete --purge legion-mlflow || true

## helm-install: Install mlflow helm chart
helm-install: helm-delete
	helm install helms/legion-mlflow --name legion-mlflow --namespace legion --debug -f helms/values.yaml --atomic --wait --timeout 120

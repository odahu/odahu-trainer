SHELL := /bin/bash

-include .env

.EXPORT_ALL_VARIABLES:

.PHONY: help


help: Makefile
	@echo "Choose a command run in "$(PROJECTNAME)":"
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'
	@echo

## build-docker-mlflow-toolchain: Build MLFlow toolchain docker image
build-docker-mlflow-toolchain:
	docker build -t legion/mlflow-toolchain:latest -f containers/mlflow-toolchain/Dockerfile .

## build-docker-mlflow-tracking-server: Build MLFlow docker image
build-docker-mlflow-tracking-server:
	docker build -t legion/k8s-mlflow-server:latest -f containers/mlflow-tracking-server/Dockerfile .

build-temp-model:
	cd final2 && \
	docker build -t legion-temp/final-model:latest .

run-temp-model:
	docker run -ti --rm \
	  -p 5001:5000 \
	  -v `pwd`/runner.py:/legion-cmd/runner.py \
	  -v `pwd`/entrypoint.py:/legion-cmd/entrypoint.py \
	  legion-temp/final-model:latest

## run-server: Start MLFLow server in Docker
run-server:
	docker run -ti --rm \
	  -p 5000:5000 \
	  -v `pwd`/server-mlruns:/mlruns \
	  legion/k8s-mlflow-server

## run-sandbox: Start MLFLow sandbox
run-sandbox:
	docker run -ti --rm \
	  -e MLFLOW_TRACKING_URI=http://`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'`:5000 \
	  -v `pwd`:/ex \
	  -v `pwd`:/legion-cmd \
	  -w /ex \
	  legion/mlflow-toolchain:latest bash

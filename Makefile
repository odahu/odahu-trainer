SHELL := /bin/bash

BUILD_TAG=latest
TAG=

-include .env
include mlflow/Makefile

.DEFAULT_GOAL := help

## install-vulnerabilities-checker: Install the vulnerabilities-checker
install-vulnerabilities-checker:
	./scripts/install-git-secrets-hook.sh install_binaries

## check-vulnerabilities: Сheck vulnerabilities in the source code
check-vulnerabilities:
	./scripts/install-git-secrets-hook.sh install_hooks
	git secrets --scan -r

## docker-build: Build docker image
docker-build:
	docker build -t odahu/odahu-flow-mlflow-toolchain:${BUILD_TAG} -f containers/mlflow-toolchain/Dockerfile .

## docker-push-api: Push docker image
docker-push:
	docker tag odahu/odahu-flow-mlflow-toolchain:${BUILD_TAG} ${DOCKER_REGISTRY}/odahu/odahu-flow-mlflow-toolchain:${TAG}
	docker push ${DOCKER_REGISTRY}/odahu/odahu-flow-mlflow-toolchain:${TAG}

## help: Show the help message
help: Makefile
	@echo "Choose a command run in "$(PROJECTNAME)":"
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sort | sed -e 's/\\$$//' | sed -e 's/##//'
	@echo

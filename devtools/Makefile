# Makefile for devtools. Will build a docker image named `sklearn-devel` and a
# script `devenv.sh`.

# You can set these variables from the command line.
DOCKER            ?= docker
DOCKER_IMAGE_NAME ?= sklearn-devel
PYTHON_BASE       ?= python:2.7.13-jessie
UID               ?= 1000

.PHONY: docker-build clean

all: clean docker-build # env-cmd

clean:
	rm -rf docker/

docker-build:
	cd docker/ && \
	docker build  \
		--build-arg BASE_IMAGE=$(PYTHON_BASE) \
		--build-arg UID=$(UID) \
		-t $(DOCKER_IMAGE_NAME) .

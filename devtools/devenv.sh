#!/usr/bin/env bash

docker run --rm -it -v `pwd`:/project sklearn-devel $@

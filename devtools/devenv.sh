#!/usr/bin/env bash

"${DOCKER:-docker}" run --rm -it -v "$(pwd)":/project \
                    ${EXTRA_ARGS:-} \
                    ${DOCKER_IMAGE:-sklearn-devel} $@

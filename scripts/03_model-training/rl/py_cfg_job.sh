#!/bin/sh
# Python stand-in for a launcher's config check: records the launcher's arguments in $CFG_ARGS, then lets hydra print
# the composed job config with the real interpreter and exit (hydra handles --cfg before main: no Ray, no GPU).
printf '%s\n' "$@" > "$CFG_ARGS"
CUDA_VISIBLE_DEVICES= exec $FP_VENVS/verl/bin/python -B "$@" --cfg job --resolve

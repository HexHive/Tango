#!/usr/bin/env bash

docker run \
    --env TANGO_NO_PROFILE=1 \
    --env TANGO_PROFILE_time_elapsed=1 \
    --env TANGO_PROFILE_time_crosstest=1 \
    --env TANGO_PROFILE_snapshots=1 \
    --env TANGO_PROFILE_states=1 \
    --env TANGO_PROFILE_inferred_snapshots=1 \
    --env TANGO_PROFILE_total_savings=1 \
    --env TANGO_PROFILE_total_misses=1 \
    --env TANGO_PROFILE_total_hits=1 \
    --env TANGO_PROFILE_eg_savings=1 \
    --env TANGO_PROFILE_eg_misses=1 \
    --env TANGO_PROFILE_eg_hits=1 \
    --env TANGO_PROFILE_dt_savings=1 \
    --env TANGO_PROFILE_dt_misses=1 \
    --env TANGO_PROFILE_dt_hits=1 \
    --env TANGO_PROFILE_dtex_savings=1 \
    --env TANGO_PROFILE_dtex_misses=1 \
    --env TANGO_PROFILE_dtex_hits=1 \
    --env TANGO_PROFILE_snapshot_cov=1 \
    --env TANGO_PROFILE_total_cov=1 \
    -v $PWD:/home/tango \
    -p 8080:8080 \
    --rm -it --privileged tango:latest "$@" \
    -o tracker.track_heat true \
    -o generator.log_model_history true \
    -o generator.log_time_step 300 \
    -o generator.log_flush_buffer 1024 \
    \
    -o strategy.type inference \
    -o strategy.dump_stats true \
    -o strategy.recursive_collapse true \
    -o strategy.disperse_heat true \
    -o strategy.broadcast_state_schedule true \
    -o generator.broadcast_mutation_feedback true \
    \
    -o strategy.extend_on_groups true \
    -o strategy.dt_predict true \
    -o strategy.dt_extrapolate true \
    \
    -o strategy.validate true \
    \
    -o strategy.inference_batch 50 \

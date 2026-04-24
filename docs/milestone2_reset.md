# Milestone 2 Reset: Causal Alignment For Accompaniment

## Problem Framing

The active Milestone 2 problem is not "improve OLTW" and not "tune the
Kalman filter." The problem is:

> estimate musical position causally from an incoming performance stream, with
> low enough latency and high enough accuracy to support accompaniment.

Offline DTW can repair local mistakes with future context. A real-time system
cannot. The first Milestone 2 question is therefore:

> what failure modes appear when alignment must be causal, and what minimal
> online method exposes those failures clearly?

## Active Reset Baselines

The reset path keeps the existing advanced `kalman_oltw` implementation in the
repo, but treats it as parked for later comparison rather than the active
development path.

The active baselines are:

- `offline_dtw`: offline reference for what good alignment can look like when
  future context is available
- `oltw` / `oltw_global`: existing external online baselines
- `naive_online_dtw`: full-width causal normalized online DTW, anchored at the
  reference origin, with no tracker
- `basic_kalman_online_dtw`: the same measurement stream with a plain
  constant-velocity Kalman filter on top

This isolates the core causal-alignment problem before revisiting advanced
search windows, recovery policies, measurement gating, or tracker variants.

## Benchmark Ladder

- `small`: fastest 6-pair smoke benchmark for quick pipeline checks
- `development`: balanced causal-alignment benchmark for iteration
- `paper_test`: held-out evaluation once the reset baselines are understood

## Development Benchmark Design

`development` mode is the default iteration benchmark for Milestone 2 reset
work.

Design:

- 3 pieces
- 4 recordings per piece
- all directed within-piece pairs
- expected size: 36 directed pairs when all three pieces are eligible

Selection policy:

- prefer `Chopin_Op017No4`, `Chopin_Op024No2`, and `Chopin_Op030No2`
- require four annotated recordings per piece
- choose a duration-balanced 4-recording window per piece
- respect the benchmark warp-factor cap when selecting that window
- fall back to other eligible pieces if a preferred piece is not available

This benchmark is meant to be small enough for repeated runs and large enough
to expose drift, lag, local ambiguity, and tempo-mismatch failures that the
6-pair `small` mode can miss.

## Evaluation Focus

The reset baselines should be compared using:

- per-pair beat-alignment metrics
- piece-level summaries
- early-vs-late error summaries
- pairwise error traces for manual failure analysis

The goal of the reset phase is to determine whether the dominant problem is:

- bias in the online DP measurements
- tracker lag or smoothing behavior
- both

Only after that diagnosis should the project return to the advanced
`kalman_oltw` search, coupling, and preset experiments.

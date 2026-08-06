---
layout: home

hero:
  name: qkd
  text: QKD with the hardware in the loop
  tagline: Excess noise is an output assembled from measured hardware parameters, not an input.
  image:
    src: /icon.svg
    alt: qkd
  actions:
    - theme: brand
      text: Getting Started
      link: /getting-started
    - theme: alt
      text: Usage by example
      link: /usage
    - theme: alt
      text: Protocol coverage
      link: /guide/protocols
    - theme: alt
      text: GitHub
      link: https://github.com/plutoniumm/qkd

features:
  - title: Excess noise is an output
    details: >
      The budget is assembled from laser linewidth, electronic noise, quantum
      efficiency, DAC and ADC bit depth, pilot power ratio, symbol rate and
      fibre length. The key rate follows from it.
  - title: DSP in the loop
    details: >
      Pilot-assisted phase recovery and carrier-frequency-offset estimation run
      over simulated symbols. Their residual phase error is an excess-noise
      term.
  - title: Discrete and continuous
    details: >
      Gaussian modulation and M-PSK; BB84-WCP with decoy states, six-state,
      SARG04, B92, DPS, COW; BBM92 and E91; both untrusted midpoints, CV-MDI and
      MDI-BB84, each on a Rust engine.
  - title: Every plane is named
    details: >
      Excess noise requires ref=. Detector trust is a separate flag, a
      statement about the adversary.
  - title: It refuses rather than approximates
    details: >
      A configuration with no published bound raises and names the restriction:
      a trusted detector on the analytic discrete-modulation bound, or a
      device-independent CHSH claim.
  - title: Native Rust core
    details: >
      Symplectic covariance, a truncated number basis, the symbol pipeline, the
      cone solvers and every key-rate formula, in f64 on CPU with rayon.
      Dispatch asks for a precision, not a device.
---

## What this is

A Python package over a native Rust core (`qkd._core`, PyO3, abi3-py311).
[Getting Started](/getting-started) installs it.

## What you can get a number out of

[The protocol matrix](/guide/protocols#matrix); [Usage by example](/usage) runs
each row.

## Three things to know before quoting a number

- Verification is graded on two axes: [Validation](/guide/validation#two-tiers).
- Most `q.Link` families are asymptotic: [Security](/guide/security#asymptotic).
- Every `q.Link` run is CPU: [the GPU path](/roadmap#gpu).

## Where next

[Architecture](/architecture); [Conventions](/guide/conventions).

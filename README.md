# Room Acoustics FEM Analysis — Project Overview

## What this project is

A thesis project using finite element (FEM) simulation to analyze the
acoustic behavior of a room — how sound pressure builds up, resonates,
and varies throughout the space in response to a noise source, and how
that response compares against the room's theoretical acoustic
properties.

## Motivation

Real rooms don't respond uniformly to sound at every frequency — they
have resonances (room modes) where the response is much stronger than
elsewhere, and nulls where it's much weaker, both determined by the
room's dimensions and the positions of the source and listener.

## What the code does

The main analysis pipeline (`room-fem-mass-source.py`):

1. **Builds and meshes a rectangular room** in ANSYS Mechanical APDL,
   using an acoustic finite element formulation.
2. **Applies a single point noise source**, modeled as an ideal point
   monopole (a mass-source excitation) — the direct finite element
   representation of a compact acoustic source.
3. **Runs a harmonic acoustic solve** across a chosen frequency range,
   computing the complex sound pressure field throughout the room at
   each frequency.
4. **Extracts the sound pressure level (SPL) response at a listener
   position**, interpolated at the listener's exact coordinates rather
   than approximated from the nearest mesh node.
5. **Compares the result against theoretical room modes** — the
   closed-form rigid-wall resonance frequencies for a room of these
   dimensions — rather than relying on empirical peak-detection, turning
   the analysis into a direct validation of the FEM model against known
   acoustic theory.
6. **Produces field maps** — 3D and plane cross-section visualizations
   of the pressure field at the room's strongest resonances — both as
   static images and as portable data files for interactive exploration.

## Infrastructure

The simulation runs on **ANSYS MAPDL**, driven via **PyMAPDL**, submitted
as batch jobs through **Slurm** on AUTh's Aristotle HPC cluster — needed
because an accurate room-scale acoustic mesh, solved across a useful
frequency range, is too computationally demanding for a personal
machine.

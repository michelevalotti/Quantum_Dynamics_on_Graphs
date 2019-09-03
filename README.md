# Transport properties of continuous qunatum walks on fixed graphs

## Overview
Random walks have been the subject of great interest in many areas of science. Their quantum counterpart, quantum random walks, have been shown to have many potential applications. In this work their transport properties on different types of fixed graphs are explored, focusing on continuous quantum walks. These walks are continuous in time but discrete in space, moving on a lattice of fixed ponts. Other effects such as asymmetrical potentials, decoherence and particle interactions are modelled for a simple graph. A theoretical framework for the dynamics of quantum particles on various lattices is a useful ideal model to be compared to experimental results, and in this work it is applied to square lattices, carbon nanotubes and graphs with a high degree of clustering. The results from the simulation of nanotubes show that this model can replicate their transport properties despite not directly accounting for their electrical properties. Results from random graphs show the importance of a regular structure and high connectivity for fast quantum transport.


## Project Structure
A more detailed description of the project are presented in the project report. Continuous quantum walks have been explored in 1 and 2 dimensions, using the walk on the line as both a proof of concept and a simple enviroinment to test more complicated effects like decoherence and asymmetric potentials, which will later be applied in 2 dimensions as well.

- 1 dimension \
Simulations include coined (discrete in space and time) and continuous (discrete in space but not in time) quantum walks, simulated with a pure wavefunction and with a density matrix, which allows for decoherence effects. The transport properties are measured through standard deviation from the starting point
- 2 dimensions
  - Squares \
  Simulations include multiple (2) particles, a search algorithm, and measures of the transport properties through standard deviation (ibear and radial) and distance between particles. The simulations are run on a square lattices and one of the scripts allows to vary the length of the edges.
  - Hexagons \
  Simulations include a 2 dimensional flat hexagonal lattice and a pseudo 3 dimensional one that mimics carbon nanotubes, where the particle moves on the surface of the tube. Transport properties are also measured through the arrival probability, or the probability for the particle to reach one end of the tube.
  - Clusters \
  Simulations include graphs formed by random clusters and clusters with a regualar, hexagonal pattern, connected in more or less symmetric ways. Transport properties are measured through standard deviation and arrival probability, and the results are compared to those obtained on the more symmetric carbon nanotubes. Here we find that a regular pattern is fundamental in the efficiency of transport for particles on a graph, and even small irregularties cause a significant drop in the speed of transport.

# Fast and efficient stochastic integration routines

Thomas Breunung and Balakumar Balachandran

Department of Mechanical Engineering, University of Maryland, College Park, Maryland 20742, USA

# License
This software is made public for research use only. It may be modified and redistributed under the terms of the MIT License.

# Content
This repsitory accompanies [1].

## Small noise expansion
The folder "Smallnoise_straightforward" contains code computing the small noise expansion for the Duffing oscillator. It reproduces the Fig 1b in Section 3 of [1]. 

## Small noise integrator (SNI)
The folder "SNI_evaluate" contains code written to perform the numerical experiments detailed in section 4 of [1].
"SNI_eval_osci_chai.m" computes the sample paths of the oscillator chain via the SNI algorithm and compares the outcome to the Euler-Maruyama approximation. 
The file "SNI_eval_osci_chain_OM1p9.m" performs a similar numerical experiment with a higher excitation frequency. 
The randomly parameteriyzed oscillator chain is evaluates in "SNI_eval_osci_chain_randpars.m". 

For further information please also look at the comments in the code. 

## Gaussian kernel approximation (GKA)
The folder "GKA" contains code for section 5 of [1]. "Stochint_GKA_1D.m" performs the GKA for the Duffing oscillator and "Stochint_GKA_3D.m" for a chain with three masses.

# Reference
[1] T. Breunung and B. Balachandran. Computationally efficient simulations of stochastically perturbed nonlinear dynamical systems, Jounrnal of Computational and Nonlinear Dynamics, accepted. 

# Notes
The codes were tested and debugged using MATLAB R2021a, R2020a and R2019b. 

Maintained by Thomas Breunung, thomasbr at umd dot umd June 15, 2022.

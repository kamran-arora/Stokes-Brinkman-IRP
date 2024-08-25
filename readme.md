Python code to simulate the [Stokes-Brinkman equations](https://onlinelibrary.wiley.com/doi/full/10.1002/fld.5199) using the finite element method

- generalised_stokes.py solves the PDE for a supplied configuration
- configurations can be added in configurations.py by inheriting from the abstract base class
- mg_parallel_channel.py solves the 1D toy problem given in [this paper](https://onlinelibrary.wiley.com/doi/full/10.1002/fld.5199)
- supports [Braess-Sarazin](https://www.sciencedirect.com/science/article/pii/S0168927496000591), and [Vanka](https://www.sciencedirect.com/science/article/pii/S0021999122001851?casa_token=8__56-2JQ1cAAAAA:17c4P8Z0_ViV4frA23RIhAtIKHEyo9jki-dHlx693ux8-mzHNzfcGv0SGiiGtjLZwGu1gzw) either as standalone preconditioner or coupled with multigrid

The only dependency is Firedrake. This can be installed following these [instructions](https://www.firedrakeproject.org/download.html)

## Acknowledgements
I would like to thank Dr. Eike Mueller (University of Bath), Dr. Alexander Belozerov (University of Bath), and Dr. Yang Chen (University of Bath) for their help in producing this code.

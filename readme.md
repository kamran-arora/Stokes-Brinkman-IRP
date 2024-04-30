Python code to simulate the Stokes-Brinkman equations [Chen] [1]

- generalised_stokes.py solves the PDE for a supplied configuration
- configurations can be added in configurations.py
- mg_parallel_channel.py solves a 1D toy problem given in [Chen] [1]
- support Braess-Sarazin [Braess] [2], and Vanka [Vanka] [3] either as standalone
preconditioner or coupled with multigrid

The only dependency is Firedrake. This can be installed following [these] [4] instructions.


[1]: https://onlinelibrary.wiley.com/doi/full/10.1002/fld.5199
[2]: https://www.sciencedirect.com/science/article/pii/S0168927496000591
[3]: https://www.sciencedirect.com/science/article/pii/S0021999122001851?casa_token=8__56-2JQ1cAAAAA:17c4P8Z0_ViV4frA23RIhAtIKHEyo9jki-dHlx693ux8-mzHNzfcGv0SGiiGtjLZwGu1gzw
[4]: https://www.firedrakeproject.org/download.html
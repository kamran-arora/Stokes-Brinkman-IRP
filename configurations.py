from abc import ABC, abstractmethod
from firedrake import *

__all__ = ("ConfigurationSquareWithCylindricalHole", "ConfigurationDualScaleDryFibres")


class Configuration(ABC):
    """Base class for configurations

    All lengthscaless are measured in units of L = 1mm
    """

    @abstractmethod
    def beta(self, mesh):
        """Return beta =  L^2 /k_s

        :arg mesh: mesh for extracting coordinates
        """
        pass


class ConfigurationSquareWithCylindricalHole(Configuration):
    """Square domain with cylindrical hole

    All lengths are measured in units of L = 1mm
    """

    def __init__(self, R=0.4):
        """Initialise instance

        :arg R: radius of hole"""
        self.R = R
        self.Lx = 1.0
        self.Ly = 1.0

    def beta(self, mesh, ks):
        """Return beta =  L^2 /k_s

        :arg mesh: mesh for extracting coordinates
        """
        ks = ks
        x, y = SpatialCoordinate(mesh)
        return conditional((x - 0.5) ** 2 + (y - 0.5) ** 2 < self.R**2, 0.0, 1 / ks)


class ConfigurationDualScaleDryFibres(Configuration):
    """Dual scale dry fibres setup

    All lengths are measured in units of L = 1mm
    """

    def __init__(self, packing="regular"):
        """Initialise instance

        :arg packing: fibre packing, can be 'regular' or 'hexagonal'
        """
        self.packing = packing
        if self.packing == "regular":
            self.Lx = 6.6
            self.Ly = 1.0
        else:
            self.Lx = 8.0
            self.Ly = 1.68
        self.ks = 1.17175e-5
        self.a = 2.2
        self.b = 0.25

    def beta(self, mesh):
        """Return beta =  L^2 /k_s

        :arg mesh: mesh for extracting coordinates
        """

        x, y = SpatialCoordinate(mesh)
        return conditional(
            (x - 0.5 * self.Lx) ** 2 / self.a**2
            + (y - 0.5 * self.Ly) ** 2 / self.b**2
            < 1,
            1 / self.ks,
            0.0,
        )

    @property
    def volume_fraction(self):
        """Ratio a*b/(Lx*Ly)"""
        return self.a * self.b / (self.Lx * self.Ly)

class ParallelChannel(Configuration):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def beta(self, mesh, ks):
        """Return beta =  L^2 /k_s

        :arg mesh: mesh for extracting coordinates
        """
        x, y = SpatialCoordinate(mesh)
        return conditional(x < self.a, 0.0, 1 / ks)
    
class ParallelChannel1D(Configuration):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def beta(self, mesh, ks):
        """Return beta =  1/k_s

        :arg mesh: mesh for extracting coordinates
        """
        x, = SpatialCoordinate(mesh)
        return conditional(x < self.a, 0.0, 1 / ks)
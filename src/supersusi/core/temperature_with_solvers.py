# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:39:09 2020

@author: alauren
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.linalg as linalg


class PeatTemperature:
    def __init__(self, spara, mean_Ta):
        """
        input:
            spara, contains dimensions of soil (peat) object
            mean_Ta is mean air temperature over the whole time, set as lower boundary condition
        """
        # print (spara['nLyrs'], spara['dzLyr'] )
        self.nLyrs_hydro = spara.nLyrs
        self.nLyrs = self.nLyrs_hydro + 30
        self.dz = spara.dzLyr
        self.z = (
            np.cumsum(np.ones(self.nLyrs) * self.dz) - self.dz / 2.0
        )  # depth of the layer center point, m
        self.mean_Ta = mean_Ta
        self.heat_capacity = 3860000.0 * self.dz  # J m-3
        self.heat_of_vaporization = 2467700  # J/kg
        self.Nt = 24  # number of subtimesteps in the time step (here every 2 hrs)

        self.A = self.create_matrix(mode=spara.temperature_solve_mode)

        print("Peat temperature profile initialized")

    def create_matrix_sparse(self):
        """
        Uses a sparse matrix to solve the diff eq.
        """
        T = 86400  # timestep in seconds, s
        t = np.linspace(0, T, self.Nt + 1)  # mesh points in time
        dt = t[1] - t[0]

        D = 1e-7  # Thermal diffusivity of peat, m2 s-1, de Vries 1975
        F = D * dt / self.dz**2

        main = np.zeros(self.nLyrs + 1)
        lower = np.zeros(self.nLyrs)
        upper = np.zeros(self.nLyrs)
        # self.b     = np.zeros(self.nLyrs+1)

        # Precompute sparse matrix
        main[:] = 1 + 2 * F
        lower[:] = -F  # 1
        upper[:] = -F  # 1

        # Insert boundary conditions, Diritchlet (constant value) boundaries
        main[0] = 1
        main[self.nLyrs] = 1

        # Create the main matrix
        return diags(
            diagonals=[main, lower, upper],
            offsets=[0, -1, 1],
            shape=(self.nLyrs + 1, self.nLyrs + 1),
            format="csr",
        )

    def create_matrix_dense(self):
        T = 86400  # timestep in seconds, s
        t = np.linspace(0, T, self.Nt + 1)  # mesh points in time
        dt = t[1] - t[0]

        D = 1e-7  # Thermal diffusivity of peat, m2 s-1, de Vries 1975
        F = D * dt / self.dz**2

        # Create the main matrix
        A = (
            np.diag([1 + 2 * F] * (self.nLyrs + 1))
            + np.diag([-F] * (self.nLyrs), k=1)
            + np.diag([-F] * (self.nLyrs), k=-1)
        )

        # Insert boundary conditions, Diritchlet (constant value) boundaries
        A[0, 0] = 1
        A[self.nLyrs, self.nLyrs] = 1

        return A

    def create_matrix_dense_banded(self):
        T = 86400  # timestep in seconds, s
        t = np.linspace(0, T, self.Nt + 1)  # mesh points in time
        dt = t[1] - t[0]

        D = 1e-7  # Thermal diffusivity of peat, m2 s-1, de Vries 1975
        F = D * dt / self.dz**2

        # The following strange matrix shape is the one needed by the scipy.linalg.solveh_banded() algorithm:
        A = np.zeros((2, self.nLyrs + 1), dtype=float)
        # Row 1 contains the values in the main diagonal
        A[1, :] = 1 + 2 * F
        # Row 0 contains the values  in the upper-diagonal
        A[0, :] = -F

        # Boundary conditions: dirichlet
        A[1, 0] = 1
        A[1, -1] = 1

        return A

    def create_matrix(self, mode: str):
        if mode == "sparse":
            return self.create_matrix_sparse()
        elif mode == "dense":
            return self.create_matrix_dense()
        elif mode == "dense_banded":
            return self.create_matrix_dense_banded()

    def solve_system(self, b, mode: str):
        if mode == "sparse":
            return spsolve(self.A, b)
        elif mode == "dense":
            return linalg.solve(self.A, b)
        elif mode == "dense_banded":
            return linalg.solveh_banded(self.A, b)

    def reset_domain(self):
        self.Tsoil = np.ones(self.nLyrs + 1) * self.mean_Ta
        self.lower_boundary = self.mean_Ta
        self.lower_boundary = self.mean_Ta

    def run_timestep(self, Ta, SWE, efloor, solution_mode: str):
        """
        Parameters
        ----------
        Ta : float
            Air temperature, deg C.
        SWE : float
            Snow water equivalent, m.
        efloor : float
            evaporation from surface layer m.

        Returns
        -------
        z : np array (float)
            depth of layers, m
        Tsoil : np array (float)
            peat temperature (deg C)

        """
        # Cooling by evaporation
        e_consumed = efloor * 1000 * self.heat_of_vaporization / self.Nt
        T_cool = -e_consumed / self.heat_capacity
        if SWE > 0.01:
            Ta = max(-5.0, Ta)
        else:
            Ta = Ta + T_cool

        u = np.zeros(self.nLyrs + 1)
        for n in range(0, self.Nt):
            b = self.Tsoil.copy()
            b[0] = Ta  # temp[n] #0.0  # boundary conditions
            b[-1] = self.lower_boundary
            u[:] = self.solve_system(b, mode=solution_mode)
            # u_1[:] = u
            self.Tsoil = u
        return self.z[: self.nLyrs_hydro], self.Tsoil[: self.nLyrs_hydro]

    def create_outarrays(self, nrounds, ndays, nLyrs):
        peat_temperatures = np.zeros(
            (nrounds, ndays, nLyrs)
        )  # daily peat temperature profiles
        return peat_temperatures

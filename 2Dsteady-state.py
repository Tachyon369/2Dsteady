import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import math


@jit(nopython=True)  # cuda acceleration comment out if using only cpu, works with amd graphics but buggy.
def solve(u, uo, ny, nx, k, c, tol, adi, a, b, dx2, dy2):
    e = 1
    while e > tol:
        for i in range(0, ny+2):
            for j in range(0, nx+2):
                uo[i][j] = u[i][j]
        ma = 0.00
        for i in range(2, ny):
            for j in range(2, nx+adi):
                if j == nx:
                    u[i][nx] = u[i][nx - 1]
                else:
                    u[i][j] = (1 / k) * (((a / dx2) * (uo[i][j + 1] + u[i][j - 1])) + ((b / dy2) * (u[i + 1][j] + u[i - 1][j])) + c)
                if ma < math.sqrt(abs(uo[i][j] ** 2 - u[i][j] ** 2)):
                    ma = math.sqrt(abs(uo[i][j] ** 2 - u[i][j] ** 2))
        e = ma
        print(e)

    return u


# set these values
calc_type = "e"  # type of calc e = using element size; else uses no of elements
pr_type = "t"  # show error or progress, "error" for error
adi = 1  # adiabatic wall on right side, 1 = yes, 0 = no
a = 1   # a == multiplier for d2T/dx2
b = 1   # b == multiplier for d2T/dy2 Note: even when b = 0 and l = 0 1-D isn't possible as element num is < 2
c = 100   # c = constant in eqn / heat generation
# boundary temps; Tr is ignored if adi == 1
Tl = 500
Tt = 300
Tb = 400
Tr = 100
# lengths of either axes and tol == tolerance
lx = 2
ly = 1
tol = 0.0001
# use element size or no.of elements
if calc_type == "e":
    dx = 0.01   # set element size here
    dy = 0.01
    nx = int((lx / dx) + 1)
    ny = int((ly / dy) + 1)
else:
    nx = 500    # set element number here
    ny = 250
    dx = lx/(nx-1)
    dy = ly/(ny-1)
# initial calculations
dx2 = (dx * dx)
dy2 = (dy * dy)
if dx2 == 0:
    dx2 = 1
if dx2 == 0:
    dx2 = 1
k = 2*((a/dx2)+(b/dy2))
# setup of matrices
u = np.zeros((ny+2, nx+2))
uo = np.zeros((ny+2, nx+2))
# boundary conditions setup
for i in range(1, nx + 1):
    u[1][i] = Tb
    u[ny][i] = Tt
for i in range(1, ny + 1):
    u[i][1] = Tl
    if adi == 0:
        u[i][nx] = Tr
# solving matrix
u = solve(u, uo, ny, nx, k, c, tol, adi, a, b, dx2, dy2)
# adjusting output matrix
v = np.flip(u, axis=0)
v = np.delete(v, [0, -1], axis=0)  # delete first and last row
v = np.delete(v, [0, -1], axis=1)  # delete first and last column
if b == 0:
    v = np.delete(v, [0, -1], axis=0)
if a == 0:
    v = np.delete(v, [0, -1], axis=1)
# writing to file
df = pd.DataFrame(v)
filepath = 'output.xlsx'
df.to_excel(filepath, index=False)
# plotting
plt.figure(figsize=(8, 5))
plt.title('Temperature Contour Plot')
plt.xlabel('x from 0 to 2')
plt.ylabel('y from 0 to 1')
plt.imshow(v, cmap='jet')
plt.colorbar()
plt.savefig("temperaturecontour.png", dpi=300)
plt.show()






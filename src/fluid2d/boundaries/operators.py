import numpy as np
from numba import njit, prange


# @njit(cache=True)
# def gradx(table: np.ndarray, result: np.ndarray, d: float):
#     N = table.shape[0]
#     deud = 2*d
#     for i in range(N):
#         result[i,0] = (table[i,1] - table[i,0])/d
#         for j in range(1,N-1):
#             result[i,j] = (table[i,j+1] - table[i,j-1])/deud
#         result[i,N-1] = (table[i,N-1] - table[i,N-2])/d

@njit(cache=True, parallel=True)
def gradx(table: np.ndarray, result: np.ndarray, d: float):

    N = table.shape[0]

    invd = 1.0/d
    inv2d = 1.0/(2*d)


    # bord gauche
    for i in prange(N):
        result[i,0] = (
            table[i,1]-table[i,0]
        )*invd


    # intérieur
    for i in prange(N):
        for j in range(1,N-1):
            result[i,j] = (
                table[i,j+1]
                - table[i,j-1]
            )*inv2d


    # bord droit
    for i in prange(N):
        result[i,N-1] = (
            table[i,N-1]
            - table[i,N-2]
        )*invd


@njit(cache=True, parallel=True)
def gradx_solid(table: np.ndarray, result: np.ndarray, solid_mask: np.ndarray, d: float):
    N = table.shape[0]

    invd = 1.0 / d
    inv2d = 1.0 / (2.0 * d)

    for i in prange(1, N - 1):
        for j in range(1, N - 1):

            if solid_mask[i, j]:
                result[i, j] = 0.0
                continue

            # Est
            if solid_mask[i, j + 1]:
                east = table[i, j]
            else:
                east = table[i, j + 1]

            # Ouest
            if solid_mask[i, j - 1]:
                west = table[i, j]
            else:
                west = table[i, j - 1]

            result[i, j] = (east - west) * inv2d

    # bords
    for i in prange(N):
        result[i, 0] = (
            table[i, 1] - table[i, 0]
        ) * invd

        result[i, N - 1] = (
            table[i, N - 1] - table[i, N - 2]
        ) * invd

# @njit(cache=True)
# def grady(table: np.ndarray, result: np.ndarray, d: float):
#     N = table.shape[0]
#     deud = 2*d
#     for j in range(N):
#         result[0,j] = (table[1,j] - table[0,j])/d
#         for i in range(1,N-1):
#             result[i,j] = (table[i+1,j] - table[i-1,j])/deud
#         result[N-1,j] = (table[N-1,j] - table[N-2,j])/d

# @njit(cache=True)
# def grady2(table, result, d):
#     N = table.shape[0]
#     deud = 2*d
#     for i in range(N):
#         for j in range(N):
#             if i==0:
#                 result[i,j] = (table[i+1,j] - table[i,j])/d
#             elif i==N-1:
#                result[i,j] = (table[i,j] - table[i-1,j])/d
#             else:
#                 result[i,j] = (table[i+1,j] - table[i-1,j])/deud

@njit(cache=True, parallel=True)
def grady(table: np.ndarray, result: np.ndarray, d: float):

    N = table.shape[0]
    invd = 1.0 / d
    invdeud = 1.0 / (2*d)

    # ligne du haut
    for j in prange(N):
        result[0, j] = (table[1, j] - table[0, j]) * invd

    # intérieur
    for i in prange(1, N-1):
        for j in range(N):
            result[i, j] = (
                table[i+1, j]
                - table[i-1, j]
            ) * invdeud

    # ligne du bas
    for j in prange(N):
        result[N-1, j] = (
            table[N-1, j]
            - table[N-2, j]
        ) * invd

@njit(cache=True, parallel=True)
def grady_solid(table: np.ndarray, result: np.ndarray, solid_mask: np.ndarray, d: float ):
    N = table.shape[0]

    invd = 1.0 / d
    inv2d = 1.0 / (2.0 * d)

    for i in prange(1, N - 1):
        for j in range(1, N - 1):

            if solid_mask[i, j]:
                result[i, j] = 0.0
                continue

            if solid_mask[i + 1, j]:
                south = table[i, j]
            else:
                south = table[i + 1, j]

            if solid_mask[i - 1, j]:
                north = table[i, j]
            else:
                north = table[i - 1, j]

            result[i, j] = (south - north) * inv2d

    for j in prange(N):
        result[0, j] = (
            table[1, j] - table[0, j]
        ) * invd

        result[N - 1, j] = (
            table[N - 1, j] - table[N - 2, j]
        ) * invd

# @njit(cache=True)
# def grad(table, resultx, resulty, d):
#     N = table.shape[0]
#     invd = 1.0/d
#     inv2d = 1.0/(2*d)
#     # gradx
#     for i in range(N):
#         resultx[i,0] = (table[i,1]-table[i,0])*invd
#         for j in range(1,N-1):
#             resultx[i,j] = (table[i,j+1]-table[i,j-1])*inv2d
#         resultx[i,N-1] = (table[i,N-1]-table[i,N-2])*invd
#     # grady
#     for j in range(N):
#         resulty[0,j] = (table[1,j]-table[0,j])*invd
#         for i in range(1,N-1):
#             resulty[i,j] = (table[i+1,j]-table[i-1,j])*inv2d
#         resulty[N-1,j] = (table[N-1,j]-table[N-2,j])*invd

@njit(cache=True, parallel=True)
def grad(table: np.ndarray, resultx: np.ndarray, resulty: np.ndarray, d: float):

    N = table.shape[0]

    invd = 1.0/d
    inv2d = 0.5/d

    # X
    for i in prange(N):

        resultx[i,0] = (
            table[i,1]-table[i,0]
        )*invd

        for j in range(1,N-1):

            resultx[i,j] = (
                table[i,j+1]
                -
                table[i,j-1]
            )*inv2d

        resultx[i,N-1] = (
            table[i,N-1]
            -
            table[i,N-2]
        )*invd


    # Y
    for i in prange(N):

        resulty[i,0] = (
            table[i+1,0]
            -
            table[i,0]
        )*invd

    for i in prange(1,N-1):

        for j in range(N):

            resulty[i,j] = (
                table[i+1,j]
                -
                table[i-1,j]
            )*inv2d

    for j in prange(N):

        resulty[N-1,j] = (
            table[N-1,j]
            -
            table[N-2,j]
        )*invd




# @njit(cache=True)
# def div(tablex: np.ndarray, tabley: np.ndarray, result: np.ndarray, gradx_tmp: np.ndarray, grady_tmp: np.ndarray, d: float):

#     N = tablex.shape[0]

#     gradx(tablex, gradx_tmp, d)
#     grady(tabley, grady_tmp, d)

#     for i in range(N):
#         for j in range(N):
#             result[i,j] = gradx_tmp[i,j] + grady_tmp[i,j]


# @njit(cache=True)
# def rot(tablex: np.ndarray, tabley: np.ndarray, result: np.ndarray, gradx_tmp: np.ndarray, grady_tmp: np.ndarray, d: float):

#     N = tablex.shape[0]

#     gradx(tabley, gradx_tmp, d)
#     grady(tablex, grady_tmp, d)

#     for i in range(1,N-1):
#         for j in range(1,N-1):
#             result[i,j] = gradx_tmp[i,j] - grady_tmp[i,j]




@njit(cache=True, parallel=True)
def divergence(vx: np.ndarray, vy: np.ndarray, result: np.ndarray, dx: float):
    N = vx.shape[0]
    invdeud = 1.0/(2*dx)
    for i in prange(1,N-1):
        for j in range(1,N-1):
            result[i,j] = (
                vx[i,j+1]-vx[i,j-1]
                +vy[i+1,j]-vy[i-1,j]
            )*invdeud

@njit(cache=True, parallel=True)
def divergence_solid( vx: np.ndarray, vy: np.ndarray, result: np.ndarray, dx: float, solid_mask: np.ndarray):
    N = vx.shape[0]

    inv2d = 1.0 / (2.0 * dx)

    for i in prange(1, N - 1):
        for j in range(1, N - 1):

            if solid_mask[i, j]:
                result[i, j] = 0.0
                continue

            if solid_mask[i, j + 1]:
                vx_e = 0.0
            else:
                vx_e = vx[i, j + 1]

            if solid_mask[i, j - 1]:
                vx_w = 0.0
            else:
                vx_w = vx[i, j - 1]

            if solid_mask[i + 1, j]:
                vy_s = 0.0
            else:
                vy_s = vy[i + 1, j]

            if solid_mask[i - 1, j]:
                vy_n = 0.0
            else:
                vy_n = vy[i - 1, j]

            result[i, j] = (
                vx_e - vx_w
                + vy_s - vy_n
            ) * inv2d


@njit(cache=True, parallel=True)
def rotationel_solid(tablex: np.ndarray, tabley: np.ndarray, result: np.ndarray, d: float, solid_mask: np.ndarray):

    N = tablex.shape[0]
    invdeud = 1.0/(2*d)

    for i in prange(1,N-1):
        for j in range(1,N-1):
            if solid_mask[i, j]:
                result[i, j] = 0.0
                continue

            if solid_mask[i, j + 1]:
                vy_e = 0.0
            else:
                vy_e = tabley[i, j + 1]

            if solid_mask[i, j - 1]:
                vy_w = 0.0
            else:
                vy_w = tabley[i, j - 1]

            if solid_mask[i + 1, j]:
                vx_s = 0.0
            else:
                vx_s = tablex[i + 1, j]

            if solid_mask[i - 1, j]:
                vx_n = 0.0
            else:
                vx_n = tablex[i - 1, j]
            result[i,j] = (
                (vy_e-vy_w)
                -
                (vx_s-vx_n)
                )*invdeud

@njit(cache=True, parallel=True)
def rotationel(tablex: np.ndarray, tabley: np.ndarray, result: np.ndarray, d: float):

    N = tablex.shape[0]
    invdeud = 1.0/(2*d)

    for i in prange(1,N-1):
        for j in range(1,N-1):
            result[i,j] = (
                (tabley[i,j+1]-tabley[i,j-1])
                -
                (tablex[i+1,j]-tablex[i-1,j])
                )*invdeud



@njit(cache=True)
def lap(table: np.ndarray, table_tmp: np.ndarray, d: float):

    N = table.shape[0]
    dcar = d**2

    for i in range(1,N-1):
        for j in range(1,N-1):

            table_tmp[i,j] = (
                table[i,j+1]
                + table[i,j-1]
                + table[i+1,j]
                + table[i-1,j]
                - 4*table[i,j]
            )/dcar

    for i in range(N):
        table_tmp[0,i] = 0
        table_tmp[i,0] = 0
        table_tmp[N-1,i] = 0
        table_tmp[i,N-1] = 0




EPSILON = 1e-8

@njit(cache=True, parallel=True)
def norm(tablex: np.ndarray, tabley: np.ndarray, result: np.ndarray):

    N = tablex.shape[0]

    for i in prange(N):
        for j in range(N):
            result[i,j] = np.sqrt(tablex[i,j]**2 + tabley[i,j]**2)


@njit(cache=True, parallel=True)
def normalize(tablex: np.ndarray, tabley: np.ndarray, norm_tmp: np.ndarray, resultx: np.ndarray, resulty: np.ndarray):

    N = tablex.shape[0]
    norm(tablex, tabley, norm_tmp)

    for i in prange(N):
        for j in range(N):

            inv = 1.0/(norm_tmp[i,j] + EPSILON)

            resultx[i,j] = tablex[i,j]*inv
            resulty[i,j] = tabley[i,j]*inv


@njit(cache=True, parallel=True)
def null_fill(vx: np.ndarray, vy: np.ndarray, solid_mask: np.ndarray):
    N=vx.shape[0]
    for i in prange(N):
        for j in range(N):
            if solid_mask[i,j]:
                vx[i,j]=0.0
                vy[i,j]=0.0
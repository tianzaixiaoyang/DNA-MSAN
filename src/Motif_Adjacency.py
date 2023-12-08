import numpy as np


def check_zero_matrix(W):

    if np.all(W == 0):
        raise ValueError("Motif_Adjacency_Matrix W is a zero matrix.")
    else:
        print("Motif_Adjacency_Matrix W is not a zero matrix. Continue with the program.")


def DirectionalBreakup(A):

    A[np.where(A)] = 1 
    B = np.logical_and(A, A.T).astype(int)
    U = A - B
    G = np.logical_or(A, A.T).astype(int)
    return B, U, G


def MotifAdjacency(A, motif):

    global W

    A = A - np.diag(np.diag(A))
    A = np.minimum(A, 1) 

    motif = motif.lower()
    if motif == "m1":
        B, U, G = DirectionalBreakup(A)
        W = np.multiply(np.dot(U, U), U)
    elif motif == "m2":
        B, U, G = DirectionalBreakup(A)
        C = np.multiply(np.dot(B, U), U.T) + np.multiply(np.dot(U, B), U.T) + np.multiply(np.dot(U, U), B)
        W = C + C.T
    elif motif == "m3":
        B, U, G = DirectionalBreakup(A)
        C = np.multiply(np.dot(B, B), U) + np.multiply(np.dot(B, U), B) + np.multiply(np.dot(U, B), B)
        W = C + C.T
    elif motif == "m4":
        B, U, G = DirectionalBreakup(A)
        W = np.multiply(np.dot(B, B), B)
    elif motif == "m5":
        B, U, G = DirectionalBreakup(A)
        T1 = np.multiply(np.dot(U, U), U)
        T2 = np.multiply(np.dot(U.T, U), U)
        T3 = np.multiply(np.dot(U, U.T), U)
        C = T1 + T2 + T3
        W = C + C.T
    elif motif == "m6":
        B, U, G = DirectionalBreakup(A)
        C1 = np.multiply(np.dot(U, B), U)
        C1 = C1 + C1.T
        C2 = np.multiply(np.dot(U.T, U), B)
        W = C1 + C2
    elif motif == "m7":
        B, U, G = DirectionalBreakup(A)
        C1 = np.multiply(np.dot(U.T, B), U.T)
        C1 = C1 + C1.T
        C2 = np.multiply(np.dot(U, U.T), B)
        W = C1 + C2
    elif motif == "m8":
        B, U, G = DirectionalBreakup(A)
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J = np.where(U[i, :])[0]
            for j1 in range(len(J)):
                for j2 in range(j1 + 1, len(J)):
                    k1 = J[j1]
                    k2 = J[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "m9":
        B, U, G = DirectionalBreakup(A)
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J1 = np.where(U[i, :])[0]
            J2 = np.where(U[:, i])[0]
            for j1 in range(len(J1)):
                for j2 in range(len(J2)):
                    k1 = J1[j1]
                    k2 = J2[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "m10":
        B, U, G = DirectionalBreakup(A.T)
        A = A.T
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J = np.where(U[i, :])[0]
            for j1 in range(len(J)):
                for j2 in range(j1 + 1, len(J)):
                    k1 = J[j1]
                    k2 = J[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "m11":
        B, U, G = DirectionalBreakup(A)
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J1 = np.where(B[i, :])[0]
            J2 = np.where(U[i, :])[0]
            for j1 in range(len(J1)):
                for j2 in range(len(J2)):
                    k1 = J1[j1]
                    k2 = J2[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "m12":
        B, U, G = DirectionalBreakup(A.T)
        A = A.T
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J1 = np.where(B[i, :])[0]
            J2 = np.where(U[i, :])[0]
            for j1 in range(len(J1)):
                for j2 in range(len(J2)):
                    k1 = J1[j1]
                    k2 = J2[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "m13":
        B, U, G = DirectionalBreakup(A)
        W = np.zeros(G.shape)
        N = G.shape[0]

        for i in range(N):
            J = np.where(B[i, :])[0]
            for j1 in range(len(J)):
                for j2 in range(j1 + 1, len(J)):
                    k1 = J[j1]
                    k2 = J[j2]
                    if A[k1, k2] == 0 and A[k2, k1] == 0:
                        W[i, k1] += 1
                        W[i, k2] += 1
                        W[k1, k2] += 1

        W = W + W.T
    elif motif == "bifan":
        B, U, G = DirectionalBreakup(A)
        NA = np.logical_not(A) & np.logical_not(A.T)
        W = np.zeros(G.shape)
        ai, aj = np.where(np.triu(NA, 1))

        for ind in range(len(ai)):
            x = ai[ind]
            y = aj[ind]
            xout = np.where(U[x, :])[0]
            yout = np.where(U[y, :])[0]
            common = np.intersect1d(xout, yout)
            nc = len(common)
            for i in range(nc):
                for j in range(i + 1, nc):
                    w = common[i]
                    v = common[j]
                    if NA[w, v] == 1:
                        W[x, y] += 1
                        W[x, w] += 1
                        W[x, v] += 1
                        W[y, w] += 1
                        W[y, v] += 1
                        W[w, v] += 1

        W = W + W.T
    elif motif == "edge":
        _, _, W = DirectionalBreakup(A)
    else:
        print("Unknown motif %s', motif")
    return W


def Normal_MotifAdjacency(MotifAdjacency):

    check_zero_matrix(MotifAdjacency)

    sum_neighs = MotifAdjacency.sum(1)
    M = np.diag(sum_neighs)

    for x in range(M.shape[0]):
        if M[x, x] == 0:
            M[x, x] = 1

    M_inv = np.linalg.inv(M)

    R = np.matmul(M_inv, MotifAdjacency)

    return R




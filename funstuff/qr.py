import numpy as np
from scipy.linalg import eigh as speigh

def eigh(H, eps=1e-10):
    U = diagonalize(H)
    D = U.T.conj() @ H @ U
    eigvals = np.diag(D).real
    eigvecs = U.T.real
    eigvecs /= np.linalg.norm(eigvecs, axis=1)
    return eigvals, eigvecs


def zeroify(a, eps=1e-10):
    a.real[abs(a.real) < eps] = 0
    a.imag[abs(a.imag) < eps] = 0
    return a


def diagonalize(H, eps=1e-10, maxiter=1000):
    U = tridiagonalize(H)
    for i in range(maxiter):
        A = U.T.conj() @ H @ U
        Qd = qriter(A)
        U = U @ Qd.T.conj()
        if (
            (np.diag(H, -1) < eps).all()
            and (np.diag(H, 1) < eps).all()
        ): break
    return U


def qriter(A):
    dim = A.shape[0]
    Qd = np.eye(dim, dim, dtype=complex)
    for i in range(dim-1):
        T = Qd @ A
        x1 = T[i,i]
        x2 = T[i+1,i]
        x3 = np.sqrt(x1**2 + x2**2)
        c = x1 / x3
        s = x2 / x3
        G = np.eye(dim, dim, dtype=complex)
        G[i:i+2,i:i+2] = np.mat([[c, s], [-s, c]])
        Qd = G @ Qd
    return Qd


def tridiagonalize(H):
    dim = H.shape[0]
    U = np.eye(dim, dtype=complex)
    for i in range(dim-1):
        T = U.T.conj() @ H @ U
        P = np.eye(dim, dtype=complex)
        b = np.eye(dim-i-1, 1)
        v = T[i+1:,i]
        P[i+1:,i+1:] = householder(v, b)
        U = U @ P
    return U


def householder(v, b, phi=None):
    phi = - np.angle(v.T.conj() @ b)[0,0]
    u = v - np.exp(1j * phi) * np.linalg.norm(v) * b
    P = np.eye(b.shape[0], dtype=complex)
    if np.linalg.norm(u) == 0:
        return P
    else:
        u = u / np.linalg.norm(u)
        P -= 2 * u @ u.T.conj()
        return P


def random_H(dim):
    rng = np.random.default_rng()
    Re = (rng.standard_normal((dim, dim)) * 10) // 2
    Im = (rng.standard_normal((dim, dim)) * 10) // 2
    A = Re #+ 1j * Im
    A = np.matrix(A)
    H = 0.5 * (A + A.conj().T)
    return H


if __name__ == '__main__':
    H = random_H(4)
    print('H =', H, sep='\n')
    print('--------------------')
    eval, evec = eigh(H)
    speval, spevec = speigh(H)
    print('eigenvalues:')
    print(eval, speval)
    print('eigenvectors:')
    print(evec, spevec, sep='\n')
    print('sample eigenvector: v0 and v0 @ H / E0')
    print(evec[0], H @ evec[0] / eval[0])

import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


TOL = 1e-7
N = 30
tau = .05
round = lambda x: np.round(x, 5)
# seed = np.random.randint(0,1000)
# print(seed)
#np.random.seed(11)

def cross_matrix(u: np.ndarray) -> np.ndarray:
    """Turns a 3-vector u into a matrix W so that u x v = Wv for any vector 
    v."""
    return np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

def pure(q: np.ndarray) -> np.ndarray:
    """Returns the pure part of a quaternion."""
    return q[1:]

def real(q: np.ndarray) -> float:
    """Returns the real part of a quaternion."""
    return q[0]

def inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) 

def pure2quat(u: np.ndarray) -> np.ndarray:
    """Takes a 3 vector and appends a 0 in front, turning it into a 
    quaternion."""
    return np.array([0, u[0], u[1], u[2]])

def norm(v: np.ndarray) -> float:
    """Returns the norm of a vector."""
    return np.linalg.norm(v)

def mult_matrix(q: np.ndarray, adjoint: bool=False) -> np.ndarray:
    if len(q) == 3:
        q = pure2quat(q)
    if not adjoint:
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]]
        ])
    else:
        return np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], q[3], -q[2]],
            [q[2], -q[3], q[0], q[1]],
            [q[3], q[2], -q[1], q[0]]
        ])
    
def mult(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    if len(p) == 3:
        p = pure2quat(p)
    if len(q) == 3:
        q = pure2quat(q)
    out = mult_matrix(q)@p
    return out

def rotate(v, q):
    return pure(mult(mult(q, v), inv(q)))

def rand_quat():
    q = np.random.randn(4)
    return q/norm(q)

def lie_bracket(xi, eta):
    xi = pure2quat(xi)
    eta = pure2quat(eta)
    return pure(mult(xi, eta) - mult(eta, xi))

def exp(u: np.ndarray) -> np.ndarray:
    theta = norm(u)
    if theta < TOL:
        return np.array([1,0,0,0])
    q = np.zeros(4)
    q[0] = np.cos(theta)
    q[1:] = u*np.sin(theta)/theta
    return q

def log(q):
    q0 = real(q)
    qv = pure(q)
    theta_v = norm(qv)
    if theta_v < TOL:
        return np.zeros(3)
    return qv/theta_v*np.arccos(q0)

def dexp_du(u: np.ndarray) -> np.ndarray:
    theta = norm(u)
    if theta < TOL:
        D = np.zeros((4,3))
        D[1:,:] = np.eye(3)
        return D
    inv_theta = 1/theta
    Q = np.zeros((4,3))
    norm_u = u*inv_theta
    sin_theta = np.sin(theta)
    sinc_theta = sin_theta*inv_theta
    grad_sinc = np.cos(theta)*inv_theta - sinc_theta*inv_theta
    Q[0,:] = -sin_theta*norm_u
    Q[1:,:] = sinc_theta*np.eye(3) + \
        grad_sinc*norm_u*np.vstack([u,u,u]).T
    return Q

def f(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    return mult(q, exp(tau*u))

def df_dq(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    p = exp(tau*u)
    P = mult_matrix(p, adjoint=True)
    return P

def df_du(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    Q = mult_matrix(q)
    return tau*Q@dexp_du(tau*u)

def diff_f_q(q, u, dq):
    return mult(dq, exp(tau*u))

def diff_exp(u, xi):
    """diff of exp at u along xi."""
    return dexp_du(u)@xi

def diff_f_u(q, u, xi):
    return tau*mult(q, diff_exp(tau*u, xi))

def get_frame(q):
    e1 = mult(q, np.array([0,1,0,0]))
    e2 = mult(q, np.array([0,0,1,0]))
    e3 = mult(q, np.array([0,0,0,1]))
    return [e1, e2, e3]

def translate(q, xi):
    return mult(q, exp(xi))

def retract(q, dq):
    xi = mult(inv(q), dq)
    xi = pure(xi)
    return translate(q, xi)

def log_of_difference(q_head, q_base):
    return log(mult(inv(q_base), q_head))

def get_coords(q, v):
    E = get_frame(q)
    v_coords = np.zeros(3)
    for i in range(3):
        v_coords[i] = E[i].T@v
    return v_coords

def get_vector(q, v_coords):
    v = np.zeros(4)
    E = get_frame(q)
    for i in range(3):
        v += v_coords[i]*E[i]
    return v

def A(q, u):
    q_next = f(q, u)
    E = get_frame(q)
    A_mat = np.zeros((3,3))
    for j in range(3):
        Aj = diff_f_q(q, u, E[j])
        A_mat[:,j] = get_coords(q_next, Aj)
    return A_mat

def A(q, u):
    q_plus = f(q, u)
    E = get_frame(q)
    A_mat = np.zeros((3,3))
    for j in range(3):
        Aj = diff_f_q(q, u, E[j])
        A_mat[:,j] = get_coords(q_plus, Aj)
    return A_mat


def A_aug(q_next, q, u):
    q_plus = f(q, u)
    p = mult(inv(q_next), q_plus)
    E = get_frame(q)
    A_mat = np.zeros((3,3))
    for j in range(3):
        Aj = diff_f_q(q, u, E[j])
        #Aj = mult(inv(q_plus), Aj)
        #Aj = mult(q_next, Aj)
        Aj = mult(inv(q_next), Aj)
        Aj = diff_log(p, Aj)
        Aj = mult(q_next, Aj)
        A_mat[:,j] = get_coords(q_next, Aj)
    return A_mat

def B(q, u):
    q_plus = f(q, u)
    B_mat = np.zeros((3,3))
    E = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    for j in range(3):
        Bj = diff_f_u(q, u, E[j])
        B_mat[:,j] = get_coords(q_plus, Bj)
    return B_mat

def B_aug(q_next, q, u):
    q_plus = f(q, u)
    p = mult(inv(q_next), q_plus)
    B_mat = np.zeros((3,3))
    E = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    for j in range(3):
        Bj = diff_f_u(q, u, E[j])
        #Bj = mult(inv(q_plus), Bj)
        #Bj = mult(q_next, Bj)
        Bj = mult(inv(q_next), Bj)
        Bj = diff_log(p, Bj)
        Bj = mult(q_next, Bj)
        B_mat[:,j] = get_coords(q_next, Bj)
    return B_mat

def sq_dist(p, q):
    s = log_of_difference(p, q)
    return s.T@s

def init_traj(q0, q_des):
    # q = np.zeros((4,N + 1))
    # u = np.zeros((3,N))
    # q[:,0] = q0
    # d = np.sqrt(sq_dist(q_des, q0))
    # for k in range(N):
    #     qk = q[:,k]
    #     if sq_dist(q_des, qk) > (np.pi/2)**2:
    #         print("trajectory is outside geodesic convex space")
    #     uk = .1*log(mult(inv(qk), q_des))
    #     q[:,k + 1] = mult(qk, exp(uk))
    #     u[:,k] = uk
    # return q, u
    q = np.zeros((4, N + 1))
    u = np.zeros((3,N))
    q[:,0] = q0
    w = log(mult(inv(q0), q_des))
    #d = np.sqrt(sq_dist(q_des, q0))
    for k in range(N):
        qk = mult(q0, exp(k/N*w))
        q[:,k] = qk
        u[:,k] = w
    q[:,N] = q_des
    return q,u

t_o = np.array([1,0,0])
y_b = np.array([1,0,0])
# t_o = np.random.randn(3)
# t_o = t_o/norm(t_o)
# y_b = np.random.randn(3)
# y_b = y_b/snorm(y_b)
theta_max = 20/180*np.pi
cos_theta_max = np.cos(theta_max)
y_b_cross = mult_matrix(y_b, adjoint=True)
M_H = np.block([
    [np.zeros((4,4)), y_b_cross.T],
    [y_b_cross, np.zeros((4,4))]
])
Q = np.block([
    [np.eye(4)],
    [1/2*mult_matrix(t_o)]
])

def trajectory(q0, u):
    q = np.zeros((4, N + 1))
    q[:,0] = q0
    for k in range(N):
        q[:,k + 1] = f(q[:,k], u[:,k])
    return q

def cvx_penalty(v):
    return cvx.norm(v, 1)**2

def penalty(v):
    return np.linalg.norm(v, 1)**2

def s(q):
    y_o = rotate(y_b, q)
    out = cos_theta_max - t_o.T@y_o
    return -out

def diff_s(q, v):
    eta = pure(mult(inv(q), v))
    out = -t_o.T@(rotate(lie_bracket(eta, y_b), q))
    return -out

def S(q):
    S_mat = np.zeros(3)
    E = get_frame(q)
    for i in range(3):
        S_mat[i] = diff_s(q, E[i])
    return S_mat

def dS_dq(q):
    out = 2*Q.T@M_H@Q@q
    return out

def dlog_dq(q):
    q0 = real(q)
    qv = pure(q)
    theta = norm(qv)
    if theta < TOL:
        D = np.zeros((3,4))
        D[:,1:] = np.eye(3)
        return D  
    inv_theta = 1/theta
    D = np.zeros((3,4))
    D[:,0] = -qv*inv_theta/np.sqrt(1 - q0**2)
    D[:,1:] = np.arccos(q0)*inv_theta*(np.eye(3) - np.outer(qv, qv)*inv_theta*inv_theta)
    return D

def diff_log(q, dq):
    return dlog_dq(q)@dq

def Hess_dist(q, q0, xi):
    """Hessian of f(q) = d(q, q0)^2 along xi."""
    p = mult(inv(q0), q)
    return diff_log(p, mult(q, xi))

def Hess_dist_coords(q, q0):
    p = mult(inv(q), q0)
    log_q_q0 = mult(q, pure2quat(log(p)))
    theta = norm(log_q_q0)
    if theta < TOL:
        return np.eye(3)
    f_theta = theta/np.sin(theta)
    u = log_q_q0/theta
    uu = np.outer(u,u)
    H = uu + f_theta*np.cos(theta)*(np.eye(4) - np.outer(q, q) - uu)
    Q = mult_matrix(q)
    return (Q.T@H@Q)[1:,1:]

def cvx_penalty(v):
    return cvx.norm(v, 1)

def penalty(v):
    return np.linalg.norm(v, 1)


def plot_boresight_trajectory(q, riem_q):
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')

    # plot sphere
    U, V = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    X = np.cos(U)*np.sin(V)
    Y = np.sin(U)*np.sin(V)
    Z = np.cos(V)
    ax.plot_surface(X, Y, Z, alpha=.1)

    # plot cone avoidance set

    y = np.zeros((3, 50))
    e1 = np.array([1,0,0])
    y0 = np.array([np.cos(theta_max), np.sin(theta_max), 0])
    k = 0
    for t in np.linspace(0, np.pi*2, 50):
        qt = exp(t*e1)
        yt = rotate(y0, qt)
        y[:,k] = yt
        k += 1

    ax.plot(y[0,:], y[1,:], y[2,:], linewidth=2, c='red', markersize=2, label='keep out zone')
    ax.plot(t_o[0], t_o[1], t_o[2], '.', c='red', markersize=2, label='t_o')






    # plot traj
    y = np.zeros((3,N))
    for k in range(N):
        qk = q[:,k]
        y_o = rotate(y_b, qk)
        y[:,k] = y_o
    y_o_0 = rotate(y_b, q0)
    y_o_des = rotate(y_b, q_des)
    ax.plot(y[0,:], y[1,:], y[2,:], '.', c='orange',  markersize=5, label='SCvx')

    y = np.zeros((3,N))
    for k in range(N):
        qk = riem_q[:,k]
        y_o = rotate(y_b, qk)
        y[:,k] = y_o

    ax.plot(y[0,:], y[1,:], y[2,:], '.', c='blue', markersize=7, label='iSCvx')

    ax.plot(y_o_0[0], y_o_0[1], y_o_0[2], '.', markersize=10, c='black', label='start')
    ax.plot(y_o_des[0], y_o_des[1], y_o_des[2], '.', c='green', markersize=10, label='desired')
    
    plt.title(f"Boresight trajectory. N={N}, tau={tau}, theta_max={theta_max/np.pi*180}")
    plt.grid()
    plt.legend()
    ax.view_init(elev=17, azim=19)
    plt.savefig(f"./N{N}_tau{tau}_theta_max{theta_max/np.pi*180}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

riem_hist = []
E_hist = []

while True:
    while True:
        xi = np.random.randn(3)
        xi = xi/norm(xi)
        xi = np.random.rand()*np.pi/2*xi
        q_des = exp(xi)
        y_o = rotate(y_b, q_des)
        ang = np.arccos(y_o.T@t_o)
        if ang > theta_max:
            break


    while True:
        #xi = -xi + .01*np.random.randn(3)
        #xi = xi/norm(xi)
        xi = np.random.randn(3)
        xi = xi/norm(xi)
        xi = np.random.rand()*np.pi/2*xi
        q0 = exp(xi)
        y_o = rotate(y_b, q0)
        ang = np.arccos(y_o.T@t_o)
        if ang > theta_max and np.sqrt(sq_dist(q0, q_des)) < np.pi/2:
            break
    

    traj_breaks_constraint = False
    last_pass = False
    q, u = init_traj(q0, q_des)
    for qk in q.T:
        y_o = rotate(y_b, qk)
        ang = np.arccos(y_o.T@t_o)
        if ang < theta_max:
            traj_breaks_constraint = True
    last_pass = ang > theta_max
    if traj_breaks_constraint and last_pass:
        break

# for qk in q.T:
#     y_o = rotate(y_b, qk)
#     ang = np.arccos(y_o.T@t_o)
#     dist2qdes = np.sqrt(sq_dist(qk, q_des))
#     print(ang > theta_max, dist2qdes)
    

print("Trajectory initialized")

def h(q):
    return sq_dist(q, q_des)/2

def diff_h(q, xi):
    return log_of_difference(q, q_des).T@xi

def grad_h(q):
    return mult(q, log_of_difference(q, q_des))

def diff_grad_h(q, dq):
    p = mult(inv(q_des), q)
    return mult(dq, log(p)) + mult(q, diff_log(p, mult(inv(q_des), dq)))

def Hess_h(q, dq):
    p = mult(inv(q_des), q)
    return mult(q, dlog_dq(p)@dq)

penalty_lambda = 1e5
state_lambda = 1
control_lambda = .1
final_state_lambda = 10

def cost(q, u):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*h(qk) + \
            control_lambda*uk.T@uk + \
            penalty_lambda*penalty(log_of_difference(qk_p1, f(qk, uk))) + \
            penalty_lambda*np.abs(s(qk))
    qN = q[:,N]
    J += final_state_lambda*h(qN)
    return J

def true_cost(q, u):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*(qk - q_des).T@(qk - q_des) + \
            control_lambda*uk.T@uk
    qN = q[:,N]
    J += final_state_lambda*(qN - q_des).T@(qN - q_des)
    return J

def Euclidean_cost(q, u):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*(qk - q_des).T@(qk - q_des) + \
            control_lambda*uk.T@uk + \
            penalty_lambda*penalty(qk_p1 - f(qk, uk)) + \
            penalty_lambda*np.abs(s(qk))
    qN = q[:,N]
    J += final_state_lambda*(qN - q_des).T@(qN - q_des)
    return J

def cvx_linearized_cost(q, u, q_des, eta, xi, v, s1):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        s1k = s1[:,k]
        Hk = Hess_dist_coords(qk, q_des)
        pk = log(mult(inv(q_des), qk))
        J += penalty_lambda*cvx_penalty(vk) + \
            penalty_lambda*cvx_penalty(s1k) + \
            control_lambda*cvx.sum_squares(uk + xik) + \
            state_lambda*(h(qk) + pk.T @ etak + 1/2*cvx.quad_form(etak, Hk))
        
    qN = q[:,N]
    pN = log(mult(inv(q_des), qN))
    etaN = eta[:,N]
    HN = Hess_dist_coords(qN, q_des)
    J += final_state_lambda*(h(qN) + pN.T @ etaN + 1/2*cvx.quad_form(etaN, HN))
    return J

def linearized_cost(q, u, q_des, eta, xi, v, s1):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        s1k = s1[:,k]
        Hk = Hess_dist_coords(qk, q_des)
        pk = log(mult(inv(q_des), qk))
        J += penalty_lambda*penalty(vk) + \
            penalty_lambda*penalty(s1k) + \
            control_lambda*np.linalg.norm(uk + xik)**2 + \
            state_lambda*(h(qk) + pk.T @ etak + 1/2*etak@Hk@etak)
        
    qN = q[:,N]
    pN = log(mult(inv(q_des), qN))
    etaN = eta[:,N]
    HN = Hess_dist_coords(qN, q_des)
    J += final_state_lambda*(h(qN) + pN.T @ etaN + 1/2*etaN@HN@etaN)
    return J

def translate_traj(q, eta):
    q_plus = np.zeros((4, N + 1))
    for k in range(N + 1):
        q_plus[:,k] = translate(q[:,k], eta[:,k])
    return q_plus

def euc_cost(q, u):
    J = 0 
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qk_p1 = q[:, k + 1]
        J += state_lambda*(qk - q_des).T@(qk - q_des) + \
            control_lambda*uk.T@uk + \
            penalty_lambda*penalty(qk_p1 - f(qk, uk)) + \
            penalty_lambda*np.abs(s(qk))
    qN = q[:,N]
    J += final_state_lambda*(qN - q_des).T@(qN - q_des)
    return J

def euc_cvx_linearized_cost(q, u, q_des, eta, xi, v, s_prime):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        sk = s_prime[:,k]
        J += state_lambda*cvx.sum_squares(qk + etak - q_des) + \
            control_lambda*cvx.sum_squares(uk + xik) + \
            penalty_lambda*cvx_penalty(vk) + \
            penalty_lambda*cvx_penalty(sk)
    qN = q[:,N]
    etaN = eta[:,N]
    J += final_state_lambda*cvx.sum_squares(qN + etaN - q_des)
    return J

def euc_linearized_cost(q, u, q_des, eta, xi, v, s_prime):
    J = 0
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        etak = eta[:,k]
        xik = xi[:,k]
        vk = v[:,k]
        sk = s_prime[:,k]
        J += state_lambda*np.linalg.norm(qk + etak - q_des)**2 + \
            control_lambda*np.linalg.norm(uk + xik)**2 + \
            penalty_lambda*penalty(vk) + \
            penalty_lambda*penalty(sk)
    qN = q[:,N]
    etaN = eta[:,N]
    J += final_state_lambda*np.linalg.norm(qN + etaN - q_des)**2
    return J


def convex_optimal_control_subproblem(q, u, r):
    eta = cvx.Variable((3, N + 1))
    xi = cvx.Variable((3, N))
    v = cvx.Variable((3, N))
    s1_prime = cvx.Variable((1, N))
    constraints = [eta[:,0] == np.zeros(3)]
    for k in range(N):
        qk = q[:,k]
        qk_p1 = q[:,k + 1]
        uk = u[:,k]
        etak = eta[:,k]
        etak_p1 = eta[:,k + 1]
        xik = xi[:,k]
        Ak = A_aug(qk_p1, qk, uk)
        Bk = B_aug(qk_p1, qk, uk)
        Sk = S(qk)
        vk = v[:,k]
        s1k = s1_prime[:,k]
        constraints.extend([
            etak_p1 == log_of_difference(f(qk, uk), qk_p1) + Ak@etak + Bk@xik + vk,
            cvx.norm(xik, 2) <= r,
            s(qk) + Sk@etak - s1k <= 0,
            s1k >= 0,
        ])
    J = cvx_linearized_cost(q, u, q_des, eta, xi, v, s1_prime)
    problem = cvx.Problem(cvx.Minimize(J), constraints)
    opt_value = problem.solve(solver=cvx.CLARABEL)
    return eta.value, xi.value, v.value, s1_prime.value

def euc_convex_optimal_control_subproblem(q, u, r):
    eta = cvx.Variable((4, N + 1))
    xi = cvx.Variable((3, N))
    v = cvx.Variable((4, N))
    s_prime = cvx.Variable((1, N))
    constraints = [eta[:,0] == np.zeros(4)]
    for k in range(N):
        qk = q[:,k]
        qk_p1 = q[:, k + 1]
        uk = u[:,k]
        etak = eta[:,k]
        etak_p1 = eta[:,k + 1]
        xik = xi[:,k]
        Ak = df_dq(qk, uk)
        Bk = df_du(qk, uk)
        Sk = dS_dq(qk)
        vk = v[:,k]
        sk = s_prime[:,k]
        constraints.extend([
            etak_p1 + qk_p1 == f(qk, uk) + Ak@etak + Bk@xik + vk,
            cvx.norm(xik, 2) <= r,
            s(qk) + Sk@etak - sk <= 0,
            sk >= 0
        ])
    J = euc_cvx_linearized_cost(q, u, q_des, eta, xi, v, s_prime)
    problem = cvx.Problem(cvx.Minimize(J), constraints)
    opt_value = problem.solve(solver=cvx.CLARABEL)
    return eta.value, xi.value, v.value, s_prime.value

r = 1
rl = 0
alpha = 2
beta = 3.2
eps_tol = 1e-3
rho0 = 0
rho1 = .25
rho2 = .7

state_traj_hist = []
control_traj_hist = []
riem_last_q = []
riem_true_cost_hist = [true_cost(q, u)]
try:
    print("RIEMANNIAN")
    k = 0
    r = 1
    while True:
        
        # step 1     
        eta, xi, v, s1_prime = convex_optimal_control_subproblem(q, u, r)

        # step 2    
        Delta_J = cost(q, u) - cost(translate_traj(q, eta), u + xi)
        Delta_L = cost(q, u) - linearized_cost(q, u, q_des, eta, xi, v, s1_prime)
        if np.abs(Delta_J) < eps_tol:
            print(k, "|Delta_J| < eps_tol")
            break
        else:
            rho_k = np.abs(Delta_J)/Delta_L
        # print(
        #     f"step: {k}, " +
        #     f"log10_r: {round(np.log10(r))}, " +
        #     f"true cost: {round(true_cost(q, u))}, " +
        #     f"log10_Delta_J: {round(np.log10(np.abs(Delta_J)))}"
        # )
        # step 3
        
        if rho_k < rho0:
            r = r/alpha
        else:
            q = translate_traj(q, eta)
            u = u + xi
            state_traj_hist.append(q)
            control_traj_hist.append(u)
            riem_true_cost_hist.append(true_cost(q, u))
            riem_last_q.append(norm(q[:,-1] - q_des))
            if rho_k < rho1:
                r = r/alpha
            elif rho_k >= rho2:
                r = r*beta
            
            r = max(r, rl)
            k = k + 1
    riem_hist.append(k)
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qkp1 = q[:,k + 1]
        vk = v[:,k]
        s1k = s1_prime[:,k]
        print(
            f"idx: {k}, " + 
            f"dist2qdes: {np.sqrt(sq_dist(qk, q_des))}, " +
            f"dynamic constr: {round(norm(qkp1 - f(qk, uk)))}, " +
            f"norm quat: {round(norm(qk))}, " + 
            f"state constr: {s(qk) < 0}"
        )
    print("cost: ", true_cost(q, u))
    riem_diff_hist = []
    for i in range(len(state_traj_hist)):
        q_i = state_traj_hist[i]
        u_i = control_traj_hist[i]
        diff = np.linalg.norm(q - q_i)**2 + np.linalg.norm(u - u_i)**2
        riem_diff_hist.append(diff)

    riem_q = q.copy()

    r = 1
    q, u = init_traj(q0, q_des)
    print("EUCLIDEAN")
    state_traj_hist = []
    control_traj_hist = []
    euc_true_cost_hist = [true_cost(q, u)]
    euc_last_q = []
    k = 0

    while True:
        
        # step 1     
        eta, xi, v, s_prime = euc_convex_optimal_control_subproblem(q, u, r)
        qN = q[:,N]
        etaN = eta[:,N]

        # step 2    
        Delta_J = euc_cost(q, u) - euc_cost(q + eta, u + xi)
        Delta_L = euc_cost(q, u) - euc_linearized_cost(q, u, q_des, eta, xi, v, s_prime)
        if np.abs(Delta_J) < eps_tol:
            print(k, "|Delta_J| < eps_tol")
            break
        else:
            rho_k = np.abs(Delta_J)/Delta_L
        # print(
        #     f"step: {k}, " +
        #     f"log10_r: {round(np.log10(r))}, " +
        #     f"true cost: {round(true_cost(q, u))}, " +
        #     f"log10_Delta_J: {round(np.log10(np.abs(Delta_J)))}"
        # )
        # step 3
        
        if rho_k < rho0:
            r = r/alpha
        else:
            
            q = q + eta
            u = u + xi
            state_traj_hist.append(q)
            control_traj_hist.append(u)
            euc_true_cost_hist.append(true_cost(q, u))
            euc_last_q.append(norm(q[:,-1] - q_des))
            if rho_k < rho1:
                r = r/alpha
            elif rho_k >= rho2:
                r = r*beta
            
            r = max(r, rl)
            k = k + 1
            
    E_hist.append(k)
    for k in range(N):
        qk = q[:,k]
        uk = u[:,k]
        qkp1 = q[:,k + 1]
        vk = v[:,k]
        sk = s_prime[:,k]
        print(
            f"idx: {k}, " + 
            f"dist2qdes: {np.sqrt(sq_dist(qk, q_des))}, " +
            f"dynamic const: {round(norm(qkp1 - f(qk, uk)))}, " +
            f"norm quat: {round(norm(qk))}, " + 
            f"state constr: {s(qk) < 0}"
        )

    print("cost: ", true_cost(q, u))
    euc_diff_hist = []
    for i in range(len(state_traj_hist)):
        q_i = state_traj_hist[i]
        u_i = control_traj_hist[i]
        diff = np.linalg.norm(q - q_i)**2 + np.linalg.norm(u - u_i)**2
        euc_diff_hist.append(diff)


    # plt.figure(figsize=(6,10))
    

    # plt.subplot(3,1,1)
    # plt.title(f"tau={tau}, N={N}")
    # plt.grid()
    # plt.xlabel("Iteration")
    # plt.ylabel("|Xi - X*|")
    # plt.semilogy(riem_diff_hist[:-1], label='iSCvx')
    # plt.semilogy(euc_diff_hist[:-1], label='SCvx')
    # plt.legend()



    # plt.subplot(3,1,2)
    # plt.grid()
    # plt.xlabel("Iteration")
    # plt.ylabel("C(q,u)")
    # plt.plot(riem_true_cost_hist, label='iSCvx')
    # plt.plot(euc_true_cost_hist, label='SCvx')
    # plt.legend()


    # plt.subplot(3,1,3)
    # plt.grid()
    # plt.xlabel("Iteration")
    # plt.ylabel("|q(N) - q_{des}|")
    # plt.semilogy(riem_last_q, label='iSCvx')
    # plt.semilogy(euc_last_q, label='SCvx')
    # plt.legend()

    # plt.savefig(f"./tau_{tau}_N_{N}.pdf", format="pdf", bbox_inches="tight")
    # plt.tight_layout()

    # plt.show()
    
    plot_boresight_trajectory(q, riem_q)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.grid()
    plt.ylabel("Trajectory cost")
    plt.xlabel("iteration")
    plt.plot(euc_true_cost_hist,label='SCvx')
    plt.plot(riem_true_cost_hist,label='iSCvx')
    plt.legend()

    plt.subplot(2,1,2)
    plt.grid()
    plt.ylabel("Dist to final trajectory")
    plt.xlabel("iteration")
    plt.semilogy(euc_diff_hist,label='SCvx')
    plt.semilogy(riem_diff_hist,label='iSCvx')
    plt.legend()
    plt.show()




except cvx.error.DCPError:
    print("DCPError)")



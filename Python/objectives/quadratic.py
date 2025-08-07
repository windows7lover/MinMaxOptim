import numpy as np

def func_matrices(dim_x, dim_y, params):
    L_xx, L_yy, L_xy, L_yx, mu_x, mu_y = params 
    
    A_k = np.random.rand(dim_x, dim_x)
    U_a, _ = np.linalg.qr(A_k)
    diag_a = np.flip(np.linspace(mu_x, L_xx, dim_x))
    sigma_a = np.diag(diag_a)
    A  = U_a.T @ sigma_a @ U_a

    B_k = np.random.rand(dim_y, dim_x)
    U_b, D, V_b = np.linalg.svd(B_k, full_matrices=False)
    diag_b = np.flip(np.linspace(0, L_xy, dim_y))
    sigma_b = np.diag(diag_b)
    B  = U_b @ sigma_b @ V_b

    C_k = np.random.rand(dim_y, dim_y)
    U_c, _ = np.linalg.qr(C_k)
    diag_c = np.flip(np.linspace(mu_y, L_yy, dim_y))
    sigma_c = np.diag(diag_c)
    C  = U_c.T @ sigma_c @ U_c

    d = 100*np.random.rand(dim_x, 1)
    e = 100*np.random.rand(dim_y, 1)
    f_mat = [A, B, C, d, e]
    return f_mat

def grad(f_mat, x, y):
    A, B, C, d, e = f_mat
    return (A@x + B.T @ y + d), (-C@y + B@x - e)

def compute_initial_best_response(f_mat, initial_point):
    A, B, C, d, e = f_mat
    x0, y0 = initial_point
    x = np.linalg.solve(A,  -B.T@y0-d)
    y = np.linalg.solve(C, B@x0-e)
    return x, y

def compute_nash_equilibrium(f_mat):
    A, B, C, d, e = f_mat
    C_inv = np.linalg.inv(C)
    x = np.linalg.solve(A+B.T@C_inv@B, (B.T@C_inv@e - d))
    y = np.linalg.solve(C, B@x-e)
    return x, y

def compute_hessian_y(f_mat):
    A, B, C, d, e = f_mat
    return -C

def compute_hessian_x(f_mat):
    A, B, C, d, e = f_mat
    return A

def compute_hessian_xy(f_mat):
    A, B, C, d, e = f_mat
    return B

def compute_hessian_yx(f_mat):
    A, B, C, d, e = f_mat
    return B

def compute_function_values(f_mat, x, y):
    A, B, C, d, e = f_mat
    a = 0.5 * x.T @ A @ x
    b = y.T@B@x
    c = - 0.5 * y.T @C @y
    func_value = a + b + c + np.dot(d.T, x) - np.dot(e.T, y)
    return np.squeeze(func_value)

def compute_primal_dual_gap(f_mat, x, y):
    best_x, best_y = compute_initial_best_response(f_mat, (x,y))
    func1 = compute_function_values(f_mat, x, best_y)
    func2 = compute_function_values(f_mat, best_x, y)
    return func1 - func2

def compute_lyapunov(f_mat, params, x, y, x_st, y_st):
    g_x, g_y_st = grad(f_mat, x, y_st)
    g_x_st, g_y = grad(f_mat, x_st, y)
    norm_y_st = np.linalg.norm(g_y_st, ord=2)**2
    norm_x_st = np.linalg.norm(g_x_st, ord=2)**2

    L_xx, L_yy, L_xy, L_yx, mu_x, mu_y = params

    lyap = compute_function_values(f_mat, x, y_st)-compute_function_values(f_mat, x_st, y)+(0.5*(1/mu_x)*norm_x_st)+(0.5*(1/mu_y)*norm_y_st)
    return lyap

def compute_function_diff(f_mat, x, y, x_st, y_st):
    return compute_function_values(f_mat, x, y_st)-compute_function_values(f_mat, x_st, y)

def create_domain_values(dim_x, dim_y):
    
    x1 = np.linspace(-1.1, 1.1, dim_x)
    x2 = np.linspace(-1.1, 1.1, 50)

    xx1, xx2 = np.meshgrid(x1, x2, indexing='ij')
    xx = xx1+xx2
    
    y1 = np.linspace(-1.1, 1.1, dim_y)
    y2 = np.linspace(-1.1, 1.1, 50)
    
    yy1, yy2 = np.meshgrid(y1, y2, indexing='ij')
    yy = yy1+yy2
    return xx.T, yy.T

def compute_function(dim_x, dim_y, f_mat):
    xx, yy = create_domain_values(dim_x, dim_y)
    func_values = []
    for i in range(xx.shape[0]):
        func_values.append(compute_function_values(f_mat, xx[i][:, None], yy[i][:, None]))
    return func_values
        
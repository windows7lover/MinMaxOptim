import numpy as np
from objectives.quadratic import compute_primal_dual_gap, compute_lyapunov, grad

def extra_gradient(N_iter, initial_point, equilibrium_point, params, func_mat, tol=1.0e-5):
    x0, y0 = initial_point
    x, y = x0, y0
    x_nash, y_nash = equilibrium_point
    L_xx, L_yy, L_xy, L_yx, mu_x, mu_y = params

    LR_x = 1.0 / ( 4*(L_xx))
    LR_y = 1.0 / ( 4*(L_yy))

    loss_hist = np.zeros((N_iter+1, 1))
    lyap_hist = np.zeros((N_iter+1, 1))
    primd_gap = np.zeros((N_iter+1, 1))
    grad_norm = np.zeros((N_iter+1, 1))
    primd_gap[0] = compute_primal_dual_gap(func_mat, x, y)
    lyap_hist[0] = compute_lyapunov(func_mat, params, x, y, x, y)
    loss_hist[0] = (np.linalg.norm(x0-x_nash, ord=2)**2) + (np.linalg.norm(y0-y_nash, ord=2)**2)
    g_x, g_y = grad(func_mat, x, y)
    grad_norm[0] = (np.linalg.norm(g_x, ord=2)**2) + (np.linalg.norm(g_y, ord=2)**2)
    for i in range(N_iter):
        g_x, g_y = grad(func_mat, x, y)

        x = x - LR_x*g_x
        y = y + LR_y*g_y
        
        g_x, g_y = grad(func_mat, x, y)
        x = x - LR_x*g_x
        y = y + LR_y*g_y
        
        loss_hist[i+1]=(np.linalg.norm(x-x_nash, ord=2)**2) + (np.linalg.norm(y-y_nash, ord=2)**2)
        grad_norm[i+1] = (np.linalg.norm(g_x, ord=2)**2) + (np.linalg.norm(g_y, ord=2)**2)
        lyap_hist[i+1] = compute_lyapunov(func_mat, params, x, y, x, y)
        primd_gap[i+1] = compute_primal_dual_gap(func_mat, x, y)
        
        if loss_hist[i+1] <= tol:
            break
    stats = [loss_hist, grad_norm, lyap_hist, primd_gap]
    return stats
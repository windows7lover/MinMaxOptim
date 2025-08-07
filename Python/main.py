import numpy as np


from objectives.quadratic import func_matrices, compute_nash_equilibrium
from algorithms.simGDA import simultaneous_gradient
from algorithms.altGDA import alternate_gradient
from algorithms.extraGD import extra_gradient


def plot(stats_gda, stats_alt_gda, stats_extra_gda):
    import matplotlib.pyplot as plt
    import pylab
    pylab.rcParams['figure.figsize'] = (6.5, 5)
    plt.rcParams.update({'font.size': 22})
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    [loss_hist_gda, grad_norm_gda, lyap_gda, primd_gda] = stats_gda
    [loss_hist_alt_gda, grad_norm_alt_gda, lyap_alt_gda, primd_alt_gda] = stats_alt_gda
    [loss_hist_extra_gda, grad_norm_extra_gda, lyap_extra_gda, primd_extra_gda] = stats_extra_gda
    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138),
             (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141),
             (23, 190, 207), (158, 218, 229)] 
    tableau20 = [ (c[0]/255., c[1]/255., c[2]/255.) for c in tableau20]

    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    fig3 = plt.figure(3)
    ax3 = fig3.gca()
    fig4 = plt.figure(4)
    ax4 = fig4.gca()
    fig5 = plt.figure(5)
    ax5 = fig5.gca()

    ax1.set(xlim=(0, 5000))
    ax2.set(xlim=(0, 5000))
    ax3.set(xlim=(0, 5000))
    ax4.set(xlim=(0, 5000))
    ax5.set(xlim=(0, 5000))
    

    ax1.semilogy(loss_hist_gda, c=tableau20[0], label='SimGDA')
    ax2.semilogy(loss_hist_gda, c=tableau20[0], label='SimGDA')
    ax3.semilogy(grad_norm_gda, c=tableau20[0], label='SimGDA')
    ax4.semilogy(grad_norm_gda, c=tableau20[0], label='SimGDA')
    ax5.semilogy(lyap_gda, c=tableau20[0], label='SimGDA')

    ax1.semilogy(loss_hist_alt_gda, c=tableau20[2], label='AltGDA')
    ax2.semilogy(loss_hist_alt_gda, c=tableau20[2], label='AltGDA')
    ax3.semilogy(grad_norm_alt_gda, c=tableau20[2], label='AltGDA')
    ax4.semilogy(grad_norm_alt_gda, c=tableau20[2], label='AltGDA')
    ax5.semilogy(lyap_alt_gda, c=tableau20[2], label='AltGDA')

    ax1.semilogy(loss_hist_extra_gda, c=tableau20[4], label='ExtraGD')
    ax2.semilogy(loss_hist_extra_gda, c=tableau20[4], label='ExtraGD')
    ax3.semilogy(grad_norm_extra_gda, c=tableau20[4], label='ExtraGD')
    ax4.semilogy(grad_norm_extra_gda, c=tableau20[4], label='ExtraGD')
    ax5.semilogy(lyap_extra_gda, c=tableau20[4], label='ExtraGD')

    ax1.set_title("Distance to equilibrium of x,y (l2 norm squared)")
    ax2.set_title("Distance to equilibrium of x_st,y_st (l2 norm squared)")
    ax3.set_title("Gradient norm squared of x,y (l2 norm)")
    ax4.set_title("Gradient norm squared of x_st,y_st (l2 norm)")
    ax5.set_title("Lyapunov function values")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    
    fig1.savefig('Python/figures/result1.png')
    fig2.savefig('Python/figures/result2.png')
    fig3.savefig('Python/figures/result3.png')
    fig4.savefig('Python/figures/result4.png')
    fig5.savefig('Python/figures/result5.png')

def main(is_plot=True):
    # Set parameters
    L_xx  = 50
    L_yy  = 50
    L_xy  = L_yx = 50
    mu_x  = 0.1
    mu_y  = 0.1
    dim_x = 100
    dim_y = 70
    N_iter = 5000

    params  = [L_xx, L_yy, L_xy, L_yx, mu_x, mu_y]

    x0, y0 = np.ones((dim_x,1)), np.ones((dim_y,1)) 

    np.random.seed(10)

    f_mat = func_matrices(dim_x, dim_y, params)
    x_nash, y_nash = compute_nash_equilibrium(f_mat)

    initial_point = (x0, y0)
    equilibrium_point = (x_nash, y_nash)

    stats_gda = simultaneous_gradient(N_iter, initial_point, equilibrium_point, params, f_mat)
    stats_alt_gda = alternate_gradient(N_iter, initial_point, equilibrium_point, params, f_mat)
    stats_extra_gda = extra_gradient(N_iter, initial_point, equilibrium_point, params, f_mat)

    if is_plot:
        plot(stats_gda, stats_alt_gda, stats_extra_gda)

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt

def plot_temporal_convergence_rate():
    # dt_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    dt_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    errors = []
    nu = 0.01
    solver_type = 'explicit'
    nx = 40
    dt_hifi = 0.001

    sol_hifi = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{nx}_dt_{dt_hifi}.txt')[:, -1]  # high-fidelity solution

    for dt in dt_values:
        sol = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{nx}_dt_{dt}.txt')[:, -1]  # load solution at final time step
        sol_diff = sol - sol_hifi
        error = np.sqrt(np.mean(sol_diff**2))
        errors.append(error)

    # compute the convergence rate
    p, _ = np.polyfit(np.log(dt_values), np.log(errors), 1)
    print(f"Rate of convergence in time at nu = {nu}, p = {p}")
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dt_values, errors, 'o-', label='Error')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    plt.title(f'Temporal convergence at nu = {nu}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def plot_spatial_convergence_rate():
    nx_values = [10, 20, 30, 40, 60]
    dx_values = [2 * np.pi / nx for nx in nx_values]
    errors = np.array([])
    
    nu = 1
    nx_hifi = 120
    solver_type = 'implicit'
    dt = 0.0001
    
    x = np.linspace(0, 2 * np.pi, nx_hifi, endpoint=False)  # spatial grid

    sol_hifi = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{nx_hifi}_dt_{dt}.txt')[:, -1]  # high-fidelity solution
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, sol_hifi, label='High-fidelity solution (nx=120)')
    plt.scatter(x[nx_hifi // 5], sol_hifi[nx_hifi // 5], color='red', label='Sampled points (nx=120)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'High-fidelity solution at nu = {nu}')
    plt.grid()

    for nx in nx_values:
        x = np.linspace(0, 2 * np.pi, nx, endpoint=False)  # spatial grid
        sol = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{nx}_dt_{dt}.txt')[:, -1]  # load solution at final time step
        plt.plot(x, sol, label=f'Coarse solution (nx={nx})')
        # sol_hifi_sample = sol_hifi[::120//nx]  # sample high-fidelity solution to match coarse grid
        # sol_diff = sol - sol_hifi_sample
        # error = np.sqrt(np.mean(sol_diff**2))
        sol_sample = sol[nx//5]  # sample point at x = 2*pi/5
        print(f"nx = {nx}, sampled solution = {sol_sample}")
        sol_hifi_sample = sol_hifi[nx_hifi//5]  # corresponding high-fidelity sample point
        error = np.abs(sol_sample - sol_hifi_sample)
        errors = np.append(errors, error)
        
    plt.legend()
    plt.show()
        
    # compute the convergence rate
    p, _ = np.polyfit(np.log(dx_values), np.log(errors), 1)
    print(f"Spatial convergence at nu = {nu}", p)

    plt.figure(figsize=(8, 5))
    plt.loglog(dx_values, errors, 'o-', label='Error')
    plt.xlabel('Spatial step size (dx)')
    plt.ylabel('Error')
    plt.title(f'Spatial convergence at nu = {nu}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def plot_solution_with_time():
    L = 2 * np.pi  # domain length
    Nx = 40     # number of spatial points
    dx = L / Nx   # spatial step size
    nu = 0.01
    x = np.linspace(0, L, Nx, endpoint=False)  # spatial grid
    solver_type = 'explicit'
    
    Lt = 18
    dt = 0.9
    dt_sample = dt * 3
    Ns = 1 + int(Lt / dt_sample)  # number of samples

    t = np.linspace(0, Lt, Ns, endpoint=True)  # time grid

    sol = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{Nx}_dt_{dt}.txt')  # load the solution from the text file
    plt.figure(figsize=(10, 6))
    plt.title(f"Solution of Burgers' equation using {solver_type} scheme, nu = {nu}, dt = {dt}, nx = {Nx}")
    plt.xlabel("x")
    plt.ylabel("u")
    # plt.ylim(-1.5, 1.5)
    for i in range(Ns):
        plt.plot(x, sol[:, i * int(dt_sample / dt)], label=f't={t[i]:.3f}s')
    plt.legend(loc = 'upper right')
    plt.grid()
    plt.show()
    
def plot_primary_and_secondary_conservation():
    L = 2 * np.pi  # domain length
    Nx = 40     # number of spatial points
    dx = L / Nx   # spatial step size
    nu = 0
    x = np.linspace(0, L, Nx, endpoint=False)  # spatial grid
    solver_type = 'implicit'
    
    Lt = 4
    dt = 0.01
    dt_sample = dt * 100
    Ns = 1 + int(Lt / dt_sample)  # number of samples

    t = np.linspace(0, Lt, Ns, endpoint=True)  # time grid

    sol = np.loadtxt(f'burgers_{solver_type}_nu_{nu}_nx_{Nx}_dt_{dt}.txt')  # load the solution from the text file
    primary = np.zeros(Ns)
    secondary = np.zeros(Ns)
    
    for i in range(Ns):
        primary[i] = np.sum(sol[:, i * int(dt_sample / dt)]) * dx
        secondary[i] = 0.5 * np.sum(sol[:, i * int(dt_sample / dt)]**2) * dx

    plt.figure(figsize=(10, 6))
    plt.plot(t, primary, label='Primary quantity')
    # plt.plot(t, secondary, label='Secondary quantity')
    plt.xlabel('Time (s)')
    plt.ylabel('Quantities')
    plt.title(f'Conservation Laws at nu = {nu}, dt = {dt}, nx = {Nx} via {solver_type} scheme')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    
    # plot_solution_with_time()

    # plot_temporal_convergence_rate()
    
    plot_spatial_convergence_rate()
    
    # plot_primary_and_secondary_conservation()


if __name__ == "__main__":
    main()
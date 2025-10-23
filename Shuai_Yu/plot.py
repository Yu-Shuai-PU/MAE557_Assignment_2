import numpy as np
import matplotlib.pyplot as plt

def plot_temporal_convergence_rate():
    # dt_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    dt_values = [2e-05, 1e-05, 5e-06]
    dt_hifi = 1e-06
    errors = []
    Nx = 33
    sample_interval = 1000
    tfinal = 0.2

    output_files_dir = f'output_Ma_0.025_Nx_33_dt_{dt_hifi}/'
    idx_tfinal_hifi = int(tfinal / (sample_interval * dt_hifi))

    rho_hifi_tfinal = np.loadtxt(f'{output_files_dir}rho/rho_{idx_tfinal_hifi}.txt')
    u1_hifi_tfinal = np.loadtxt(f'{output_files_dir}u1/u1_{idx_tfinal_hifi}.txt')
    u2_hifi_tfinal = np.loadtxt(f'{output_files_dir}u2/u2_{idx_tfinal_hifi}.txt')
    T_hifi_tfinal = np.loadtxt(f'{output_files_dir}T/T_{idx_tfinal_hifi}.txt')

    # high-fidelity solution at final time step

    for dt in dt_values:
        output_files_dir = f'output_Ma_0.025_Nx_33_dt_{dt}/'
        idx_tfinal = int(tfinal / (sample_interval * dt))

        rho_coarse_tfinal = np.loadtxt(f'{output_files_dir}rho/rho_{idx_tfinal}.txt')
        u1_coarse_tfinal = np.loadtxt(f'{output_files_dir}u1/u1_{idx_tfinal}.txt')
        u2_coarse_tfinal = np.loadtxt(f'{output_files_dir}u2/u2_{idx_tfinal}.txt')
        T_coarse_tfinal = np.loadtxt(f'{output_files_dir}T/T_{idx_tfinal}.txt')

        # compute L2 error at final time step
        error_rho = np.sqrt(np.mean((rho_coarse_tfinal - rho_hifi_tfinal)**2))
        error_u1 = np.sqrt(np.mean((u1_coarse_tfinal - u1_hifi_tfinal)**2))
        error_u2 = np.sqrt(np.mean((u2_coarse_tfinal - u2_hifi_tfinal)**2))
        error_T = np.sqrt(np.mean((T_coarse_tfinal - T_hifi_tfinal)**2))

        total_error = np.sqrt(error_rho**2 + error_u1**2 + error_u2**2 + error_T**2)
        errors.append(total_error)
        print(f"dt = {dt}, idx_tfinal = {idx_tfinal}, L2 Error at tfinal = {total_error}")

    # compute the convergence rate
    p, _ = np.polyfit(np.log(dt_values), np.log(errors), 1)
    print(f"Rate of convergence in time at Nx = {Nx}, p = {p}")
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dt_values, errors, 'o-', label='Error')
    plt.xlabel('Time step size (dt)')
    plt.ylabel('L2 Error')
    plt.title(f'Temporal convergence at Nx = {Nx}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def plot_spatial_convergence_rate():
    # dt_values = [1, 0.5, 0.2, 0.1, 0.05, 0.02]
    nx_values = [9, 17, 33]
    nx_hifi = 65
    dx_values = 1.0 / (np.array(nx_values) - 1)
    
    dt = 1e-05
    sample_interval = 1000
    dt_sample = dt * sample_interval
    tfinal = 1
    
    coordinate_numer = 15
    coordinate_denom = 16

    output_files_dir = f'output_Ma_0.025_Nx_{nx_hifi}_dt_{dt}/'
    idx_tfinal = int(tfinal / (sample_interval * dt))
    
    rho_hifi_tfinal = np.loadtxt(f'{output_files_dir}rho/rho_{idx_tfinal}.txt')
    u1_hifi_tfinal = np.loadtxt(f'{output_files_dir}u1/u1_{idx_tfinal}.txt')
    u2_hifi_tfinal = np.loadtxt(f'{output_files_dir}u2/u2_{idx_tfinal}.txt')
    T_hifi_tfinal = np.loadtxt(f'{output_files_dir}T/T_{idx_tfinal}.txt')
    rho_hifi_tfinal_sample_point = rho_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]
    u1_hifi_tfinal_sample_point = u1_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]
    u2_hifi_tfinal_sample_point = u2_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]
    T_hifi_tfinal_sample_point = T_hifi_tfinal[(coordinate_numer * nx_hifi//coordinate_denom), (coordinate_numer * nx_hifi//coordinate_denom)]

    # high-fidelity solution at final time step
    errors = []
    
    for nx in nx_values:
        output_files_dir = f'output_Ma_0.025_Nx_{nx}_dt_{dt}/'
        
        rho_tfinal = np.loadtxt(f'{output_files_dir}rho/rho_{idx_tfinal}.txt')
        u1_tfinal = np.loadtxt(f'{output_files_dir}u1/u1_{idx_tfinal}.txt')
        u2_tfinal = np.loadtxt(f'{output_files_dir}u2/u2_{idx_tfinal}.txt')
        T_tfinal = np.loadtxt(f'{output_files_dir}T/T_{idx_tfinal}.txt')
        rho_tfinal_sample_point = rho_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]
        u1_tfinal_sample_point = u1_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]
        u2_tfinal_sample_point = u2_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]
        T_tfinal_sample_point = T_tfinal[(coordinate_numer * nx//coordinate_denom), (coordinate_numer * nx//coordinate_denom)]

        error_rho = rho_tfinal_sample_point - rho_hifi_tfinal_sample_point
        error_u1 = u1_tfinal_sample_point - u1_hifi_tfinal_sample_point
        error_u2 = u2_tfinal_sample_point - u2_hifi_tfinal_sample_point
        error_T = T_tfinal_sample_point - T_hifi_tfinal_sample_point
        total_error = np.sqrt(error_rho**2 + error_u1**2 + error_u2**2 + error_T**2)
        errors.append(total_error)
        print(f"Nx = {nx}, dx = {dx_values[nx_values.index(nx)]}, L2 Error at t = {tfinal}, (x, y) = (15/16, 15/16) = {total_error}")
        
    # compute the convergence rate
    p, _ = np.polyfit(np.log(dx_values), np.log(errors), 1)
    print(f"Rate of convergence in space at dt = {dt}, p = {p}")
    
    plt.figure(figsize=(8, 5))
    plt.loglog(dx_values, errors, 'o-', label='Error')
    plt.xlabel('Spatial grid size (dx)')
    plt.ylabel('L2 Error')
    plt.title(f'Spatial convergence at dt = {dt}, rate = {p}')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    
    # plot_solution_with_time()

    # plot_temporal_convergence_rate()
    
    # plot_temporal_convergence_rate_Richardson()
    
    plot_spatial_convergence_rate()
        
    # plot_primary_and_secondary_conservation()


if __name__ == "__main__":
    main()
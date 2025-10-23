#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <filesystem>

class Parameters {
public:
    // Physical constants
    const double L = 1.0;
    const double R = 287.0;
    const double gamma = 1.4;

    double Re, Ma, Pr, T_init, a_init, U_wall, dt, dt_sample, dx1, dx2, relaxation_factor, t_final;
    int Nx1, Nx2, Nt, sample_interval;

    Parameters(double Re, double Ma, double Pr, double T_init,
               int Nx1, int Nx2, double t_final, double relaxation_factor, double dt, int sample_interval)
        : Re(Re), Ma(Ma), Pr(Pr), T_init(T_init),
          Nx1(Nx1), Nx2(Nx2), t_final(t_final),
          relaxation_factor(relaxation_factor), dt(dt), sample_interval(sample_interval)
    {
        a_init = std::sqrt(gamma * R * T_init);
        U_wall = Ma * a_init;

        dx1 = L / (Nx1 - 1);
        dx2 = L / (Nx2 - 1);
        dt_sample = dt * sample_interval;
        Nt = 1 + static_cast<int>(std::ceil(t_final / dt));
    }
};

using StateVector = std::vector<Eigen::MatrixXd>; // U = [rho, rho*u, rho*v, rho*e]

StateVector compute_rhs(const StateVector& U, double t, const Parameters& params);
void update_BCs(StateVector& U, const Parameters& params, double t_current);
void save_matrix(const Eigen::MatrixXd& matrix, const std::string& filename);

int main(int argc, char* argv[]) {
    // Check the number of inputs

    if (argc != 11) {
        std::cerr << "Error: get " << argc - 1 << " input arguments, expect 11." << std::endl;
        std::cerr << "Usage: ./solver_explicit Re Ma Pr gamma T_init Nx1 Nx2 t_final relax_factor dt sample_interval" << std::endl;
        return 1; // Error exit
    }

    try {
        double Re                = std::stod(argv[1]);
        double Ma                = std::stod(argv[2]);
        double Pr                = std::stod(argv[3]);
        double T_init            = std::stod(argv[4]);
        int    Nx1               = std::stoi(argv[5]);
        int    Nx2               = std::stoi(argv[6]);
        double t_final           = std::stod(argv[7]);
        double relaxation_factor = std::stod(argv[8]);
        double dt                = std::stod(argv[9]);
        int    sample_interval   = std::stoi(argv[10]);

        std::ostringstream output_folder_name;
        output_folder_name << "output_Ma_" << Ma << "_Nx_" << Nx1 << "_dt_" << dt;
        std::filesystem::path output_folder_path = output_folder_name.str();

        // Create the Parameters object
        Parameters params(Re, Ma, Pr, T_init, Nx1, Nx2, t_final, relaxation_factor, dt, sample_interval);

        // Set up constants
        const double pi = M_PI;
        const double CFL_limit = params.relaxation_factor * 1.0;
        const double diff_num_limit = params.relaxation_factor * 2.0 / 4.0;
        // Set up the 4-element state vector and vectors for quantities and intermediate state vectors for RK4
        StateVector U(4), U_tmp(4);
        StateVector K0(4), K1(4), K2(4), K3(4);
        StateVector dUdt0(4), dUdt1(4), dUdt2(4), dUdt3(4);
        Eigen::MatrixXd rho(params.Nx2, params.Nx1);
        Eigen::MatrixXd u1(params.Nx2, params.Nx1);
        Eigen::MatrixXd u2(params.Nx2, params.Nx1);
        Eigen::MatrixXd T(params.Nx2, params.Nx1);
        Eigen::MatrixXd a(params.Nx2, params.Nx1);

        // Step 1: Initialize the NSE state vector
        for(int i = 0; i < 4; ++i) U[i] = Eigen::MatrixXd(params.Nx2, params.Nx1);
        // Initial conditions (Eq. 26-28)
        U[0] = Eigen::MatrixXd::Ones(params.Nx2, params.Nx1); // rho
        U[1] = Eigen::MatrixXd::Zero(params.Nx2, params.Nx1); // rho*u
        U[2] = Eigen::MatrixXd::Zero(params.Nx2, params.Nx1); // rho*v
        U[3] = Eigen::MatrixXd::Ones(params.Nx2, params.Nx1); // rho*e = rho*T (dimensionless e = dimensionless T)

        double CFL = 0.0;
        double viscous_diff_num = 0.0;
        double thermal_diff_num = 0.0;
        int counter = 0;

        // Create output directories if they do not exist
        std::filesystem::create_directories(output_folder_path);
        std::filesystem::create_directories(output_folder_path / "rho");
        std::filesystem::create_directories(output_folder_path / "u1");
        std::filesystem::create_directories(output_folder_path / "u2");
        std::filesystem::create_directories(output_folder_path / "T");

        for (int k = 0; k < params.Nt; ++k) {
            double t_current = k * params.dt;
        
            // Check the CFL limit
            u1 = U[1].array() / U[0].array();
            u2 = U[2].array() / U[0].array();
            T  = U[3].array() / U[0].array();
            
            double u1_max_abs = u1.array().abs().maxCoeff();
            double u2_max_abs = u2.array().abs().maxCoeff();
            double T_max = T.array().maxCoeff();

            CFL = (u1_max_abs + std::sqrt(T_max) / params.Ma) * params.dt / params.dx1
                + (u2_max_abs + std::sqrt(T_max) / params.Ma) * params.dt / params.dx2;
                
            viscous_diff_num = (1.0/params.Re) * (params.dt / (params.dx1 * params.dx1) + params.dt / (params.dx2 * params.dx2));
            thermal_diff_num = (params.gamma/(params.Pr * params.Re)) * (params.dt / (params.dx1 * params.dx1) + params.dt / (params.dx2 * params.dx2));

            if (k % params.sample_interval == 0 || k == params.Nt - 1) {
            
                if (CFL > CFL_limit) {
                std::cerr << "Warning: CFL condition violated at k = " << k << "!" << std::endl;
                std::cerr << "  CFL # = " << CFL << ", Limit = " << CFL_limit << std::endl;
                std::cerr << "  Contributing Maxima:" << std::endl;
                std::cerr << "    |u1|_max = " << u1_max_abs << std::endl;
                std::cerr << "    |u2|_max = " << u2_max_abs << std::endl;
                std::cerr << "    T_max = " << T_max << std::endl;
                // return 1; // abort the program
            }
                if (viscous_diff_num > diff_num_limit) {
                std::cerr << "Warning: viscous diff # condition violated at k = " << k << "!" << std::endl;
                std::cerr << "  diff # = " << viscous_diff_num << ", Limit = " << diff_num_limit << std::endl;
                std::cerr << "  Contributing Maxima:" << std::endl;
                std::cerr << "    |u1|_max = " << u1_max_abs << std::endl;
                std::cerr << "    |u2|_max = " << u2_max_abs << std::endl;
                std::cerr << "    T_max = " << T_max << std::endl;
                // return 1; // abort the program
            }
                if (thermal_diff_num > diff_num_limit) {
                std::cerr << "Warning: thermal diff # condition violated at k = " << k << "!" << std::endl;
                std::cerr << "  diff # = " << thermal_diff_num << ", Limit = " << diff_num_limit << std::endl;
                std::cerr << "  Contributing Maxima:" << std::endl;
                std::cerr << "    |u1|_max = " << u1_max_abs << std::endl;
                std::cerr << "    |u2|_max = " << u2_max_abs << std::endl;
                std::cerr << "    T_max = " << T_max << std::endl;
                // return 1; // abort the program
            }
            
                if (k == 0) {
                std::cout << "--- Simulation Starting ---" << std::endl;
                std::cout << "  Ma = " << params.Ma << ", Re = " << params.Re << std::endl;
                std::cout << "  Grid = " << params.Nx1 << "x" << params.Nx2 << std::endl;
                std::cout << "  dt = " << params.dt << ", t_final = " << params.t_final << " (Nt = " << params.Nt << ")" << std::endl;
                std::cout << "  Initial CFL # = " << CFL << " (Limit = " << CFL_limit << ")" << std::endl;
                std::cout << "  Initial viscous diff # = " << viscous_diff_num << " (Limit = " << diff_num_limit << ")" << std::endl;
                std::cout << "  Initial thermal diff # = " << thermal_diff_num << " (Limit = " << diff_num_limit << ")" << std::endl;
                std::cout << "  Mean density = " << U[0].mean() << std::endl;
                std::cout << "---------------------------" << std::endl;
            }
                
                // 2. Create filenames with zero-padded numbers

                std::cout << "Saving snapshot at t* = " << t_current
                << " (timestep k = " << k
                << ", CFL # = " << CFL
                << ", mean density = " << U[0].mean()
                << ")" << std::endl;

                std::string filename_rho = (output_folder_path / "rho" / ("rho_" + std::to_string(counter) + ".txt")).string();
                std::string filename_u1  = (output_folder_path / "u1" / ("u1_" + std::to_string(counter) + ".txt")).string();
                std::string filename_u2  = (output_folder_path / "u2" / ("u2_" + std::to_string(counter) + ".txt")).string();
                std::string filename_T   = (output_folder_path / "T" / ("T_" + std::to_string(counter) + ".txt")).string();
                // 3. Call your save function for each variable
                //    (Make sure you have created an "output" directory first!)
                save_matrix(U[0], filename_rho);
                save_matrix(u1, filename_u1);
                save_matrix(u2, filename_u2);
                save_matrix(T, filename_T);

                // 4. Increment the counter for the next snapshot
                counter++;
            }

            // --- Forward Euler Scheme ---
            dUdt0 = compute_rhs(U, t_current, params);
            for (int i=0; i<4; ++i) {
                U[i] = U[i] + params.dt * dUdt0[i];
            }
            update_BCs(U, params, t_current + params.dt); // Update density BCs for U_tmp[0] = rho
        
            // Check for simulation divergence
            if (!U[0].allFinite() || !U[1].allFinite() || !U[2].allFinite() || !U[3].allFinite()) {
                std::cerr << "Error: Simulation diverged at timestep k = " << k << " (t* = " << t_current << ")" << std::endl;
                return 1;
            }

            
        }

        std::cout << "Simulation finished successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error when extracting the input arguments: " << e.what() << std::endl;
        std::cerr << "Please check the .sh file, not all arguments are valid numbers with compatible types." << std::endl;
        return 1;
    }
}

// compute_rhs: The function to compute the right-hand side of the NSE
StateVector compute_rhs(const StateVector& U, double t, const Parameters& params) {

    // Step 0: Initialization and variable extraction
    StateVector dUdt(4);
    for(int i = 0; i < 4; ++i) dUdt[i] = Eigen::MatrixXd::Zero(params.Nx2, params.Nx1);

    // Extract primitive variables, creating copies that we can modify
    Eigen::MatrixXd rho = U[0]; // (Nx2, Nx1)
    Eigen::MatrixXd u1 = U[1].array() / rho.array();
    Eigen::MatrixXd u2 = U[2].array() / rho.array();
    Eigen::MatrixXd e = U[3].array() / rho.array();
    Eigen::MatrixXd T = e; // Since e* = T*

    Eigen::MatrixXd rhou1 = rho.array() * u1.array();
    Eigen::MatrixXd rhou2 = rho.array() * u2.array();
    Eigen::MatrixXd rhoe = rho.array() * T.array();
    Eigen::MatrixXd p = (1.0 / (params.gamma * params.Ma * params.Ma)) * rho.array() * T.array(); // p* = 1/(gamma*Ma^2) * rho* * T*

    // Step 1: Compute RHS for interior points using central differences
    for (int i = 1; i < params.Nx1 - 1; ++i) { // x1 direction
        for (int j = 1; j < params.Nx2 - 1; ++j) { // x2 direction
            // --- First derivatives (using central differences, Eq. 36-37) ---
            double drhou1_dx1 = (rhou1(j, i+1) - rhou1(j, i-1)) / (2.0 * params.dx1);
            double drhou2_dx2 = (rhou2(j+1, i) - rhou2(j-1, i)) / (2.0 * params.dx2);

            // --- Continuity Equation RHS (Eq. 31) ---
            dUdt[0](j, i) = -(drhou1_dx1 + drhou2_dx2);
            
            double drhou1u1_dx1 = (rhou1(j,i+1)*u1(j,i+1) - rhou1(j,i-1)*u1(j,i-1)) / (2.0 * params.dx1);
            double drhou2u1_dx2 = (rhou2(j+1,i)*u1(j+1,i) - rhou2(j-1,i)*u1(j-1,i)) / (2.0 * params.dx2);

            double drhou1u2_dx1 = (rhou1(j,i+1)*u2(j,i+1) - rhou1(j,i-1)*u2(j,i-1)) / (2.0 * params.dx1);
            double drhou2u2_dx2 = (rhou2(j+1,i)*u2(j+1,i) - rhou2(j-1,i)*u2(j-1,i)) / (2.0 * params.dx2);
            
            double drhou1e_dx1 = (rhou1(j,i+1)*e(j,i+1) - rhou1(j,i-1)*e(j,i-1)) / (2.0 * params.dx1);
            double drhou2e_dx2 = (rhou2(j+1,i)*e(j+1,i) - rhou2(j-1,i)*e(j-1,i)) / (2.0 * params.dx2);

            double rho_r = 0.5 * (rho(j,i) + rho(j,i+1)); // _r means at i+1/2
            double du1_dx1_r = (u1(j,i+1) - u1(j,i)) / params.dx1;
            double du2_dx2 = (u2(j+1,i) - u2(j-1,i)) / (2.0 * params.dx2);
            double du2_dx2_rr = (u2(j+1,i+1) - u2(j-1,i+1)) / (2.0 * params.dx2); // _rr means at i+1
            double du2_dx2_r = (du2_dx2 + du2_dx2_rr) / 2.0;
            double tau1_r = (1.0/params.Re) * rho_r * (du1_dx1_r + du1_dx1_r - (2.0/3.0) * (du1_dx1_r + du2_dx2_r));
            double rho_l = 0.5 * (rho(j,i) + rho(j,i-1)); // _l means at i-1/2
            double du1_dx1_l = (u1(j,i) - u1(j,i-1)) / params.dx1;
            double du2_dx2_ll = (u2(j+1,i-1) - u2(j-1,i-1)) / (2.0 * params.dx2); // _ll means at i-1
            double du2_dx2_l = (du2_dx2 + du2_dx2_ll) / 2.0;
            double tau1_l = (1.0/params.Re) * rho_l * (du1_dx1_l + du1_dx1_l - (2.0/3.0) * (du1_dx1_l + du2_dx2_l));
            double dtau1_dx1 = (tau1_r - tau1_l) / params.dx1;

            double rho_u = 0.5 * (rho(j,i) + rho(j+1,i)); // _u means at j+1/2
            double du1_dx2_u = (u1(j+1,i) - u1(j,i)) / params.dx2;
            double du2_dx1   = (u2(j,i+1) - u2(j,i-1)) / (2.0 * params.dx1);
            double du2_dx1_uu = (u2(j+1,i+1) - u2(j+1,i-1)) / (2.0 * params.dx1); // _uu means at j+1
            double du2_dx1_u = (du2_dx1 + du2_dx1_uu) / 2.0;
            double tau1_u = (1.0/params.Re) * rho_u * (du1_dx2_u + du2_dx1_u);
            double rho_d = 0.5 * (rho(j,i) + rho(j-1,i)); // _d means at j-1/2
            double du1_dx2_d = (u1(j,i) - u1(j-1,i)) / params.dx2;
            double du2_dx1_dd = (u2(j-1,i+1) - u2(j-1,i-1)) / (2.0 * params.dx1); // _dd means at j-1
            double du2_dx1_d = (du2_dx1 + du2_dx1_dd) / 2.0;
            double tau1_d = (1.0/params.Re) * rho_d * (du1_dx2_d + du2_dx1_d);
            double dtau1_dx2 = (tau1_u - tau1_d) / params.dx2;

            double dp_dx1 = (p(j, i+1) - p(j, i-1)) / (2.0 * params.dx1);
            dUdt[1](j,i) = -(drhou1u1_dx1 + drhou2u1_dx2) - dp_dx1 + dtau1_dx1 + dtau1_dx2;

            double du2_dx1_r = (u2(j,i+1) - u2(j,i)) / params.dx1;
            double du1_dx2 = (u1(j+1,i) - u1(j-1,i)) / (2.0 * params.dx2);
            double du1_dx2_rr = (u1(j+1,i+1) - u1(j-1,i+1)) / (2.0 * params.dx2);
            double du1_dx2_r = (du1_dx2 + du1_dx2_rr) / 2.0;
            double tau2_r = (1.0/params.Re) * rho_r * (du2_dx1_r + du1_dx2_r);
            double du2_dx1_l = (u2(j,i) - u2(j,i-1)) / params.dx1;
            double du1_dx2_ll = (u1(j+1,i-1) - u1(j-1,i-1)) / (2.0 * params.dx2);
            double du1_dx2_l = (du1_dx2 + du1_dx2_ll) / 2.0;
            double tau2_l = (1.0/params.Re) * rho_l * (du2_dx1_l + du1_dx2_l);
            double dtau2_dx1 = (tau2_r - tau2_l) / params.dx1;

            double du2_dx2_u = (u2(j+1,i) - u2(j,i)) / params.dx2;
            double du1_dx1   = (u1(j,i+1) - u1(j,i-1)) / (2.0 * params.dx1);
            double du1_dx1_uu = (u1(j+1,i+1) - u1(j+1,i-1)) / (2.0 * params.dx1);
            double du1_dx1_u = (du1_dx1 + du1_dx1_uu) / 2.0;
            double tau2_u = (1.0/params.Re) * rho_u * (du2_dx2_u + du2_dx2_u - (2.0/3.0) * (du1_dx1_u + du2_dx2_u));
            double du2_dx2_d = (u2(j,i) - u2(j-1,i)) / params.dx2;
            double du1_dx1_dd = (u1(j-1,i+1) - u1(j-1,i-1)) / (2.0 * params.dx1);
            double du1_dx1_d = (du1_dx1 + du1_dx1_dd) / 2.0;
            double tau2_d = (1.0/params.Re) * rho_d * (du2_dx2_d + du2_dx2_d - (2.0/3.0) * (du1_dx1_d + du2_dx2_d));
            double dtau2_dx2 = (tau2_u - tau2_d) / params.dx2;

            double dp_dx2 = (p(j+1, i) - p(j-1, i)) / (2.0 * params.dx2);
            dUdt[2](j,i) = -(drhou1u2_dx1 + drhou2u2_dx2) - dp_dx2 + dtau2_dx1 + dtau2_dx2;

            double div_u = du1_dx1 + du2_dx2;

            double dT_dx1_r = (T(j,i+1) - T(j,i)) / params.dx1;
            double dT_dx1_l = (T(j,i) - T(j,i-1)) / params.dx1;
            double dq_dx1 = (params.gamma / (params.Pr * params.Re)) * (rho_r * dT_dx1_r - rho_l * dT_dx1_l) / params.dx1;

            double dT_dx2_u = (T(j+1,i) - T(j,i)) / params.dx2;
            double dT_dx2_d = (T(j,i) - T(j-1,i)) / params.dx2;
            double dq_dx2 = (params.gamma / (params.Pr * params.Re)) * (rho_u * dT_dx2_u - rho_d * dT_dx2_d) / params.dx2;
            double pdiv_u = params.gamma * (params.gamma - 1.0) * params.Ma * params.Ma * p(j,i) * div_u;
            double dissip = (2.0 * params.Ma * params.Ma * params.gamma * (params.gamma - 1.0) / params.Re)
                           * rho(j,i) * (du1_dx1 * du1_dx1 + du2_dx2 * du2_dx2
                           + 0.5 * ((du1_dx2 + du2_dx1) * (du1_dx2 + du2_dx1))
                           - (1.0/3.0) * div_u * div_u);

            dUdt[3](j,i) = -(drhou1e_dx1 + drhou2e_dx2) - pdiv_u + dq_dx1 + dq_dx2 + dissip;
        }
    }

    return dUdt;
}
// update_BCs: The function to enforce the physical BCs and update the density BC
void update_BCs(StateVector& U, const Parameters& params, double t_current) {
    
    const int Nx1 = params.Nx1;
    const int Nx2 = params.Nx2;
    const double T_wall = 1.0;
    const double u1_top_wall = std::sin(2.0 * t_current / params.Re);

    double rho_wall;
    double rho_1, rho_2, T_1, T_2;

    // On the left wall, we use dp/dn = 0 to update rho
    for (int j = 1; j < Nx2 - 1; ++j) {
        rho_1 = U[0](j, 1);
        rho_2 = U[0](j, 2);
        T_1   = U[3](j, 1) / rho_1; // e = T
        T_2   = U[3](j, 2) / rho_2; // e = T

        if (std::abs(6.0 * T_wall - 4.0 * T_1 + T_2) < 1.0e-12) {
            rho_wall = rho_1; // safety check
        }
        else {
            rho_wall = T_wall * (4.0 * rho_1 - rho_2) / (6.0 * T_wall - 4.0 * T_1 + T_2);
        }

        // Update other boundary conditions on the left wall
        U[0](j, 0) = rho_wall;         // rho
        U[1](j, 0) = 0.0;               // rhou1
        U[2](j, 0) = 0.0;               // rhou2
        U[3](j, 0) = rho_wall * T_wall; // rhoe = rho * T_wall
    }

    // On the right wall
    for (int j = 1; j < Nx2 - 1; ++j) {
        rho_1 = U[0](j, Nx1 - 2);
        rho_2 = U[0](j, Nx1 - 3);
        T_1   = U[3](j, Nx1 - 2) / rho_1; // e = T
        T_2   = U[3](j, Nx1 - 3) / rho_2; // e = T

        if (std::abs(6.0 * T_wall - 4.0 * T_1 + T_2) < 1.0e-12) {
            rho_wall = rho_1; // safety check
        }
        else {
            rho_wall = T_wall * (4.0 * rho_1 - rho_2) / (6.0 * T_wall - 4.0 * T_1 + T_2);
        }

        // Update other boundary conditions on the right wall
        U[0](j, Nx1 - 1) = rho_wall;         // rho
        U[1](j, Nx1 - 1) = 0.0;               // rhou1
        U[2](j, Nx1 - 1) = 0.0;               // rhou2
        U[3](j, Nx1 - 1) = rho_wall * T_wall; // rhoe = rho * T_wall
    }

    // On the bottom wall
    for (int i = 1; i < Nx1 - 1; ++i) {
        rho_1 = U[0](1, i);
        rho_2 = U[0](2, i);
        T_1   = U[3](1, i) / rho_1; // e = T
        T_2   = U[3](2, i) / rho_2; // e = T

        if (std::abs(6.0 * T_wall - 4.0 * T_1 + T_2) < 1.0e-12) {
            rho_wall = rho_1; // safety check
        }
        else {
            rho_wall = T_wall * (4.0 * rho_1 - rho_2) / (6.0 * T_wall - 4.0 * T_1 + T_2);
        }

        // Update other boundary conditions on the bottom wall
        U[0](0, i) = rho_wall;         // rho
        U[1](0, i) = 0.0;               // rhou1
        U[2](0, i) = 0.0;               // rhou2
        U[3](0, i) = rho_wall * T_wall; // rhoe = rho * T_wall
    }
    // On the top wall
    for (int i = 1; i < Nx1 - 1; ++i) {
        rho_1 = U[0](Nx2 - 2, i);
        rho_2 = U[0](Nx2 - 3, i);
        T_1   = U[3](Nx2 - 2, i) / rho_1; // e = T
        T_2   = U[3](Nx2 - 3, i) / rho_2; // e = T

        if (std::abs(6.0 * T_wall - 4.0 * T_1 + T_2) < 1.0e-12) {
            rho_wall = rho_1; // safety check
        }
        else {
            rho_wall = T_wall * (4.0 * rho_1 - rho_2) / (6.0 * T_wall - 4.0 * T_1 + T_2);
        }

        // Update other boundary conditions on the top wall
        U[0](Nx2 - 1, i) = rho_wall;               // rho
        U[1](Nx2 - 1, i) = rho_wall * u1_top_wall; // rhou1
        U[2](Nx2 - 1, i) = 0.0;                    // rhou2
        U[3](Nx2 - 1, i) = rho_wall * T_wall;      // rhoe = rho * T_wall
    }

}
// save_matrix: Function to save a matrix to a file
void save_matrix(const Eigen::MatrixXd& matrix, const std::string& filename) {
    const static Eigen::IOFormat TXTFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::setprecision(15);
        file << matrix.format(TXTFormat);
        file.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}
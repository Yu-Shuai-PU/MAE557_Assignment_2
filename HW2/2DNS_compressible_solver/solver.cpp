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
#include <filesystem> // **NEW**: Include the filesystem library

// Helper function to save a matrix to a file
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

class Parameters {
public:
    // Physical constants
    const double L = 1.0;
    const double R = 287.0;
    const double gamma = 1.4;

    double Re, Ma, Pr, T_init, a_init, U_wall, dt, dt_sample, dx1, dx2, relaxation_factor;
    int Nx1, Nx2, Nt, sample_interval, t_final;

    Parameters(double Re, double Ma, double Pr, double T_init,
               int Nx1, int Nx2, int t_final, double relaxation_factor, double dt, int sample_interval)
        : Re(Re), Ma(Ma), Pr(Pr), T_init(T_init),
          Nx1(Nx1), Nx2(Nx2), t_final(t_final),
          relaxation_factor(relaxation_factor), dt(dt), sample_interval(sample_interval)
    {
        a_init = std::sqrt(gamma * R * T_init);
        U_wall = Ma * a_init;

        dx1 = 1.0 / (Nx1 - 1);
        dx2 = 1.0 / (Nx2 - 1);
        dt_sample = dt * sample_interval;
        Nt = 1 + static_cast<int>(t_final / dt);
    }
};

using StateVector = std::vector<Eigen::MatrixXd>; // U = [rho, rho*u, rho*v, rho*e]

StateVector compute_rhs(const StateVector& U, double t, const Parameters& params);

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
        int    t_final           = std::stoi(argv[7]);
        double relaxation_factor = std::stod(argv[8]);
        double dt                = std::stod(argv[9]);
        int    sample_interval   = std::stoi(argv[10]);

        // 使用解析出的参数创建 Parameters 对象
        // 所有派生参数（dx, Nt 等）都在构造函数中自动计算好了
        Parameters params(Re, Ma, Pr, T_init, Nx1, Nx2, t_final, relaxation_factor, dt, sample_interval);

        // Set up constants
        const double pi = M_PI;
        const double CFL_limit = params.relaxation_factor * 2.0 * std::sqrt(2.0);
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
        int counter = 0;

        if (!std::filesystem::exists("output")) { // Check if "output" directory exists
            std::filesystem::create_directory("output"); // Create "output" directory if it doesn't exist
            std::filesystem::create_directory("output/rho"); // Create "output" directory if it doesn't exist
            std::filesystem::create_directory("output/u1"); // Create "output" directory if it doesn't exist
            std::filesystem::create_directory("output/u2"); // Create "output" directory if it doesn't exist
            std::filesystem::create_directory("output/T"); // Create "output" directory if it doesn't exist
        }

        for (int k = 0; k < params.Nt; ++k) {
            double t_current = k * params.dt;
        

            // Check the CFL limit
            u1 = U[1].array() / U[0].array();
            u2 = U[2].array() / U[0].array();
            T  = U[3].array() / U[0].array();
            // Enforce boundary conditions on primitive variables
            u1.col(0).setZero();
            u1.col(params.Nx1 - 1).setZero();
            u1.row(0).setZero();
            u1.row(params.Nx2 - 1).setConstant(std::sin(2.0 * t_current / params.Re));
            u2.col(0).setZero();
            u2.col(params.Nx1 - 1).setZero();
            u2.row(0).setZero();
            u2.row(params.Nx2 - 1).setZero();
            T.col(0).setOnes();
            T.col(params.Nx1 - 1).setOnes();
            T.row(0).setOnes();
            T.row(params.Nx2 - 1).setOnes();

            a = params.a_init * T.array().sqrt();

            CFL = (params.U_wall * u1.array().abs().maxCoeff() + a.array().maxCoeff()) * params.dt / params.dx1
                + (params.U_wall * u2.array().abs().maxCoeff() + a.array().maxCoeff()) * params.dt / params.dx2;

            if (CFL > CFL_limit) {
                std::cerr << "Warning: CFL condition violated (CFL = " << CFL << ", limit = " << CFL_limit << ")." << std::endl;
                return 1;
            }

            if (k % params.sample_interval == 0 || k == params.Nt - 1) {
                
                // 2. Create filenames with zero-padded numbers
                std::stringstream filename_rho, filename_u1, filename_u2, filename_T;

                std::cout << "Saving snapshot at t* = " << t_current << " (timestep k = " << k << ")" << std::endl;

                filename_rho << "output/rho/rho_" << std::setw(4) << std::setfill('0') << counter << ".txt";
                filename_u1  << "output/u1/u1_"  << std::setw(4) << std::setfill('0') << counter << ".txt";
                filename_u2  << "output/u2/u2_"  << std::setw(4) << std::setfill('0') << counter << ".txt";
                filename_T   << "output/T/T_"   << std::setw(4) << std::setfill('0') << counter << ".txt";

                // 3. Call your save function for each variable
                //    (Make sure you have created an "output" directory first!)
                save_matrix(U[0], filename_rho.str());
                save_matrix(u1, filename_u1.str());
                save_matrix(u2, filename_u2.str());
                save_matrix(T, filename_T.str());

                // 4. Increment the counter for the next snapshot
                counter++;
            }

            // --- 4th-Order Runge-Kutta Scheme ---
            dUdt0 = compute_rhs(U, t_current, params);
            for (int i=0; i<4; ++i) {
                K0[i] = params.dt * dUdt0[i];
                U_tmp[i] = U[i] + 0.5 * K0[i];
            }
            dUdt1 = compute_rhs(U_tmp, t_current + 0.5 * params.dt, params);
            for (int i=0; i<4; ++i) {
                K1[i] = params.dt * dUdt1[i];
                U_tmp[i] = U[i] + 0.5 * K1[i];
            }
            dUdt2 = compute_rhs(U_tmp, t_current + 0.5 * params.dt, params);
            for (int i=0; i<4; ++i) {
                K2[i] = params.dt * dUdt2[i];
                U_tmp[i] = U[i] + K2[i];
            }
            dUdt3 = compute_rhs(U_tmp, t_current + params.dt, params);

            for(int i=0; i<4; ++i) {
                K3[i] = params.dt * dUdt3[i];
                U[i] = U[i] + (1.0 / 6.0) * (K0[i] + 2.0 * K1[i] + 2.0 * K2[i] + K3[i]);
            }
        
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

    // Enforce boundary conditions on primitive variables
    u1.col(0).setZero();
    u1.col(params.Nx1 - 1).setZero();
    u1.row(0).setZero();
    u1.row(params.Nx2 - 1).setConstant(std::sin(2.0 * t / params.Re));
    u2.col(0).setZero();
    u2.col(params.Nx1 - 1).setZero();
    u2.row(0).setZero();
    u2.row(params.Nx2 - 1).setZero();
    T.col(0).setOnes();
    T.col(params.Nx1 - 1).setOnes();
    T.row(0).setOnes();
    T.row(params.Nx2 - 1).setOnes();

    // Re-calculate conservative variables on boundaries where primitives were changed.
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

    // Step 2: Compute RHS for Density on Walls (excluding corners)
    // Left wall (i=0)
    for (int j = 1; j < params.Nx2 - 1; ++j) {
        double du1_dx1 = (-3.0*u1(j,0) + 4.0*u1(j,1) - u1(j,2)) / (2.0*params.dx1);
        dUdt[0](j,0) = -rho(j,0) * du1_dx1;
    }
    // Right wall (i=Nx1-1)
    for (int j = 1; j < params.Nx2 - 1; ++j) {
        double du1_dx1 = (3.0*u1(j,params.Nx1-1) - 4.0*u1(j,params.Nx1-2) + u1(j,params.Nx1-3)) / (2.0*params.dx1);
        dUdt[0](j, params.Nx1-1) = -rho(j, params.Nx1-1) * du1_dx1;
    }
    // Bottom wall (j=0)
    for (int i = 1; i < params.Nx1 - 1; ++i) {
        double du2_dx2 = (-3.0*u2(0,i) + 4.0*u2(1,i) - u2(2,i)) / (2.0*params.dx2);
        dUdt[0](0,i) = -rho(0,i) * du2_dx2;
    }
    // Top wall (j=Nx2-1) excluding the points next to the top-left and top-right corners
    for (int i = 2; i < params.Nx1 - 2; ++i) {
        double du2_dx2 = (3.0*u2(params.Nx2-1,i) - 4.0*u2(params.Nx2-2,i) + u2(params.Nx2-3,i)) / (2.0*params.dx2);
        double drho_dx1 = (rho(params.Nx2-1, i+1) - rho(params.Nx2-1, i-1)) / (2.0*params.dx1);
        dUdt[0](params.Nx2-1, i) = -rho(params.Nx2-1,i)*du2_dx2 - u1(params.Nx2-1,i)*drho_dx1;
    }

    // Top wall points next to corners (i=1 and i=Nx1-2) using one-sided differences for drho_dx1 since the density at corners is not well-defined
    // Top wall (j=Nx2-1, i=1)
    double du2_dx2_tl = (3.0*u2(params.Nx2-1,1) - 4.0*u2(params.Nx2-2,1) + u2(params.Nx2-3,1)) / (2.0*params.dx2);
    double drho_dx1_tl = (-3.0*rho(params.Nx2-1,1) + 4.0*rho(params.Nx2-1,2) - rho(params.Nx2-1,3)) / (2.0*params.dx1);
    dUdt[0](params.Nx2-1, 1) = -rho(params.Nx2-1,1)*du2_dx2_tl - u1(params.Nx2-1,1)*drho_dx1_tl;

    double du2_dx2_tr = (3.0*u2(params.Nx2-1,params.Nx1-2) - 4.0*u2(params.Nx2-2,params.Nx1-2) + u2(params.Nx2-3,params.Nx1-2)) / (2.0*params.dx2);
    double drho_dx1_tr = (3.0*rho(params.Nx2-1,params.Nx1-2) - 4.0*rho(params.Nx2-1,params.Nx1-3) + rho(params.Nx2-1,params.Nx1-4)) / (2.0*params.dx1);
    dUdt[0](params.Nx2-1, params.Nx1-2) = -rho(params.Nx2-1,params.Nx1-2)*du2_dx2_tr - u1(params.Nx2-1,params.Nx1-2)*drho_dx1_tr;

    // // Top wall (j=Nx2-1)
    // for (int i = 1; i < params.Nx1 - 1; ++i) {
    //     double du2_dx2 = (3.0*u2(params.Nx2-1,i) - 4.0*u2(params.Nx2-2,i) + u2(params.Nx2-3,i)) / (2.0*params.dx2); // Eq. 46
    //     double drho_dx1 = (rho(params.Nx2-1, i+1) - rho(params.Nx2-1, i-1)) / (2.0*params.dx1);
    //     dUdt[0](params.Nx2-1, i) = -rho(params.Nx2-1,i)*du2_dx2 - u1(params.Nx2-1,i)*drho_dx1; // Eq. 25
    // }

    // Step 3: Compute RHS for Density on Corner Points (actually we don't need to, but for clarity)
    // dUdt[0](0, 0) = 0.0;
    // dUdt[0](0, params.Nx1 - 1) = 0.0;
    // dUdt[0](params.Nx2-1, 0) = 0.0;
    // dUdt[0](params.Nx2-1, params.Nx1 - 1) = 0.0;

    return dUdt;
}




// int main() {

//     const double pi = M_PI;
//     const static Eigen::IOFormat TXTFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n");

//     double Lt = 1;
//     double dt = 0.00001;
//     double dt_sample = dt;
//     int    Nx = 120;
//     int    Nt = 1 + Lt / dt;
//     int    Ns = 1 + Lt / dt_sample;
//     double dx = Lx / Nx; // periodic boundary condition
//     double nu = 0.01;

//     // generate spatial grid
//     Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, 0, Lx - dx);
//     // specify the initial condition
//     Eigen::VectorXd u_init = Eigen::VectorXd::Zero(Nx);
//     // create a vector to store the old & new current solution
//     Eigen::VectorXd u_old = Eigen::VectorXd::Zero(Nx);
//     Eigen::VectorXd u_new = Eigen::VectorXd::Zero(Nx);
//     // create a vector to store the advection term
//     Eigen::VectorXd uu_dx = Eigen::VectorXd::Zero(Nx);
//     // create a vector to store the diffusion term
//     Eigen::VectorXd u_dxx = Eigen::VectorXd::Zero(Nx);
//     // create a matrix to store the sampled solution
//     Eigen::MatrixXd u_sample = Eigen::MatrixXd::Zero(Nx, Ns);

//     int j, k, kl, kr;

//     // set the initial condition
//     u_init = x.array().sin() * (-(x.array() - pi) * (x.array() - pi)).exp();
//     u_old = u_init; // set the initial condition as the old current solution

//     // The first approach is explicit: forward Euler + 1st-order upwind scheme + 2nd-order centered scheme

//     double dt_max = dx * dx / (2 * nu + u_init.cwiseAbs().maxCoeff() * dx); // stability condition
//     std::cout << "Maximum allowable time step size for stability: dt_max = " << dt_max << std::endl;

//     for (j = 0; j < Nt; ++j) {

//         std::cout << "current maximal abs value of u = " << u_old.cwiseAbs().maxCoeff() << std::endl;
//         if (!u_old.allFinite()) {
//             std::cerr << "Simulation diverged at time step j = " << j << " (t = " << j * dt << ")" << std::endl;
//             break;
//         }

//         if (j % int(dt_sample / dt) == 0) {
//             u_sample.col(j / int(dt_sample / dt)) = u_old; // sample the current solution
//         }

//         for (k = 0; k < Nx; ++k) {
//             kl = (k - 1 + Nx) % Nx; // periodic boundary condition
//             kr = (k + 1) % Nx;
//             // 1st-order upwind scheme for the advection term
//             // recall that if u>=0 (u<0), we have to use backward (forward) scheme to ensure numerical stability
//             uu_dx(k) = ((u_old(k) + std::abs(u_old(k))) / 2.0) * (u_old(k) - u_old(kl)) / dx
//                      + ((u_old(k) - std::abs(u_old(k))) / 2.0) * (u_old(kr) - u_old(k)) / dx;
//             u_dxx(k) = (u_old(kl) - 2 * u_old(k) + u_old(kr)) / (dx * dx); // 2nd-order centered scheme
//         }

//         u_new = u_old.array() + dt * (-1 * uu_dx.array() + nu * u_dxx.array()); // update the solution
//         u_old = u_new; // update the old current solution

//     }
    
//     std::stringstream filename_str;
//     filename_str << "burgers_explicit_nu_" << nu << "_nx_" << Nx << "_dt_" << dt << ".txt";
//     std::string filename = filename_str.str();
//     std::ofstream file_explicit(filename);
//     file_explicit << u_sample.format(TXTFormat);
//     file_explicit.close();

//     return 0;
// }
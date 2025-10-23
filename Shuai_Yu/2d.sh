#!/bin/bash -l

#SBATCH --job-name=2DLidCav
#SBATCH --time=03:59:59
#SBATCH --nodes=1

Re="100.0"
Ma="0.025"
Pr="0.7"
gamma="1.4"
T_init="300.0"
Nx1="9"
Nx2="9"
t_final="500.0"
relaxation_factor="0.6"
dt="0.00001"
sample_interval=500000

echo "Starting job..."
srun ./solver_euler $Re $Ma $Pr $T_init $Nx1 $Nx2 $t_final $relaxation_factor $dt $sample_interval
echo "Job finished..."

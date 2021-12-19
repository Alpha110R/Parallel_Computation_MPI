# Parallel_Computation_MPI
Parallel computation exercise to create cartesian topology 2 dimensions and use Shear sort and Odd-Even sort
Each process has a rectangle area and its ID. The algorithm uses shear sort and the communication between the process is done by the Odd-Even sort.
It sorts by the areas, if the areas are equals it will be by the ID.
In the PDF there ara the instructions for this exercise.
mpicc Targil2.c -lm -o targil
mpiexec -n 16 ./targil

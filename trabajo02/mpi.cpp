

#include <iostream>
#include <mpi.h>

void Mat_vect_mult(
	double local_A[], double local_x[], double local_y[],
	int local_m, int n, int local_n, MPI_Comm comm);

int main(int argc, char *argv[])
{
	int comm_sz, my_rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int n = 1000; // Tamaño total de la matriz y el vector
	int local_n = n / comm_sz;
	int local_m = local_n;

	double *local_A = (double *)malloc(local_m * n * sizeof(double));
	double *local_x = (double *)malloc(local_n * sizeof(double));
	double *local_y = (double *)malloc(local_m * sizeof(double));

	// Inicializar local_A, local_x, y local_y según las necesidades

	Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, MPI_COMM_WORLD);

	// Recopilar resultados de todos los procesos
	double *y = NULL;
	if (my_rank == 0)
	{
		y = (double *)malloc(n * sizeof(double));
	}

	MPI_Gather(local_y, local_m, MPI_DOUBLE, y, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (my_rank == 0)
	{
		// Realizar acciones finales con el resultado global y
		free(y);
	}

	free(local_A);
	free(local_x);
	free(local_y);

	MPI_Finalize();
	return 0;
}

void Mat_vect_mult(
	double local_A[], double local_x[], double local_y[],
	int local_m, int n, int local_n, MPI_Comm comm)
{
	double *x;
	int local_i, j;
	int local_ok = 1;

	x = (double *)malloc(n * sizeof(double));
	MPI_Allgather(local_x, local_n, MPI_DOUBLE,
				  x, local_n, MPI_DOUBLE, comm);

	for (local_i = 0; local_i < local_m; local_i++)
	{
		local_y[local_i] = 0.0;
		for (j = 0; j < n; j++)
		{
			local_y[local_i] += local_A[local_i * n + j] * x[j];
		}
	}

	free(x);
}

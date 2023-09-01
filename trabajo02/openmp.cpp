
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void Hello(void); 
void Odd_Even_Sort(int *a, int n, int thread_count);
void Odd_Even_Sort2(int *a, int n, int thread_count);

int main()
{
	const int n = 100000;			// Tamaño del arreglo
	const int thread_count = 4; // Número de hilos

	int *arr1 = new int[n];
	int *arr2 = new int[n];

	srand(time(nullptr));
	for (int i = 0; i < n; ++i)
	{
		arr1[i] = rand() % 1000;
		arr2[i] = arr1[i];
	}

	auto start1 = high_resolution_clock::now();
	Odd_Even_Sort(arr1, n, thread_count);
	auto stop1 = high_resolution_clock::now();
	auto duration1 = duration_cast<milliseconds>(stop1 - start1);

	auto start2 = high_resolution_clock::now();
	Odd_Even_Sort2(arr2, n, thread_count);
	auto stop2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(stop2 - start2);

	cout << "Tiempo para Odd_Even_Sort: " << duration1.count() << " ms" << endl;
	cout << "Tiempo para Odd_Even_Sort2: " << duration2.count() << " ms" << endl;

	delete[] arr1;
	delete[] arr2;

	return 0;
}

void Hello(void)
{
	int my_rank = omp_get_thread_num();
	int thread_count = omp_get_num_threads();
	printf("Hello from thread %d of %d\n", my_rank, thread_count);
}

void Odd_Even_Sort(int *a, int n, int thread_count)
{
	int tmp;

	for (int phase = 0; phase < n; phase++)
	{
		if (phase % 2 == 0)
		{
#pragma omp parallel for num_threads(thread_count) default(none) shared(a, n) private(tmp)
			for (int i = 1; i < n; i += 2)
			{
				if (a[i - 1] > a[i])
				{
					tmp = a[i - 1];
					a[i - 1] = a[i];
					a[i] = tmp;
				}
			}
		}
		else
		{
#pragma omp parallel for num_threads(thread_count) default(none) shared(a, n) private(tmp)
			for (int i = 1; i < n - 1; i += 2)
			{
				if (a[i] > a[i + 1])
				{
					tmp = a[i + 1];
					a[i + 1] = a[i];
					a[i] = tmp;
				}
			}
		}
	}
}

void Odd_Even_Sort2(int *a, int n, int thread_count)
{
	int tmp;
	int phase;
#pragma omp parallel num_threads(thread_count) default(none) shared(a, n) private(i, tmp, phase)
	for (phase = 0; phase < n; phase++)
	{
		if (phase % 2 == 0)
#pragma omp for
			for (int i = 1; i < n; i += 2)
			{
				if (a[i - 1] > a[i])
				{
					tmp = a[i - 1];
					a[i - 1] = a[i];
					a[i] = tmp;
				}
			}
		else
#pragma omp for
			for (int i = 1; i < n - 1; i += 2)
			{
				if (a[i] > a[i + 1])
				{
					tmp = a[i + 1];
					a[i + 1] = a[i];
					a[i] = tmp;
				}
			}
	}
}

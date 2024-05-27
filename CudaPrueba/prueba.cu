#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "simutorno.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Windows.h>

#define ERROR_CHECK { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

typedef LARGE_INTEGER timeStamp;
double getTime();


int SimulacionTornoCPU(int pasossim, int vtotal, int utotal)
{
	/* Parametros de mecanizado */
	double incA = 360.0 / (double)PuntosVueltaHelicoide;
	for (int u = 0; u < S.UPoints; u++) /* Para cada punto de la superficie */
	{
		for (int v = 0; v < S.VPoints; v++)
		{
			double AvanceMin = 1e10;
			double angle = 0.0;
			for (int i = 0; i < pasossim; i++)  /* Giro del torno en el eje X */
			{
				// Se rota el punto actual (solo interesa la coordenada y)
				double py = S.Buffer[v][u].y * cos(angle * M_PI_180) - S.Buffer[v][u].z * sin(angle * M_PI_180);
				// Calcula la distancia al origen del punto transformado 
				// Si y es la menor se almacena
				if (py < AvanceMin)
				{
					AvanceMin = py;
				}
				angle += incA;
			}
			int p = S.VPoints * u + v;
			CPUBufferMenorY[p] = AvanceMin;
		}
	}
	return OKSIM;
}

int SimulacionTornoGPU(int pasossim, int vtotal, int utotal){ return OKSIM;}

// Declaraciones adelantadas de funciones
int SimulacionTornoCPU(int pasossim, int vtotal, int utotal);
int LeerSuperficie(const char* fichero, int& vtotal, int& utotal);

void runTest(int argc, char** argv)
{
	/* Variables de tiempo */
	double gpu_start_time, gpu_end_time;
	double cpu_start_time, cpu_end_time;

	/* Numero de argumentos */
	if (argc != 3)
	{
		fprintf(stderr, "Numero de parametros incorecto\n");
		fprintf(stderr, "Uso: %s superficie pasossim\n", argv[0]);
		return;
	}

	/* Datos de los tests */
	int utotal = 0;
	int vtotal = 0;
	/* Apertura de Fichero */
	printf("Prueba simulaci?n torno...\n");
	/* Datos de la superficie */
	if (LeerSuperficie((char*)argv[1], vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Lectura de superficie incorrecta\n");
		return;
	}
	int pasossim = atoi((char*)argv[2]);
	// Creaci?n buffer resultados para versiones CPU y GPU
	CPUBufferMenorY = (double*)malloc(S.UPoints * S.VPoints * sizeof(double));
	GPUBufferMenorY = (double*)malloc(S.UPoints * S.VPoints * sizeof(double));

	/* Algoritmo a paralelizar */
	cpu_start_time = getTime();
	if (SimulacionTornoCPU(pasossim, vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Simulaci?n CPU incorrecta\n");
		BorrarSuperficie();
		if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
		if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
		exit(1);
	}
	cpu_end_time = getTime();
	/* Algoritmo a implementar */
	cudaSetDevice(0);
	gpu_start_time = getTime();
	if (SimulacionTornoGPU(pasossim, vtotal, utotal) == ERRORSIM)
	{
		fprintf(stderr, "Simulaci?n GPU incorrecta\n");
		BorrarSuperficie();
		if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
		if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
		return;
	}
	cudaDeviceSynchronize();
	gpu_end_time = getTime();
	// Comparaci?n de correcci?n
	int comprobar = OKSIM;
	for (int i = 0; i < S.UPoints * S.VPoints; i++)
	{
		if (fabs(CPUBufferMenorY[i] - GPUBufferMenorY[i]) < 1e-6)
		{
			comprobar = ERRORSIM;
			fprintf(stderr, "Fallo en paso %d de simulaci?n, valor correcto Y=%lf\n", i, CPUBufferMenorY[i]);
		}
	}
	// Impresion de resultados
	if (comprobar == OKSIM)
	{
		printf("Simulaci?n correcta!\n");

	}
	// Impresi?n de resultados
	printf("Tiempo ejecuci?n GPU : %fs\n", \
		gpu_end_time - gpu_start_time);
	printf("Tiempo de ejecuci?n en la CPU : %fs\n", \
		cpu_end_time - cpu_start_time);
	printf("Se ha conseguido un factor de aceleraci?n %fx utilizando CUDA\n", (cpu_end_time - cpu_start_time) / (gpu_end_time - gpu_start_time));
	// Limpieza de buffers
	BorrarSuperficie();
	if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
	if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
	return;
}

int main(int argc, char** argv)
{
	runTest(argc, argv);
	getchar();
}

/* Funciones auxiliares */
double getTime()
{
	timeStamp start;
	timeStamp dwFreq;
	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);
	return double(start.QuadPart) / double(dwFreq.QuadPart);
}

/// <summary>
/// Leer el fichero de superficie .for
/// </summary>
/// <param name="fichero"></param>
/// <param name="vtotal"></param>
/// <param name="utotal"></param>
/// <returns></returns>
int LeerSuperficie(const char* fichero, int& vtotal, int& utotal)
{
	int i, j, count;		/* Variables de bucle */
	FILE* fpin; 			/* Fichero */
	char cadena[255];
	double x, y, z;

	cadena[0] = 0;
	/* Apertura de Fichero */
	if ((fpin = fopen(fichero, "r")) == NULL) return ERRORSIM;
	/* Lectura de cabecera */
	while (!feof(fpin) && strcmp(cadena, "[HEADER]")) fscanf(fpin, "%s\n", cadena);
	if (fscanf(fpin, "SECTION NUMBER=%d\n", &utotal) < 0) return ERRORSIM;
	if (fscanf(fpin, "POINTS PER SECTION=%d\n", &vtotal) < 0) return ERRORSIM;
	if (fscanf(fpin, "STEP=%lf\n", &PasoHelicoide) < 0) return ERRORSIM;
	if (fscanf(fpin, "POINTS PER ROUND=%d\n", &PuntosVueltaHelicoide) < 0) return ERRORSIM;
	if (utotal * vtotal <= 0) return ERRORSIM;
	/* Localizacion de comienzo */
	while (!feof(fpin) && strcmp(cadena, "[GEOMETRY]")) fscanf(fpin, "%s\n", cadena);
	if (feof(fpin)) return ERRORSIM;
	/* Inicializaci?n de parametros geometricos */
	if (CrearSuperficie(utotal, vtotal) == ERRORSIM) return ERRORSIM;
	/* Lectura de coordenadas */
	count = 0;
	for (i = 0; i < utotal; i++)
	{
		for (j = 0; j < vtotal; j++)
		{
			if (!feof(fpin))
			{
				fscanf(fpin, "%lf %lf %lf\n", &x, &y, &z);
				S.Buffer[j][i].x = x;
				S.Buffer[j][i].y = y;
				S.Buffer[j][i].z = z;
				count++;
			}
			else break;
		}
	}
	fclose(fpin);
	if (count != utotal * vtotal) return ERRORSIM;
	return OKSIM;
}




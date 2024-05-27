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

// Macro para verificar errores de CUDA
#define ERROR_CHECK { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

typedef LARGE_INTEGER timeStamp; // Definición de un alias para la gestión de tiempos
double getTime(); // Declaración de la función para obtener el tiempo actual

// Simulación en CPU
int SimulacionTornoCPU(int pasossim, int vtotal, int utotal)
{
    double incA = 360.0 / (double)PuntosVueltaHelicoide; // Incremento del ángulo para cada paso de simulación
    for (int u = 0; u < S.UPoints; u++) // Itera sobre los puntos de la superficie en la dirección U
    {
        for (int v = 0; v < S.VPoints; v++) // Itera sobre los puntos de la superficie en la dirección V
        {
            double AvanceMin = 1e10; // Inicializa la mínima distancia
            double angle = 0.0; // Inicializa el ángulo
            for (int i = 0; i < pasossim; i++) // Itera sobre los pasos de la simulación
            {
                // Calcula la nueva posición del punto rotado (solo interesa la coordenada y)
                double py = S.Buffer[v][u].y * cos(angle * M_PI_180) - S.Buffer[v][u].z * sin(angle * M_PI_180);
                // Si la nueva posición y es menor que la mínima, se actualiza AvanceMin
                if (py < AvanceMin)
                {
                    AvanceMin = py;
                }
                angle += incA; // Incrementa el ángulo
            }
            int p = S.VPoints * u + v; // Calcula la posición en el buffer de resultados
            CPUBufferMenorY[p] = AvanceMin; // Guarda el valor mínimo en el buffer
        }
    }
    return OKSIM; // Retorna código de éxito
}

int SimulacionTornoGPU(int pasossim, int vtotal, int utotal) { return OKSIM; } // Simulación en GPU (a implementar)

// Declaraciones adelantadas de funciones
int SimulacionTornoCPU(int pasossim, int vtotal, int utotal);
int LeerSuperficie(const char* fichero, int& vtotal, int& utotal);

void runTest(int argc, char** argv)
{
    // Variables de tiempo para medir rendimiento
    double gpu_start_time, gpu_end_time;
    double cpu_start_time, cpu_end_time;

    // Verifica el número de argumentos
    if (argc != 3)
    {
        fprintf(stderr, "Numero de parametros incorrecto\n");
        fprintf(stderr, "Uso: %s superficie pasossim\n", argv[0]);
        return;
    }

    // Datos de la superficie
    int utotal = 0;
    int vtotal = 0;

    // Apertura de Fichero
    printf("Prueba simulación torno...\n");

    // Lectura de la superficie desde un archivo
    if (LeerSuperficie(argv[1], vtotal, utotal) == ERRORSIM)
    {
        fprintf(stderr, "Lectura de superficie incorrecta\n");
        return;
    }

    int pasossim = atoi(argv[2]); // Convierte los pasos de simulación de cadena a entero

    // Creación de buffer para resultados de CPU y GPU
    CPUBufferMenorY = (double*)malloc(S.UPoints * S.VPoints * sizeof(double));
    GPUBufferMenorY = (double*)malloc(S.UPoints * S.VPoints * sizeof(double));

    // Ejecución del algoritmo en CPU
    cpu_start_time = getTime();
    if (SimulacionTornoCPU(pasossim, vtotal, utotal) == ERRORSIM)
    {
        fprintf(stderr, "Simulación CPU incorrecta\n");
        BorrarSuperficie();
        if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
        if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
        exit(1);
    }
    cpu_end_time = getTime();

    // Ejecución del algoritmo en GPU
    cudaSetDevice(0);
    gpu_start_time = getTime();
    if (SimulacionTornoGPU(pasossim, vtotal, utotal) == ERRORSIM)
    {
        fprintf(stderr, "Simulación GPU incorrecta\n");
        BorrarSuperficie();
        if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
        if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
        return;
    }
    cudaDeviceSynchronize();
    gpu_end_time = getTime();

    // Comparación de resultados entre CPU y GPU
    int comprobar = OKSIM;
    for (int i = 0; i < S.UPoints * S.VPoints; i++)
    {
        if (fabs(CPUBufferMenorY[i] - GPUBufferMenorY[i]) >= 1e-6)
        {
            comprobar = ERRORSIM;
            fprintf(stderr, "Fallo en paso %d de simulación, valor correcto Y=%lf\n", i, CPUBufferMenorY[i]);
        }
    }

    // Impresión de resultados
    if (comprobar == OKSIM)
    {
        printf("Simulación correcta!\n");
    }

    // Impresión de tiempos de ejecución
    printf("Tiempo ejecución GPU: %fs\n", gpu_end_time - gpu_start_time);
    printf("Tiempo de ejecución en la CPU: %fs\n", cpu_end_time - cpu_start_time);
    printf("Se ha conseguido un factor de aceleración %fx utilizando CUDA\n", (cpu_end_time - gpu_start_time) / (gpu_end_time - gpu_start_time));

    // Limpieza de buffers
    BorrarSuperficie();
    if (CPUBufferMenorY != NULL) free(CPUBufferMenorY);
    if (GPUBufferMenorY != NULL) free(GPUBufferMenorY);
}

int main(int argc, char** argv)
{
    runTest(argc, argv); // Ejecuta la prueba de simulación
    getchar(); // Espera una entrada del usuario antes de salir
}

// Función para obtener el tiempo actual
double getTime()
{
    timeStamp start;
    timeStamp dwFreq;
    QueryPerformanceFrequency(&dwFreq); // Obtiene la frecuencia del contador de rendimiento
    QueryPerformanceCounter(&start); // Obtiene el valor actual del contador
    return double(start.QuadPart) / double(dwFreq.QuadPart); // Calcula el tiempo en segundos
}

// Leer el fichero de superficie
int LeerSuperficie(const char* fichero, int& vtotal, int& utotal)
{
    int i, j, count;
    FILE* fpin;
    char cadena[255];
    double x, y, z;

    cadena[0] = 0;
    // Apertura del fichero
    if ((fpin = fopen(fichero, "r")) == NULL) return ERRORSIM;

    // Lectura de la cabecera
    while (!feof(fpin) && strcmp(cadena, "[HEADER]")) fscanf(fpin, "%s\n", cadena);
    if (fscanf(fpin, "SECTION NUMBER=%d\n", &utotal) < 0) return ERRORSIM;
    if (fscanf(fpin, "POINTS PER SECTION=%d\n", &vtotal) < 0) return ERRORSIM;
    if (fscanf(fpin, "STEP=%lf\n", &PasoHelicoide) < 0) return ERRORSIM;
    if (fscanf(fpin, "POINTS PER ROUND=%d\n", &PuntosVueltaHelicoide) < 0) return ERRORSIM;
    if (utotal * vtotal <= 0) return ERRORSIM;

    // Localización de la sección de geometría
    while (!feof(fpin) && strcmp(cadena, "[GEOMETRY]")) fscanf(fpin, "%s\n", cadena);
    if (feof(fpin)) return ERRORSIM;

    // Inicialización de la superficie
    if (CrearSuperficie(utotal, vtotal) == ERRORSIM) return ERRORSIM;

    // Lectura de coordenadas
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

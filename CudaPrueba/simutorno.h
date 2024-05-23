#pragma once
/*----------------------------------------------------------------------------*/
/*  FICHERO:       simutornoCPU.h									          */
/*  AUTOR:         Antonio Jimeno											  */
/*													                          */
/*  RESUMEN												                      */
/*  ~~~~~~~												                      */
/* Fichero de definiciones y estructuras                                      */
/*    						                                                  */
/*----------------------------------------------------------------------------*/

#ifndef _SIMUTORNO_H_
#define _SIMUTORNO_H_

/*============================================================================ */
/* Constantes											                       */
/*============================================================================ */
#define ERRORSIM 1
#define OKSIM    0
#define TOOLYPOS  1000.0
#define TOOLWIDTH 10.0
#define M_PI_180 0.01745329252

/*============================================================================ */
/* Estructuras											                       */
/*============================================================================ */

struct sTPoint3D
{
	double x;
	double y;
	double z;
};
typedef struct sTPoint3D TPoint3D;

typedef struct sTConf TConf;

struct sTSurf
{
	int UPoints;
	int VPoints;
	TPoint3D** Buffer;
};
typedef struct sTSurf TSurf;


/*============================================================================ */
/* Variables Globales										                   */
/*============================================================================ */
TSurf S;
double* GPUBufferMenorY;
double* CPUBufferMenorY;
double PasoHelicoide;
int PuntosVueltaHelicoide;


/*============================================================================ */
/* Funciones de tratamiento de memoria							 */
/*============================================================================ */
void BorrarSuperficie(void)
{
	int i;
	if (S.Buffer != NULL)
	{
		for (i = 0; i < S.VPoints; i++)
			if (S.Buffer[i] != NULL) free(S.Buffer[i]);
		free(S.Buffer);
		S.Buffer = NULL;
	}
}

int CrearSuperficie(int uPoints, int vPoints)
{
	int j;
	S.UPoints = uPoints;
	S.VPoints = vPoints;
	S.Buffer = (TPoint3D**)malloc(S.VPoints * sizeof(void*));
	if (S.Buffer == NULL) return ERRORSIM;
	for (j = 0; j < S.VPoints; j++)
	{
		S.Buffer[j] = (TPoint3D*)malloc(S.UPoints * (int)sizeof(TPoint3D));
		if (S.Buffer[j] == NULL)
		{
			BorrarSuperficie();
			return ERRORSIM;
		}
	}
	return OKSIM;
}


#endif


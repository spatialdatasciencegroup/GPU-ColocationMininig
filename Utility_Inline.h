#pragma once
#include "Constant.h"
#include "DataTypes.h"
#include "math_functions.h"
//static __inline__ __host__ __device__ Real ArcInRadians(Real lon1, Real lat1, Real lon2, Real lat2, param parameter) {
//	Real latitudeArc = (lat1 - lat2) * parameter.DEG_TO_RAD;
//	Real longitudeArc = (lon1 - lon2) * parameter.DEG_TO_RAD;
//	Real latitudeH = sin(latitudeArc * 0.5);
//	latitudeH *= latitudeH;
//	Real lontitudeH = sin(longitudeArc * 0.5);
//	lontitudeH *= lontitudeH;
//	Real tmp = cos(lat1*parameter.DEG_TO_RAD) * cos(lat2*parameter.DEG_TO_RAD);
//	return 2.0 * asin(sqrt(latitudeH + tmp*lontitudeH));
//}
static __inline__ __host__ __device__ Real ArcInRadians(Real lon1, Real lat1, Real lon2, Real lat2) {
	Real latitudeArc = (lat1 - lat2) * CONSTANT_PARAMETER.DEG_TO_RAD;
	Real longitudeArc = (lon1 - lon2) * CONSTANT_PARAMETER.DEG_TO_RAD;
	Real latitudeH = sin(latitudeArc * 0.5);
	latitudeH *= latitudeH;
	Real lontitudeH = sin(longitudeArc * 0.5);
	lontitudeH *= lontitudeH;
	Real tmp = cos(lat1*CONSTANT_PARAMETER.DEG_TO_RAD) * cos(lat2*CONSTANT_PARAMETER.DEG_TO_RAD);
	return 2.0 * asin(sqrt(latitudeH + tmp*lontitudeH));
}
static __inline__ __host__ __device__ Real device_distanceinMeters(Integer idx1, Integer idx2) {
	Real x2 = c_dainstanceLocationX[idx2];
	Real x1 = c_dainstanceLocationX[idx1];
	Real y2 = c_dainstanceLocationY[idx2];
	Real y1 = c_dainstanceLocationY[idx1];
	//Real distance = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
	Real squareDistance = pow((x2 - x1), 2) + pow((y2 - y1), 2);
	return squareDistance;
}

__device__ static SInteger getNeighborCellID(Integer x, Integer y, Integer TID) {
	SInteger NCID;
	NCID = y*GRID_STRUCTURE.gridSize.x + x;
	if (TID%GRID_STRUCTURE.gridSize.x == 0) {
		if ((NCID + 1) % GRID_STRUCTURE.gridSize.x == 0) {
			return -1;
		}
		return NCID;
	}
	else if ((TID + 1) % GRID_STRUCTURE.gridSize.x == 0) {
		if (NCID % GRID_STRUCTURE.gridSize.x == 0) {
			return -1;
		}
		return NCID;
	}
	else {
		return NCID;
	}
}

static __inline__ StatusType CudaSafeCall(cudaError_t Status)
{
	if (Status == cudaErrorInvalidValue)
	{
		printf("Symbol");
	}
	if (cudaSuccess != Status)
	{
		printf(cudaGetErrorString(Status));
		/*printf("\t in Function ");
		printf(Function);*/
		return ST_CUDAERR;
	}
	return ST_NOERR;
}

#pragma once
#pragma once
#include "includes.h"
#define BLOCK_MAX_DIM 1024
typedef unsigned int Integer; //no negative values range 0 to  4,294,967,295
typedef long long LongInteger; 
//typedef __int64 Integer;
//typedef __int64 Index64;
typedef int Index64;
typedef int SInteger; //with both positive and negative values
typedef float Real;


struct sNeighbor {
	vector<struct sInstance> neighbors;
};

struct sFeatureStats {
	Integer start;
	Integer end;
	Integer count;
};

struct sLocations {
	Real x;
	Real y;
};


struct MyComparator
{
	const vector<Integer> & value_vector;

	MyComparator(const vector<Integer> & val_vec) :
		value_vector(val_vec) {}

	bool operator()(Integer i1, Integer i2)
	{
		return value_vector[i1] < value_vector[i2];
	}
};

struct gridDomain {
	Integer width;
	Integer height;
	Integer numCells;
	Integer MaxCellID;

};

struct param {
	Real thresholdDistance;
	Real squaredDistanceThreshold;
	Real PIthreshold;
	Real DEG_TO_RAD;
	Real EARTH_RADIUS_IN_METERS;
	Integer FilterON_OFF;

};

struct scalar2 {
	Real x;
	Real y;
};

struct Integer2 {
	Integer x;
	Integer y;

};

struct cBox {
	scalar2 minBound;
	scalar2 maxBound;

};
struct gridBox {
	cBox computeZone;
	Integer2 gridSize;
	scalar2 offsets;
	Integer totalCells;
	Real cellWidth;
};

enum StatusType
{
	ST_NOERR,
	ST_FILEERR,
	ST_PARAMERR,
	ST_CALCERR,
	ST_MEMERR,
	ST_CUDAERR,
	ST_ETCERR
};




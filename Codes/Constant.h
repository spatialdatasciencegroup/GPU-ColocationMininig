#pragma once
#include <cuda_runtime.h>	
#include <cuda.h>
#include "DataTypes.h"
__constant__ param CONSTANT_PARAMETER;
__constant__ Integer *c_dainstanceList;
__constant__ Integer *c_dafeatureInstanceStart;
__constant__ Integer *c_dafeatureInstanceEnd;
__constant__ Integer *c_dafeatureInstanceCount;
__constant__ Real	*c_dainstanceLocationX;
__constant__ Real	*c_dainstanceLocationY;
__constant__ Integer c_maxInstancesNum;
__constant__ Integer c_maxFeaturesNum;

//__constant__ Integer *c_daSlotCouts;
//__constant__ Integer *c_daIndexes;
__constant__ SInteger *c_dacellEventInstanceCount;
__constant__ gridBox GRID_STRUCTURE;
__constant__ SInteger *c_dacellEventInstanceStart;
__constant__ SInteger *c_dacellEventInstanceEnd;
__constant__ Integer *c_dainstanceCellIDs;
__constant__ Integer *c_dacellBasedSortedIndex;
#pragma once
#include "DataTypes.h"
#include "Constant.h"
#include "Utility_Inline.h"
#include "device_launch_parameters.h"

static __inline__ __device__ size_t CudaGetTargetID()
{
	return blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ static void degree2TableSlotCounter_Kernel(size_t maxCount, Integer* slots, size_t start, size_t end, size_t secondstart, Integer* d_bitmap,Integer combiningFeatureID)
{
	size_t TID = CudaGetTargetID();
	size_t idx1 = TID + start;
	if (idx1 > end || TID > maxCount) {
		return;
	}
	//SInteger neighborCells[]
	Integer currentCell = c_dainstanceCellIDs[idx1];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	SInteger Range = 1;
	for (SInteger j = -1; j <= Range; ++j)
	{
		for (SInteger i = -1; i <= Range; ++i)
		{
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells)
			{
				continue;
			}
			size_t index = NCID*c_maxFeaturesNum + combiningFeatureID;
			size_t count = c_dacellEventInstanceCount[index];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[index];
				size_t endIndex = c_dacellEventInstanceStart[index] + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer instanceID = c_dacellBasedSortedIndex[pos];
					Real dist = device_distanceinMeters(idx1, instanceID);
					if (dist <= CONSTANT_PARAMETER.squaredDistanceThreshold) {
						Integer slotValue = slots[TID];
						slotValue++;
						slots[TID]= slotValue;
						size_t firstcount = end - start;
						size_t bitIndex1 = TID;
						size_t bitIndex2 = instanceID - secondstart + firstcount;
						d_bitmap[bitIndex1] = 1;
						d_bitmap[bitIndex2] = 1;
					}
				}
			}
		}
	}
}
__global__ static void degree2TableGenerator_Kernel(Integer*slots, Index64* indexes, Integer* d_intermediate, size_t start, size_t end, Integer combiningFeatureID)
{
	size_t TID = CudaGetTargetID();
	size_t idx1 = TID + start;
	if (idx1 > end ) {
		return;
	}
	Integer slotsValue = slots[TID];
	if (slotsValue == 0) {
		return;
	}
	size_t index = indexes[TID];
	Integer currentCell = c_dainstanceCellIDs[idx1];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);

	SInteger Range = 1;
	for (SInteger j = -1; j <= Range; ++j)
	{
		for (SInteger i = -1; i <= Range; ++i)
		{
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells)
			{
				continue;
			}
			size_t indx = NCID*c_maxFeaturesNum + combiningFeatureID;
			size_t count = c_dacellEventInstanceCount[indx];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[indx];
				size_t endIndex = startIndex + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer instanceID = c_dacellBasedSortedIndex[pos];
					Real dist = device_distanceinMeters(idx1, instanceID);
					if (dist <= CONSTANT_PARAMETER.squaredDistanceThreshold) {
						d_intermediate[index] = idx1;
						index++;
						d_intermediate[index] = instanceID;
						index++;
					}
				}
			}
		}
	}
}

__device__ static SInteger checkIfCommonNeighbour(Integer NCID, Integer currentCell) {
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	SInteger Range = 1;
	for (SInteger j = -1; j <= Range; ++j)
	{
		for (SInteger i = -1; i <= Range; ++i)
		{
			SInteger NextNCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells)
			{
				continue;
			}
			if (NextNCID == NCID) {
				return 1;
			}
			if (NextNCID > NCID) {
				return 0;
			}
		}
	}
	return 0;
}

__global__ static void generalInstanceTableSlotCounter_Kernel4(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID)
{
	size_t TID = CudaGetTargetID();
	if (TID >= instaceCountTable1) {
		return;
	}
	Integer degreek = degree - 1;
	size_t counter = 0;
	SInteger Range = 1;
	Integer currentInstanceID = table1[TID*degreek];
	Integer currentCell = c_dainstanceCellIDs[currentInstanceID];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	for (SInteger j = -1; j <= 1; ++j) {
		for (SInteger i = -1; i <= 1; ++i) {
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells) {
				continue;
			}
			size_t lastIndex = NCID*c_maxFeaturesNum + lastFeatureID;
			size_t count = c_dacellEventInstanceCount[lastIndex];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[lastIndex];
				size_t endIndex = startIndex + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer lastInstanceID = c_dacellBasedSortedIndex[pos];
					Integer flag2 = 1;
					for (int degIndex = 0; degIndex < degreek; degIndex++) {
						Integer cIndex = TID*degreek + degIndex;
						currentInstanceID = table1[cIndex];
						Real dist = device_distanceinMeters(currentInstanceID, lastInstanceID);
						if (dist > CONSTANT_PARAMETER.squaredDistanceThreshold) {
							flag2 = 0;
							break;
						}
					}
					if (flag2 == 1) {
						counter++;
						slots[TID] = counter;
						size_t offset = 0;
						Integer bitIndex = 0;
						Integer fType;
						Integer value = 0;
						for (size_t idx4 = 0; idx4 < degreek; idx4++) {
							size_t idxCurrent = TID*degreek + idx4;
							value = table1[idxCurrent];
							fType = coloc[idx4];
							size_t fstart = c_dafeatureInstanceStart[fType];
							size_t fcount = c_dafeatureInstanceCount[fType];
							bitIndex = value - fstart + offset;
							offset += fcount;
							d_bitmap[bitIndex] = 1;
						}
						Integer lastStart = c_dafeatureInstanceStart[lastFeatureID];
						bitIndex = lastInstanceID - lastStart + offset;
						d_bitmap[bitIndex] = 1;
					}
				}
			}
		}
	}
}

__global__ static void generateInstanceTableGeneral_Kernel4(size_t instaceCountTable1, Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID)
{
	size_t TID = CudaGetTargetID();
	size_t idx1 = TID;
	TID = idx1 + startFrom;
	if (TID >= instaceCountTable1) {
		return;
	}
	Integer slotsValue = slots[TID];
	if (slotsValue == 0) {
		return;
	}

	size_t index;
	if (startFrom > 0) {
		size_t indexPos = indexes[startFrom];
		index = indexes[TID] - indexPos;
	}
	else {
		index = indexes[TID];
	}
	Integer degreek = degree - 1;
	SInteger Range = 1;
	Integer currentInstanceID = table1[TID*degreek];
	Integer currentCell = c_dainstanceCellIDs[currentInstanceID];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	for (SInteger j = -1; j <= Range; ++j)
	{
		for (SInteger i = -1; i <= Range; ++i)
		{
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells)
			{
				continue;
			}
			size_t lastFeatureCellIndex = NCID*c_maxFeaturesNum + lastFeatureID;
			size_t count = c_dacellEventInstanceCount[lastFeatureCellIndex];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[lastFeatureCellIndex];
				size_t endIndex = startIndex + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer lastInstanceID = c_dacellBasedSortedIndex[pos];
					Integer flag2 = 1;
					for (int degIndex = 0; degIndex < degreek; degIndex++) {
						currentInstanceID = table1[TID*degreek + degIndex];
						Real dist = device_distanceinMeters(currentInstanceID, lastInstanceID);
						if (dist > CONSTANT_PARAMETER.squaredDistanceThreshold) {
							flag2 = 0;
							break;
						}
					}
					if (flag2 == 1) {
						for (size_t idx4 = 0; idx4 < degreek; idx4++) {
							size_t idxCurrent = TID*degreek + idx4;
							Integer value = table1[idxCurrent];
							d_intermediate[index] = value;
							index++;
						}
					}
				}
			}
		}
	}
}

__global__ static void generalInstanceTableSlotCounter_Kernel(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID)
{
	size_t TID = CudaGetTargetID();
	if (TID>= instaceCountTable1) {
		return;
	}
	Integer degreek = degree - 1;
	size_t counter = 0;
	SInteger Range = 1;
	Integer currentInstanceID = table1[TID*degreek];
	Integer currentCell = c_dainstanceCellIDs[currentInstanceID];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	for (SInteger j = -1; j <= 1; ++j) {
		for (SInteger i = -1; i <= 1; ++i) {
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells) {
				continue;
			}
			size_t lastIndex = NCID*c_maxFeaturesNum + lastFeatureID;
			size_t count = c_dacellEventInstanceCount[lastIndex];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[lastIndex];
				size_t endIndex = startIndex + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer lastInstanceID = c_dacellBasedSortedIndex[pos];
					Integer flag2 = 1;
					for (int degIndex = 0; degIndex < degreek; degIndex++) {
						 Integer cIndex =TID*degreek + degIndex;
						currentInstanceID = table1[cIndex];
						Real dist = device_distanceinMeters(currentInstanceID, lastInstanceID);
						if (dist > CONSTANT_PARAMETER.squaredDistanceThreshold) {
							flag2 = 0;
							break;
						}
					}
					if (flag2 == 1) {
						counter++;
						slots[TID] = counter;
						size_t offset = 0;
						Integer bitIndex = 0;
						Integer fType;
						Integer value = 0;
						for (size_t idx4 = 0; idx4 < degreek; idx4++) {
							size_t idxCurrent = TID*degreek + idx4;
							value = table1[idxCurrent];
							fType = coloc[idx4];
							size_t fstart = c_dafeatureInstanceStart[fType];
							size_t fcount = c_dafeatureInstanceCount[fType];
							bitIndex = value - fstart + offset;
							offset += fcount;
							d_bitmap[bitIndex] = 1;
						}
						Integer lastStart = c_dafeatureInstanceStart[lastFeatureID];
						bitIndex = lastInstanceID - lastStart + offset;
						d_bitmap[bitIndex] = 1;
					}
				}
			}
		}
	}
}

__global__ static void generateInstanceTableGeneral_Kernel(size_t instaceCountTable1,Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID)
{
	size_t TID = CudaGetTargetID();
	size_t idx1 = TID;
	TID = idx1 + startFrom;
	if (TID >= instaceCountTable1) {
		return;
	}
	Integer slotsValue = slots[TID];
	if (slotsValue == 0) {
		return;
	}

	size_t index;
	if (startFrom > 0) {
		size_t indexPos = indexes[startFrom];
		index = indexes[TID] - indexPos;
	}
	else {
	//size_t index = indexes[TID];
		index = indexes[TID];
	}
	Integer degreek = degree - 1;
	SInteger Range = 1;
	Integer currentInstanceID = table1[TID*degreek];
	Integer currentCell = c_dainstanceCellIDs[currentInstanceID];
	Integer CIDY = currentCell / GRID_STRUCTURE.gridSize.x;
	Integer CIDX = currentCell - (GRID_STRUCTURE.gridSize.x*CIDY);
	for (SInteger j = -1; j <= Range; ++j)
	{
		for (SInteger i = -1; i <= Range; ++i)
		{
			SInteger NCID = getNeighborCellID(CIDX + i, CIDY + j, currentCell);
			if (NCID < 0 || NCID >= GRID_STRUCTURE.totalCells)
			{
				continue;
			}
			size_t lastFeatureCellIndex = NCID*c_maxFeaturesNum + lastFeatureID;
			size_t count = c_dacellEventInstanceCount[lastFeatureCellIndex];
			if (count > 0) {
				size_t startIndex = c_dacellEventInstanceStart[lastFeatureCellIndex];
				size_t endIndex = startIndex + count - 1;
				for (size_t pos = startIndex; pos <= endIndex; pos++) {
					Integer lastInstanceID = c_dacellBasedSortedIndex[pos];
					Integer flag2 = 1;
					for (int degIndex = 0; degIndex < degreek; degIndex++) {
						currentInstanceID = table1[TID*degreek + degIndex];
						Real dist = device_distanceinMeters(currentInstanceID, lastInstanceID);
						if (dist > CONSTANT_PARAMETER.squaredDistanceThreshold) {
							flag2 = 0;
							break;
						}
					}
					if (flag2 == 1) {
						for (size_t idx4 = 0; idx4 < degreek; idx4++) {
							size_t idxCurrent = TID*degreek + idx4;
							Integer value = table1[idxCurrent];
							d_intermediate[index] = value;
							index++;
						}
						d_intermediate[index] = lastInstanceID;
						index++;
					}
				}
			}
		}
	}
}

__global__ static void scalebyConstant_Kernel(Integer degree, Index64* indexes, size_t instaceCountTable1) {

	size_t TID = CudaGetTargetID();
	if (TID >= instaceCountTable1) {
		return;
	}
	size_t temp = indexes[TID];
	size_t value = temp * degree;
	indexes[TID] = value;
}

__device__ static void checkandUpdate(Integer degree, Integer* d_coloc, Integer* quadrant, Integer* countMap) {
	Integer check = 0;
	for (Integer i = 0; i < degree; i++) {
		Integer eventID = d_coloc[i];
		for (Integer j = 0; j < 4; j++) {
			Integer cellID = quadrant[j];
			Integer index = cellID*c_maxFeaturesNum + eventID;
			if (c_dacellEventInstanceCount[index]>0) {
				check++;
				break;
			}
		}
		if (check <= i) {
			break;
		}
	}
	if (check != degree) {
		return;
	}
	for (Integer i = 0; i < degree; i++) {
		for (Integer j = 0; j < 4; j++) {
			Integer cellID = quadrant[j];
			Integer eventID = d_coloc[i];
			Integer index = cellID*c_maxFeaturesNum + eventID;
			if (c_dacellEventInstanceCount[index]>0) {
				Integer value = c_dacellEventInstanceCount[index];
				countMap[i*GRID_STRUCTURE.totalCells + cellID] = value;
			}
		}
	}
}


__global__ static void filter_Kernel(int* isCellProcessed, SInteger* d_cellEventInstanceCount, Integer maxFeaturesNum, Integer* d_coloc, Integer degree, Integer *countMap) {

	__shared__ int s_cXId, s_cYId, cnt[16][2][2];
	__shared__ bool done, cellUpdate[2][2];
	s_cXId = blockIdx.x; 
	s_cYId = blockIdx.y;
	done = false;
	int cXId = s_cXId + threadIdx.y, cYId = s_cYId + threadIdx.z, cId = cYId * GRID_STRUCTURE.gridSize.x + cXId, fId;

	if(s_cYId >= GRID_STRUCTURE.gridSize.y - 1 || s_cXId >= GRID_STRUCTURE.gridSize.x - 1 || threadIdx.x >= degree)return;
	
	fId = d_coloc[threadIdx.x];
	int indx = cId * c_maxFeaturesNum + fId;

	cnt[threadIdx.x][threadIdx.y][threadIdx.z] = c_dacellEventInstanceCount[indx];
	//cnt[threadIdx.x][cXId][cYId] = d_cellEventInstanceCount[indx];

	__syncthreads();

	int s = 0;
	if(threadIdx.y ==0 && threadIdx.z ==0){
		s = cnt[threadIdx.x][0][0] + cnt[threadIdx.x][1][0] + cnt[threadIdx.x][0][1] + cnt[threadIdx.x][1][1];
		if(s == 0)done = true;
	}

	__syncthreads();

	if(done)return;	
	if(threadIdx.x == 0)cellUpdate[threadIdx.y][threadIdx.z] = atomicOr(isCellProcessed + cId, 1);
		
	__syncthreads();

	if(!cellUpdate[threadIdx.y][threadIdx.z])atomicAdd(countMap + threadIdx.x, cnt[threadIdx.x][threadIdx.y][threadIdx.z]);

	return;
}


__global__ static void filter2_Kernel(int* isCellProcessed, SInteger* d_cellEventInstanceCount, Integer maxFeaturesNum, Integer* coloc, 
					Integer cNum, Integer degree, Integer *countMap) {

	__shared__ int cnt[16][2][2], coId[16];
	__shared__ bool done, cellUpdate[2][2];
	done = false;

	if(blockIdx.z >= GRID_STRUCTURE.gridSize.y - 1 || blockIdx.y >= GRID_STRUCTURE.gridSize.x - 1 || threadIdx.x >= degree || blockIdx.x >= cNum)return;
	if(threadIdx.y == 0 && threadIdx.z == 0)coId[threadIdx.x] = *(coloc + blockIdx.x * degree + threadIdx.x);
	int cXId = blockIdx.y + threadIdx.y, cYId = blockIdx.z + threadIdx.z, cId = cYId * GRID_STRUCTURE.gridSize.x + cXId, fId;

	__syncthreads();

	fId = coId[threadIdx.x];
	int indx = cId * c_maxFeaturesNum + fId;

	//cnt[threadIdx.x][threadIdx.y][threadIdx.z] = c_dacellEventInstanceCount[indx];
	cnt[threadIdx.x][threadIdx.y][threadIdx.z] = d_cellEventInstanceCount[indx];

	__syncthreads();

	int s = 0;
	if(threadIdx.y ==0 && threadIdx.z ==0){
		if( !(cnt[threadIdx.x][0][0] + cnt[threadIdx.x][1][0] + cnt[threadIdx.x][0][1] + cnt[threadIdx.x][1][1]) )done = true;
	}

	__syncthreads();

	if(done)return;	
	if(threadIdx.x == 0)cellUpdate[threadIdx.y][threadIdx.z] = atomicOr(isCellProcessed + blockIdx.x * GRID_STRUCTURE.totalCells + cId, 1);
	//if(threadIdx.x == 0)cellUpdate[threadIdx.y][threadIdx.z] = atomicOr(isCellProcessed + blockIdx.x * GRID_STRUCTURE.totalCells + cId, 1);
		
	__syncthreads();

	if(!cellUpdate[threadIdx.y][threadIdx.z]){
		
		atomicAdd(countMap + blockIdx.x * degree + threadIdx.x, cnt[threadIdx.x][threadIdx.y][threadIdx.z]);
	}

	return;
}


__global__ static void filter2_2_Kernel(int* isCellProcessed, SInteger* d_cellEventInstanceCount, Integer maxFeaturesNum, Integer degree, Integer *countMap) {
	__shared__ int cnt[16][2][2], fSum[16];

	if(blockIdx.y >= GRID_STRUCTURE.gridSize.y - 1 || blockIdx.x >= GRID_STRUCTURE.gridSize.x - 1 || threadIdx.x >= maxFeaturesNum)return;

	int cId, indx;
	cId = (blockIdx.y + threadIdx.z) * GRID_STRUCTURE.gridSize.x + (blockIdx.x + threadIdx.y);
	indx = cId * maxFeaturesNum + threadIdx.x;

	//cnt[threadIdx.x][threadIdx.y][threadIdx.z] = c_dacellEventInstanceCount[indx];
	cnt[threadIdx.x][threadIdx.y][threadIdx.z] = d_cellEventInstanceCount[indx];

	__syncthreads();

	fSum[threadIdx.x] = cnt[threadIdx.x][0][0] + cnt[threadIdx.x][0][1] + cnt[threadIdx.x][1][0] + cnt[threadIdx.x][1][1];
	if(!fSum[threadIdx.x])return;
	
	__syncthreads();

	int magicNumber;
	bool done;
	for(int k = threadIdx.x + 1; k < maxFeaturesNum; k++){	
		if(!fSum[k])continue;
		done = atomicOr(isCellProcessed + (threadIdx.x * (maxFeaturesNum - threadIdx.x) + (threadIdx.x - 1) * threadIdx.x / 2) * GRID_STRUCTURE.totalCells + cId, 1);
		if(done)continue;
		atomicAdd(countMap + magicNumber * degree, cnt[threadIdx.x][threadIdx.y][threadIdx.z]);
		if(!cnt[k][threadIdx.y][threadIdx.z])continue;
		atomicAdd(countMap + magicNumber * degree + 1, cnt[k][threadIdx.y][threadIdx.z]);
	}

	return;
}

__global__ static void threshold_Kernel(Integer* countMap, Integer* featureInstanceCount, Integer* coloc, Real threshold, Integer degree, Integer colocNum){
	__shared__ Real PI[32];

	PI[threadIdx.x] = 1.0;

	if(blockIdx.x >= colocNum || threadIdx.x >= degree)return;

	PI[threadIdx.x] = *(countMap + blockIdx.x * degree + threadIdx.x) * 1.0 / (Real)*(featureInstanceCount + *(coloc + blockIdx.x * degree + threadIdx.x));

	__syncthreads();

	//if(threadIdx.x == 0)if(PI[1]<PI[0])PI[0] = PI[1];
	if(threadIdx.x == 0){
		for(int i=1;i<degree;i++)if(PI[0]>PI[i])PI[0]=PI[i];
	}


	/*int p = 1, shf = blockDim.x>>p;
	while(threadIdx.x + shf < blockDim.x>>(p-1)){
		if(PI[threadIdx.x] > PI[threadIdx.x + shf])PI[threadIdx.x] = PI[threadIdx.x + shf];
		shf = blockDim.x>>++p;

		__syncthreads();	
	}*/

	if(threadIdx.x == 0)*(countMap + blockIdx.x * degree) = PI[0] >= threshold? 1 : 0;
	//if(threadIdx.x == 0)*(countMap + blockIdx.x * degree) = PI[0] *1000;
	return;
}

__global__ static void getSelectionIndex_Kernel(size_t instaceCountTable2, Integer* d_selectionIndex, Integer degree, Integer* d_table2, size_t firstEventStartsFrom,Integer* d_checkIfIndex) {
	size_t TID = CudaGetTargetID();
	if (TID >= instaceCountTable2) {
		return;
	}
	size_t idx;
	size_t value;
	size_t relativeValue;
	idx = TID*(degree - 1);
	if (TID == 0) {
		value = d_table2[0];
		relativeValue = value - firstEventStartsFrom;
		d_selectionIndex[relativeValue] = TID;
		d_checkIfIndex[relativeValue] = 1;
		return;
	}
	size_t lowerIndex = TID*(degree-1) - (degree-1);
	size_t value0 = d_table2[lowerIndex];
	value = d_table2[idx];
	if (value0 == value) {
		return;
	}
	relativeValue = value - firstEventStartsFrom;
	d_selectionIndex[relativeValue] = TID;
	d_checkIfIndex[relativeValue] = 1;
}

__global__ static void initialization_Kernel(size_t size,Integer* selecetionIndex, Integer val) {
	size_t TID = CudaGetTargetID();
	if (TID >= size) {
		return;
	}
	selecetionIndex[TID] = val;
}





////////////// Added by Danial -----------------------
__device__ Integer getCellID(gridBox gridStructure, Real x, Real y) {
	Real RelativeX = x - gridStructure.computeZone.minBound.x;
	Real RelativeY = y - gridStructure.computeZone.minBound.y;

	RelativeX /= gridStructure.cellWidth;
	RelativeY /= gridStructure.cellWidth;
	Integer i = (Integer)RelativeX;
	Integer j = (Integer)RelativeY;
	const Integer CellID = (gridStructure.gridSize.x * j) + i;
	return CellID;
}



__global__ void kernel_CountCellFeatures(gridBox gridStructure, size_t fNum, size_t maxInstancesNum, Integer* instanceList,
	Real* instanceX, Real* instanceY, Integer* cellBasedSortedIndex, Integer* instanceCellId, Integer* instanceCellId2, SInteger* cellEventInstanceCount, SInteger* cellEventInstanceStart){
	
	int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if(pointIndex >= maxInstancesNum)return;

	cellBasedSortedIndex[pointIndex] = pointIndex;

	Integer cellID = getCellID(gridStructure, instanceX[pointIndex], instanceY[pointIndex]);

	instanceCellId[pointIndex] = cellID;
	instanceCellId2[pointIndex] = cellID;

	atomicInc( (unsigned int*) cellEventInstanceCount + cellID * fNum + instanceList[pointIndex], 1000000);
	atomicInc( (unsigned int*) cellEventInstanceStart + cellID * fNum + instanceList[pointIndex], 1000000);

	return;
}



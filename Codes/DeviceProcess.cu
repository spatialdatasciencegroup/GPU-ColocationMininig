#include <stdio.h>
#include "DataTypes.h"
#include "DeviceProcess.h"
#include "DeviceProcess_Kernel.h"
#include "Utility_Inline.h"
#include "includes.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#define USECDUASTREAM
//Round a / b to nearest higher integer value	
size_t iDivUp(size_t a, size_t b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
size_t getmin(Integer a, Integer b) {
	if (a < b) {
		return a;
	}
	return b;
}
class CThreadScaler
{
private:
	size_t Dg;
	size_t Db;
public:
	CThreadScaler(size_t NumThreads)
	{
		Db = getmin(BLOCK_MAX_DIM, NumThreads);
		if (Db > 0)
		{
			Dg = iDivUp(NumThreads, Db);
		}
		else
		{
			Dg = 0;
		}
	}
	size_t Grids()
	{
		return Dg;
	}
	size_t Blocks()
	{
		return Db;
	}
};
extern "C" {
	//degree2Processing
	StatusType degree2InstanceTableSlotCounter(size_t maxCount, Integer* slots, size_t start, size_t end, size_t secondstart, Integer* d_bitmap, Integer combiningFeatureID) {
		StatusType status;
		if (maxCount > 0) {
			CThreadScaler TS(maxCount);
			degree2TableSlotCounter_Kernel << < TS.Grids(), TS.Blocks() >> > (maxCount, slots, start, end, secondstart, d_bitmap, combiningFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			//CCT_ERROR_CHECK(StatusType);
			return status;
		}
		return ST_CUDAERR;
	}
	StatusType degree2InstanceTableGenerator(size_t maxCount, Integer*slots, Index64* indexes, Integer* d_intermediate, size_t start, size_t end, Integer combiningFeatureID) {
		StatusType status;
		if (maxCount > 0) {
			CThreadScaler TS(maxCount);
			degree2TableGenerator_Kernel << < TS.Grids(), TS.Blocks() >> > (slots, indexes, d_intermediate, start, end, combiningFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			//CCT_ERROR_CHECK(StatusType);
			return status;
		}
		return ST_CUDAERR;
	}

	//generalAnyDegreeProcessing
	StatusType generalInstanceTableSlotCounter(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID) {
		StatusType status;
		if (instaceCountTable1 > 0) {
			CThreadScaler TS(instaceCountTable1);
			generalInstanceTableSlotCounter_Kernel << < TS.Grids(), TS.Blocks() >> > (instaceCountTable1, slots, degree, table1, d_bitmap, coloc, lastFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			//CCT_ERROR_CHECK(StatusType);
			return status;
		}
		return ST_CUDAERR;
	}

	StatusType calcInstanceTableGeneral(size_t instaceCountTable1, Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID)
	{
		StatusType status;
		if (instaceCountTable1 > 0) {
			CThreadScaler TS(instaceCountTable1 - startFrom);
			generateInstanceTableGeneral_Kernel << < TS.Grids(), TS.Blocks() >> > (instaceCountTable1, slots, indexes, d_intermediate, degree, table1, startFrom, lastFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			//CCT_ERROR_CHECK(StatusType);
			return status;
		}
		return ST_CUDAERR;
	}


	Real getParticipationIndex(Integer*& d_bitmap, Integer degree, size_t i, vector<Integer> candiColocations, Integer*& featureInstanceStart, Integer*& featureInstanceEnd, Integer*& featureInstanceCount) {
		thrust::device_ptr<Integer> d_bitmapPtr = thrust::device_pointer_cast(d_bitmap);
		Real PI = 1.0;
		Integer from;
		Integer to;
		for (Integer j = 0; j < degree; j++) {
			Integer index = candiColocations[i*degree + j];
			Integer totalInstance = featureInstanceCount[index];
			if (j == 0) {
				from = 0;
				to = totalInstance;
			}
			else {
				from = to - 1;
				to = from + totalInstance + 1;
			}
			size_t bitSum = thrust::reduce(d_bitmapPtr + from, d_bitmapPtr + to - 1);
			Real pr = bitSum / (Real)totalInstance;
			if (pr < PI) {
				PI = pr;
			}
		}
		//thrust::device_free(d_bitmapPtr);
		return PI;
	}

	Real getUpperBoundPI(Integer degree, Integer*& d_countMap, Integer* h_coloc, Integer*& featureInstanceCount, Integer totalCells) {
		thrust::device_ptr<Integer> d_countMapPtr = thrust::device_pointer_cast(d_countMap);
		Real PI = 1.0;
		Integer from;
		Integer to;
		for (Integer j = 0; j < degree; j++) {
			Integer index = h_coloc[j];
			Integer totalInstance = featureInstanceCount[index];
			from = j*totalCells;
			to = (j + 1)*totalCells - 1;
			Integer bitSum = thrust::reduce(d_countMapPtr + from, d_countMapPtr + to);
			Real pr = bitSum / (Real)totalInstance;
			if (pr < PI) {
				PI = pr;
			}
		}
		//thrust::device_free(d_countMapPtr);
		return PI;
	}

	Integer getTotalInstances(Integer *slots, size_t table1InstanceCount) {
		Integer totalInstance = thrust::reduce(thrust::host, slots, slots + table1InstanceCount);
		//Integer totalInstance=0;
		//for(int i=0;i<table1InstanceCount;i++)totalInstance+=slots[i];
		return totalInstance;
	}

	Integer getTotalInstancesWithRange(Integer *slots, size_t startFrom, size_t endTo) {
		Integer totalInstance = thrust::reduce(thrust::host, slots + startFrom, slots + endTo);
		return totalInstance;
	}

	StatusType initializeDeviceMemConst(Integer maxInstacceNum, Integer maxFeatureNum, param parameter, Integer *instList, Integer *fetInsStart, Integer* fetInsEnd, Integer* fetInstCount, Real* insLocX, Real* insLocY, SInteger* d_cellEventInstanceCount, gridBox gridStructure, SInteger* d_cellEventInstanceStart, SInteger* d_cellEventInstanceEnd, Integer* d_instanceCellIDs, Integer* d_cellBasedSortedIndex) {
		StatusType status;
		status = CudaSafeCall(cudaMemcpyToSymbol(CONSTANT_PARAMETER, &parameter, sizeof(parameter)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dainstanceList, &instList, sizeof(instList)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dafeatureInstanceStart, &fetInsStart, sizeof(fetInsStart)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dafeatureInstanceEnd, &fetInsEnd, sizeof(fetInsEnd)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dafeatureInstanceCount, &fetInstCount, sizeof(fetInstCount)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dainstanceLocationX, &insLocX, sizeof(insLocX)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dainstanceLocationY, &insLocY, sizeof(insLocY)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_maxInstancesNum, &maxInstacceNum, sizeof(maxInstacceNum)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_maxFeaturesNum, &maxFeatureNum, sizeof(maxFeatureNum)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dacellEventInstanceCount, &d_cellEventInstanceCount, sizeof(d_cellEventInstanceCount)));
		status = CudaSafeCall(cudaMemcpyToSymbol(GRID_STRUCTURE, &gridStructure, sizeof(gridStructure)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dacellEventInstanceStart, &d_cellEventInstanceStart, sizeof(d_cellEventInstanceStart)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dacellEventInstanceEnd, &d_cellEventInstanceEnd, sizeof(d_cellEventInstanceEnd)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dainstanceCellIDs, &d_instanceCellIDs, sizeof(instList)));
		status = CudaSafeCall(cudaMemcpyToSymbol(c_dacellBasedSortedIndex, &d_cellBasedSortedIndex, sizeof(instList)));
		return status;
	}


	StatusType scalebyConstant(Integer degree, Index64* indexes, size_t instaceCountTable1) {
		StatusType status;
		if (instaceCountTable1 > 0) {
			CThreadScaler TS(instaceCountTable1);
			scalebyConstant_Kernel << < TS.Grids(), TS.Blocks() >> > (degree, indexes, instaceCountTable1);
			status = CudaSafeCall(cudaGetLastError());
			return status;
		}
		return ST_CUDAERR;
	}

	Real filter(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, Integer colocId, Integer maxFeaturesNum, Integer* featureInstanceCount, Integer* h_coloc, Integer* d_coloc, Integer degree, Integer* countMap, cudaStream_t strm) {
		Real PI = 1.0;
		//Should be relaxed later.
		dim3 gDim(gridStructure.gridSize.x, gridStructure.gridSize.y), bDim(16, 2, 2);
		filter_Kernel <<<gDim, bDim ,0 , strm>>> (isCellProcessed, d_cellEventInstanceCount, maxFeaturesNum, d_coloc, degree, countMap);
		Integer *cM=(Integer*)malloc(sizeof(Integer) * degree);
		//cudaMemcpyAsync(cM, countMap + colocId * degree, degree * sizeof(Integer), cudaMemcpyDeviceToHost, strm);
		cudaMemcpy(cM, countMap, degree * sizeof(Integer), cudaMemcpyDeviceToHost);
		//cudaStreamSynchronize(strm);
		int indx;
		Real tmp;
		// A B C D -> :
		for(int i = 0;i < degree;i++){
			indx = h_coloc[i];
			tmp = (cM[i] * 1.0) / featureInstanceCount[indx];
			if( tmp < PI)PI = tmp;
		}
		free(cM);
		return(PI);
	}

	void filter2(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, 
		Integer maxFeaturesNum, Integer* featureInstanceCount, Real threshold, Integer* d_coloc, Integer colocNum, Integer degree, Integer* countMap) {
		cudaError_t cuEr;
		//Should be relaxed later.
		dim3 gDim(colocNum, gridStructure.gridSize.x, gridStructure.gridSize.y), bDim(8, 2, 2);

		filter2_Kernel <<<gDim, bDim>>> (isCellProcessed, d_cellEventInstanceCount, maxFeaturesNum, d_coloc, colocNum, degree, countMap);

		cuEr = cudaDeviceSynchronize();
		if(cuEr != cudaSuccess){cout<<"\nError in filter 2 kernels.\n";exit(2);}

		threshold_Kernel <<<dim3(colocNum, 1, 1), dim3(16, 1, 1)>>> (countMap, featureInstanceCount, d_coloc, threshold, degree, colocNum);

		cuEr = cudaDeviceSynchronize();
		if(cuEr != cudaSuccess){cout<<"\nError in threshold kernel.\n";exit(2);}
		return;
	}

	void filter2_2(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, Integer maxFeaturesNum, 
			Integer* featureInstanceCount, Real threshold, Integer* coloc, Integer colocNum, Integer degree, Integer* countMap) {
		cudaError_t cuEr;
		dim3 gDim(gridStructure.gridSize.x, gridStructure.gridSize.y, 1), bDim(16, 2, 2);

		filter2_2_Kernel <<<gDim, bDim>>> (isCellProcessed, d_cellEventInstanceCount, maxFeaturesNum, degree, countMap);

		threshold_Kernel <<<colocNum, 32>>> (countMap, featureInstanceCount, coloc, threshold, degree, colocNum);

		cudaDeviceSynchronize();

		return;
	}

	StatusType getSelectionIndex(size_t instaceCountTable2, Integer* d_selectionIndex, Integer degree, Integer* d_table2, size_t firstEventStartsFrom, Integer* d_checkIfIndex) {
		StatusType status;
		if (instaceCountTable2 > 0) {
			CThreadScaler TS(instaceCountTable2);
			getSelectionIndex_Kernel << < TS.Grids(), TS.Blocks() >> > (instaceCountTable2, d_selectionIndex, degree, d_table2, firstEventStartsFrom, d_checkIfIndex);
			status = CudaSafeCall(cudaGetLastError());
			return status;
		}
		return ST_CUDAERR;

	}
	StatusType initialization(size_t size, Integer* selecetionIndex, Integer val) {
		StatusType status;
		if (size > 0) {
			CThreadScaler TS(size);
			initialization_Kernel << < TS.Grids(), TS.Blocks() >> > (size, selecetionIndex, val);
			status = CudaSafeCall(cudaGetLastError());
			return status;
		}
		return ST_CUDAERR;
	}

	void sortIndexByCellID(size_t size, Integer* cellBasedSortedIndex, Integer* instanceCellIDs)
	{
		Integer* temp;
		temp = new Integer[size];
		memcpy(temp, instanceCellIDs, size *sizeof(Integer));
		thrust::sort_by_key(thrust::host, temp, temp + size, cellBasedSortedIndex);
		delete[] temp;
	}



	//for test
	StatusType generalInstanceTableSlotCounter4(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID) {
		StatusType status;
		if (instaceCountTable1 > 0) {
			CThreadScaler TS(instaceCountTable1);
			generalInstanceTableSlotCounter_Kernel4 << < TS.Grids(), TS.Blocks() >> > (instaceCountTable1, slots, degree, table1, d_bitmap, coloc, lastFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			return status;
		}
		return ST_CUDAERR;
	}

	StatusType calcInstanceTableGeneral4(size_t instaceCountTable1, Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID)
	{
		StatusType status;
		if (instaceCountTable1 > 0) {
			CThreadScaler TS(instaceCountTable1 - startFrom);
			generateInstanceTableGeneral_Kernel4 << < TS.Grids(), TS.Blocks() >> > (instaceCountTable1, slots, indexes, d_intermediate, degree, table1, startFrom, lastFeatureID);
			status = CudaSafeCall(cudaGetLastError());
			return status;
		}
		return ST_CUDAERR;
	}



//------------- Added by Danial -------------------

	void CountCellFeatures(gridBox gridStructure, size_t fNum, size_t maxInstancesNum, Integer* d_instanceList, Real* d_instanceLocationX, Real* d_instanceLocationY, 
			Integer** d_cellBasedSortedIndex, Integer** d_instanceCellId, SInteger** d_cellEventInstanceCount, SInteger** d_cellEventInstanceStart){

		cudaError_t status = cudaMalloc((void**)d_cellBasedSortedIndex, maxInstancesNum * sizeof(Integer));
		status = cudaMalloc((void**)d_cellEventInstanceCount, fNum * sizeof(SInteger) * gridStructure.totalCells);
		status = cudaMalloc((void**)d_cellEventInstanceStart, fNum * sizeof(SInteger) * gridStructure.totalCells);
		status = cudaMalloc((void**)d_instanceCellId, sizeof(Integer) * maxInstancesNum);
		Integer* d_instanceCellId2;
		status = cudaMalloc((void**)&d_instanceCellId2, sizeof(Integer) * maxInstancesNum);
                if(status != cudaSuccess)printf("\nUnable to allocate device memory!\n");

		cudaMemset(*d_cellEventInstanceCount, 0,  gridStructure.totalCells * fNum * sizeof(SInteger));
		cudaMemset(*d_cellEventInstanceStart, 0,  gridStructure.totalCells * fNum * sizeof(SInteger));

       	 	dim3 blockSize(1024, 1, 1), gridSize(maxInstancesNum / 1000, 1, 1);	

		kernel_CountCellFeatures<<<gridSize, blockSize>>>(gridStructure, fNum, maxInstancesNum, d_instanceList, 
			d_instanceLocationX, d_instanceLocationY, *d_cellBasedSortedIndex, *d_instanceCellId, d_instanceCellId2, *d_cellEventInstanceCount, *d_cellEventInstanceStart);
	
		cudaDeviceSynchronize();

                thrust::device_ptr<Integer> d_ptr_val(*d_cellBasedSortedIndex);
                thrust::device_ptr<Integer> d_ptr_key(d_instanceCellId2);
                thrust::device_ptr<SInteger> d_ptr_start(*d_cellEventInstanceStart);


                //Integer *start0 = (Integer*) malloc(sizeof(Integer) * fNum * gridStructure.totalCells);
                //cudaMemcpy(start0, *d_cellEventInstanceStart, gridStructure.totalCells * fNum * sizeof(Integer), cudaMemcpyDeviceToHost);


		thrust::sort_by_key(d_ptr_key, d_ptr_key + maxInstancesNum, d_ptr_val);
		
  		thrust::exclusive_scan(d_ptr_start, d_ptr_start + gridStructure.totalCells * fNum, d_ptr_start);



                //Real *x = (Real*) malloc(sizeof(Real) * maxInstancesNum);
                //cudaMemcpy(x, d_instanceLocationX, maxInstancesNum * sizeof(Real) , cudaMemcpyDeviceToHost);
                //Real *y = (Real*) malloc(sizeof(Real) * maxInstancesNum);
                //cudaMemcpy(y, d_instanceLocationY, maxInstancesNum * sizeof(Real) , cudaMemcpyDeviceToHost);
		//for(int i=0;i<maxInstancesNum;i++){cout<<x[i]<<" , "<<y[i]<<endl;}

                //Integer *start = (Integer*) malloc(sizeof(Integer) * fNum * gridStructure.totalCells);
                //cudaMemcpy(start, *d_cellEventInstanceStart, gridStructure.totalCells * fNum * sizeof(Integer), cudaMemcpyDeviceToHost);

                //Integer *val = (Integer*) malloc(sizeof(Integer) * maxInstancesNum);
                //cudaMemcpy(val, *d_cellBasedSortedIndex, maxInstancesNum * sizeof(Integer), cudaMemcpyDeviceToHost);

                //Integer *key = (Integer*) malloc(sizeof(Integer) * maxInstancesNum);
                //cudaMemcpy(key, *d_instanceCellId, maxInstancesNum * sizeof(Integer), cudaMemcpyDeviceToHost);

		//for(int i=0;i<maxInstancesNum;i++){cout<<val[i]<<" , "<<key[i]<<endl;}
		//for(int i=0;i<gridStructure.totalCells * fNum;i++){cout<<i<<" , "<<start[i]<<endl;}
//exit(0);

		cudaFree(d_instanceCellId2);
		return;
	}




}








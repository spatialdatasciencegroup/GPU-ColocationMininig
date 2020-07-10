#pragma once
#define PARAMETER "CONSTANT_PARAMETER"

extern "C" {
	StatusType degree2InstanceTableSlotCounter(size_t maxCount, Integer* slots, size_t start, size_t end, size_t secondstart, Integer* d_bitmap, Integer combiningFeatureID);
	StatusType degree2InstanceTableGenerator(size_t maxCount, Integer*slots, Index64* indexes, Integer* d_intermediate, size_t start, size_t end, Integer combiningFeatureID);

	StatusType generalInstanceTableSlotCounter(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID);
	StatusType calcInstanceTableGeneral(size_t instaceCountTable1, Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID);

	StatusType initializeDeviceMemConst(Integer maxInstacceNum, Integer maxFeatureNum, param parameter, Integer *instList, Integer *fetInsStart, Integer* fetInsEnd, Integer* fetInstCount, Real* insLocX, Real* insLocY, SInteger* d_cellEventInstanceCount, gridBox gridStructure, SInteger* d_cellEventInstanceStart, SInteger* d_cellEventInstanceEnd, Integer* d_instanceCellIDs, Integer* d_cellBasedSortedIndex);
	Real getParticipationIndex(Integer*& d_bitmap, Integer degree, size_t i, vector<Integer> candiColocations, Integer*& featureInstanceStart, Integer*& featureInstanceEnd, Integer*& featureInstanceCount);
	Integer getTotalInstances(Integer *slots, size_t table1InstanceCount);
	Integer getTotalInstancesWithRange(Integer *slots, size_t startFrom, size_t endTo);
	StatusType scalebyConstant(Integer degree, Index64* indexes, size_t instaceCountTable1);
	//StatusType filter(Integer maxFeaturesNum, Integer totalCells, Integer* d_coloc, Integer degree, Integer* countMap);
	Real filter(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, Integer colocId, Integer maxFeaturesNum, Integer* featureInstanceCount, Integer*h_coloc, Integer* d_coloc, Integer degree, Integer* countMap, cudaStream_t strm);
	void filter2(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, Integer maxFeaturesNum, 
		Integer* d_featureInstanceCount, Real PIthreshold, Integer* d_coloc, Integer candiColocCounter, Integer degree, Integer* countMap);
	void filter2_2(int* isCellProcessed, SInteger* d_cellEventInstanceCount, gridBox gridStructure, Integer maxFeaturesNum, 
		Integer* d_featureInstanceCount, Real PIthreshold, Integer* d_coloc, Integer candiColocCounter, Integer degree, Integer* countMap);
	Real getUpperBoundPI(Integer degree, Integer*& d_countMap, Integer* h_coloc, Integer*& featureInstanceCount, Integer totalCells);

	StatusType getSelectionIndex(size_t instaceCountTable2, Integer* d_selectionIndex, Integer degree, Integer* d_table2, size_t firstEventStartsFrom, Integer* d_checkIfIndex);
	StatusType initialization(size_t size, Integer* selecetionIndex, Integer val);
	void sortIndexByCellID(size_t size, Integer* cellBasedSortedIndex, Integer* instanceCellIDs);

	//for test
	StatusType generalInstanceTableSlotCounter4(size_t instaceCountTable1, Integer* slots, Integer degree, Integer* table1, Integer* d_bitmap, Integer* coloc, Integer lastFeatureID);
	StatusType calcInstanceTableGeneral4(size_t instaceCountTable1, Integer* slots, Index64* indexes, Integer* d_intermediate, Integer degree, Integer* table1, size_t startFrom, Integer lastFeatureID);

	//By Danial
	void CountCellFeatures(gridBox gridStructure, size_t fNum, size_t maxInstancesNum, Integer* d_instanceList, Real* d_instanceLocationX, Real* d_instanceLocationY, 
		Integer** d_cellBasedSortedIndex, Integer** d_instanceCellId, SInteger** d_cellEventInstanceCount, SInteger** d_cellEventInstanceStart);
	
}

#pragma once
#include "DataTypes.h"
#include <limits>
#include <string>
#include <chrono>
#include <math.h>
#include <iomanip>
#include <ctime>
using  sec = chrono::seconds;
using get_time = chrono::steady_clock;
class colocationFinder {

public:
	colocationFinder(void);

public:
	~colocationFinder(void);

private:
	// -------------------------------- Timing Variables -------------------------
	Real diff_Time, total_Time, io_Time, transfer_Time, grid_Time, filter_Time, refinement_Time;
	cudaEvent_t startPoint, endPoint;
	// ---------------------------------------------------------------------------
	
	//Host Variables
	size_t maxInstancesNum;
	size_t maxFeaturesNum;
	size_t maxCellNum;
	std::string location;
	vector <string> featureTypes;   //done
	Integer * instanceList;         //done
	Integer *featureInstanceStart;  //done
	Integer * featureInstanceEnd;	//done
	Integer * featureInstanceCount; //done
	Real * instanceLocationX;    //done
	Real * instanceLocationY;    //done
	Integer candiColocCounter;
	struct param parameter;
	//for subset
	vector<Integer> subset;
	vector<vector<Integer>> subsetList;
	vector<Integer> h_candiColocations;
	vector<Integer> kplus2CandiColocation;
	Integer *d_candiColocations;
	vector<Integer> tempColoc;

	Integer** h_prevalentInstanceTable;
	vector<Integer> h_prevalantColocations;
	Integer h_prevelentColocationCount;
	vector<Integer> h_prevalentInstanceTableSize;

	Integer** h_prevalentInstanceTable2;
	vector<Integer> h_prevalantColocations2;
	Integer h_prevelentColocationCount2;
	vector<Integer> h_prevalentInstanceTableSize2;
	vector<Integer> prevalanceCheck;
	Integer tempOldPrevalentColocCount;

	//added in version 2
	struct gridBox gridStructure;
	SInteger* cellEventInstanceCount;
	Integer * instanceCellIDs;

	//added in version 3
	SInteger* cellEventInstanceStart;
	SInteger* cellEventInstanceEnd;
	Integer* cellBasedSortedIndex;

	//for MRF
	vector<Integer> coarseInstanceStart;  //done
	vector<Integer> coarseInstanceEnd;	//done
	vector<Integer> coarseInstanceList;
	vector<vector<Integer>> coarseCandidateInstanceTables;
	vector<Integer> coarsePrevalance;
	Integer ** coarsePrevalentInstanceTables;
	Integer *coarsePrevalentInstanceTableSize;

	vector<Integer> coarseInstanceCount;

	//CUDA Device variables
	int *d_isCellProcessed;
	Integer *d_instanceList;
	Integer *d_featureInstanceStart;
	Integer *d_featureInstanceEnd;
	Integer *d_featureInstanceCount;
	Real	*d_instanceLocationX;
	Real	*d_instanceLocationY;
	SInteger* d_cellEventInstanceCount;
	cudaStream_t m_CudaStream;

	SInteger* d_cellEventInstanceEnd;
	Integer* d_instanceCellIDs;
	Integer* d_cellBasedSortedIndex;
	SInteger* d_cellEventInstanceStart;
	//device_vector<Integer> d_cellBasedSortedIndex;
	//device_vector<SInteger> d_cellEventInstanceStart;
        

	//LOG VARIABLES
	std::string logFileName;
	std::ofstream timeLogStream;
	std::string compLogFileName;
	std::ofstream compLogStream;
	//std::string outputLocation;
	Real totalMemUsed;
	Real memoryTracker;
	std::string outputLocation;
	std::string outputFile;
	Integer totalCandidatePatterns;
	Integer totalFilteredPatterns;
	Real totalFilterTime;
	Real totalRefineTime;
	Real degree2FilterTime;

	bool needInstanceTable;
	Integer** h_prevalentSlotCounts;
	vector<Integer> hasInstanceTable;

	Integer *d_coloc;
	Integer *h_coloc;
	Integer *d_slotCounts;
	Index64 *h_Indexes;
	Index64 *d_Indexes;
	Index64 instanceMax;
	Integer *d_intermediate;
	Integer estimatePrevalent;
	Integer *d_table1;
	Integer slotLimit;
	Integer intermediateMax;
	Integer table1Max;
	Integer mapMax;
	Integer *d_countMap;
	Integer *d_countMap2;
	Integer *d_bitmap;

	Real totalKernelTime;
public:
	void Begin(Integer argc, char**argv);
	void candidateColocationGeneral(Integer degree);
	void populateData(std::string datasetFilename);
	void degree2CandidateColocationGenerator();
	bool generateandCheckSubsets(vector<Integer> &inter, Integer degree);
	void subsetGen(vector<Integer> &inter, Integer k, Integer n, Integer idx);
	bool checkSubset(vector<Integer> subsetElem);
	void generateInstanceTableGeneral(Integer degree);
	void degree2Processing();
	void clearSubsetVectors();
	Integer getIndex(vector <Integer> inner, Integer degree);
	Integer getIndex2(vector <Integer> inner, Integer degree);
	void resetPrevalentData(Integer degree);
	void clearMemory();
	void copyPrevalentColocations();
	void LoadParameter(std::string parameterFileName);


	//new added in version 2
	void setGrid(Real maxX, Real maxY, Real minX, Real minY);
	Integer getCellID(Real lon, Real lat);
	
	//CUDA FUNCTIONS
	StatusType CudaSafeCall(cudaError_t Status);
	StatusType initializeDeviceMemory();
	StatusType initializeDevicConstantMemory();

	//log related functions
	void timeLog(Integer degree, std::string function, std::string eventType, Integer duration);
	void loggingFileOpenFunction();
	void loggingFileCloseFunction();
	void compLog(Integer i, Integer degree, Integer candicolocNumber,std::string Remark, std::string PIType, Real PI);
	void compLog(Integer degree, std::string function);
	void compLog(Integer degree, Integer i, std::string totalmem, Integer candicolocNumber);
	void savetoFile2(Integer degree);
	void savetoFileGen(Integer degree);
	
	void generatePrevalentPatternsGeneral(Integer degree);
	void calcInstanceTablePatternIndexes(Integer degree);

	void tableGenRequired(Integer degree);
	bool generateandCheckSubsets2(vector<Integer> &inter, Integer degree);
	bool checkSubset2(vector<Integer> subsetElem);
	void kplus2ColocationGeneral(Integer degree);
	bool checkSubsetkplus2(vector<Integer> subsetElem);
	bool generateandCheckSubsetskplus2(vector<Integer> &inter, Integer degree);
}; 

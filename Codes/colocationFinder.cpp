#pragma warning(disable : 4996)
#include "includes.h"
#include "colocationFinder.h"
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/compare.hpp>
#include "DeviceProcess.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <algorithm>  
#include <iterator>  

#define PRINT_TIMES 0

using namespace boost;
using namespace std;

colocationFinder::colocationFinder(void) {
	cudaDeviceReset();
	total_Time = io_Time = transfer_Time = grid_Time = filter_Time = refinement_Time = 0;
        cudaEventCreate(&startPoint);
        cudaEventCreate(&endPoint);
	totalCandidatePatterns = 0;
	totalFilteredPatterns = 0;
	totalFilterTime = 0.0;
	totalRefineTime = 0.0;
	degree2FilterTime = 0.0;
	cudaMalloc((void**)&d_coloc, 10*sizeof(Integer));
	h_coloc = new Integer[10];
	instanceMax = 30000000;
	table1Max = instanceMax * 5;
	intermediateMax = 120000000 * 10;
	slotLimit = 3500000;
	h_Indexes = (Index64*) malloc(instanceMax*sizeof(Index64));
	cudaMalloc((void**)&d_Indexes, instanceMax *sizeof(Index64));
	cudaMalloc((void**)&d_slotCounts, instanceMax*sizeof(Integer));
	cudaMalloc((void**)&d_intermediate, intermediateMax*sizeof(Integer));
	cudaMalloc((void**)&d_table1, table1Max*sizeof(Integer));
}
colocationFinder::~colocationFinder(void) {
	featureTypes.clear();
	featureTypes.shrink_to_fit();
	free(instanceList);
	delete[] featureInstanceStart;
	delete[] featureInstanceEnd;
	delete[] featureInstanceCount;
	free(instanceLocationX);
	free(instanceLocationY);
	h_candiColocations.clear();
	h_candiColocations.shrink_to_fit();
	cudaFree(d_candiColocations);

}
void colocationFinder::LoadParameter(std::string parameterFileName) {
	std::string FileName = /*location+*/ parameterFileName;
	std::ifstream ifs(FileName.c_str(), std::ifstream::in);//FileName.c_str());
	if (false == ifs.is_open())
	{
		std::cout << "Parameter file missing..." << std::endl;
		exit(0);
		//return CCT_FILEERR;
	}
	ifs >> parameter.thresholdDistance;
	ifs >> parameter.PIthreshold;
	ifs >> parameter.FilterON_OFF;
	ifs.close();

	parameter.DEG_TO_RAD = DEG_TO_RAD;
	parameter.EARTH_RADIUS_IN_METERS = EARTH_RADIUS_IN_METERS;
	parameter.squaredDistanceThreshold = parameter.thresholdDistance* parameter.thresholdDistance;
}
void colocationFinder::degree2CandidateColocationGenerator() {
	for (Integer i = 0; i < featureTypes.size() - 1; i++) {
		for (Integer j = i + 1; j < featureTypes.size(); j++) {
			h_candiColocations.push_back(i);
			h_candiColocations.push_back(j);
		}
	}
}
Integer mode(Integer a[], Integer n) {
	Integer maxValue = 0, maxCount = 0, i, j;

	for (i = 0; i < n; ++i) {
		Integer count = 0;

		for (j = 0; j < n; ++j) {
			if (a[j] == a[i])
				++count;
		}

		if (count > maxCount) {
			maxCount = count;
			maxValue = a[i];
		}
	}

	return maxValue;
}
void colocationFinder::setGrid(Real maxX, Real maxY, Real minX, Real minY) {
//We just need cellEventInstanceCount
//             cellEventInstanceStart
	gridStructure.computeZone.minBound.x = minX;
	gridStructure.computeZone.minBound.y = minY;
	gridStructure.computeZone.maxBound.x = maxX;
	gridStructure.computeZone.maxBound.y = maxY + parameter.thresholdDistance;
	gridStructure.cellWidth = parameter.thresholdDistance;
	scalar2 gs;
	gs.x = (gridStructure.computeZone.maxBound.x - gridStructure.computeZone.minBound.x) / gridStructure.cellWidth;
	gs.y = (gridStructure.computeZone.maxBound.y - gridStructure.computeZone.minBound.y) / gridStructure.cellWidth;
	gridStructure.gridSize.x = (Integer)ceil(gs.x);
	gridStructure.gridSize.y = (Integer)ceil(gs.y);
	gridStructure.totalCells = gridStructure.gridSize.x * gridStructure.gridSize.y;
	Integer cSize = gridStructure.totalCells*maxFeaturesNum;
	cellEventInstanceCount = (SInteger*) malloc(cSize*sizeof(SInteger));
	instanceCellIDs = (Integer*) malloc(maxInstancesNum*sizeof(Integer));
	cellBasedSortedIndex = (Integer*) malloc(maxInstancesNum*sizeof(Integer));

        
        CountCellFeatures(gridStructure, maxFeaturesNum, maxInstancesNum, d_instanceList, d_instanceLocationX, d_instanceLocationY, 
		&d_cellBasedSortedIndex, &d_instanceCellIDs, &d_cellEventInstanceCount, &d_cellEventInstanceStart);


}
Integer colocationFinder::getCellID(Real x, Real y) {
	Real RelativeX = x - gridStructure.computeZone.minBound.x;
	Real RelativeY = y - gridStructure.computeZone.minBound.y;

	RelativeX /= gridStructure.cellWidth;
	RelativeY /= gridStructure.cellWidth;
	Integer i = (Integer)RelativeX;
	Integer j = (Integer)RelativeY;
	const Integer CellID = (gridStructure.gridSize.x * j) + i;
	return CellID;
}

void colocationFinder::populateData(std::string datasetFilename) { //pass data.csv as argument

        cudaEventRecord(startPoint, 0);

	//variable decleration and defination
	vector<string> vec;
	string line;
	size_t entryCounter = 0;
	size_t eventTypeCounter = 0;
	size_t instanceCounter = 0;
	size_t lastEventID = -1;
	vector<struct sFeatureStats> featureStats;
	Real maxX, maxY, minX, minY;
	string data(/*location +*/ datasetFilename);

	ifstream in(data.c_str());

	typedef tokenizer< escaped_list_separator<char> > Tokenizer;
	//getline(in, line); // for header
	getline(in, line); // for instance count
	maxInstancesNum = stoi(line);
	instanceLocationX = new Real[maxInstancesNum];
	instanceLocationY = new Real[maxInstancesNum];
	instanceList = new Integer[maxInstancesNum];
	getline(in, line);
	while (!line.empty())
	{
		Tokenizer tok(line);
		vec.assign(tok.begin(), tok.end());
		if (lastEventID == -1) {
			lastEventID = stoi(vec[0]);
			featureTypes.push_back(vec[0]);
			eventTypeCounter++;
		}
		else if (stoi(vec[0]) != lastEventID) {
			struct sFeatureStats tempStat;
			tempStat.start = entryCounter - instanceCounter;
			tempStat.end = entryCounter - 1;
			tempStat.count = instanceCounter;
			featureStats.push_back(tempStat);
			lastEventID = stoi(vec[0]);
			instanceCounter = 0;
			featureTypes.push_back(vec[0]);
			eventTypeCounter++;
		}
		instanceList[entryCounter] = eventTypeCounter - 1;  //hold the feature id of each instances... all the instances are sorted by type before read.
		instanceLocationX[entryCounter] = stof(vec[1]);
		instanceLocationY[entryCounter] = stof(vec[2]);
		entryCounter++;
		instanceCounter++;
		getline(in, line);
	}
	getline(in, line);
	Tokenizer tok(line);
	vec.assign(tok.begin(), tok.end());
	minX = stof(vec[0]);
	minY = stof(vec[1]);
	maxX = stof(vec[2]);
	maxY = stof(vec[3]);
	//for last feature
	

        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&io_Time, startPoint, endPoint);
	if(PRINT_TIMES)std::cout << endl << "IO time: " << io_Time << " miliseconds " << endl;


	struct sFeatureStats tempStat;
	tempStat.start = entryCounter - instanceCounter;
	tempStat.end = entryCounter - 1;
	tempStat.count = instanceCounter;
	featureStats.push_back(tempStat);
	

        cudaEventRecord(startPoint, 0);

	//converting vector to array
	Integer featureSize = featureStats.size();
	featureInstanceStart = new Integer[featureSize];
	featureInstanceEnd = new Integer[featureSize];
	featureInstanceCount = new Integer[featureSize];
	Integer MaxCount= 0; 
	for (size_t i = 0; i < featureStats.size(); i++) {
		featureInstanceStart[i] = featureStats[i].start;
		featureInstanceEnd[i] = featureStats[i].end;
		featureInstanceCount[i] = featureStats[i].count;
		if (featureInstanceCount[i] > MaxCount) {
			MaxCount = featureInstanceCount[i];
		}
	}
	mapMax = MaxCount * 8;
	cudaMalloc((void**)&d_bitmap, mapMax*sizeof(Integer));

	maxFeaturesNum = featureSize;
	in.close();

	initializeDeviceMemory();

        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&transfer_Time, startPoint, endPoint);
	if(PRINT_TIMES)std::cout << endl << "Initial CPU to GPU transfer time :  " << transfer_Time << " miliseconds " << endl;


        cudaEventRecord(startPoint, 0);

	setGrid(maxX, maxY, minX, minY);


        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&grid_Time, startPoint, endPoint);
	if(PRINT_TIMES)std::cout << endl << "Grid processing time :  " << grid_Time << " miliseconds " << endl;

}

void colocationFinder::degree2Processing() {
	cudaEventRecord(startPoint, 0);

	StatusType status;
	candiColocCounter = 0;
	memoryTracker = 0;
	vector<Real> upperboundList;
	vector<Real> PIList;
    candiColocCounter = featureTypes.size() * (featureTypes.size() - 1) / 2;

	Integer degree = 2;
	totalCandidatePatterns += candiColocCounter;
	degree2CandidateColocationGenerator();
	
	Integer pSlotCounter = 0;
	h_prevalentSlotCounts = (Integer**)malloc((candiColocCounter)*sizeof(Integer*));
	h_prevalentSlotCounts[0] = (Integer*)malloc(candiColocCounter * slotLimit * sizeof(Integer));

	status = initializeDevicConstantMemory();
	//cudaDeviceSynchronize();
	h_prevelentColocationCount = 0;
	Integer filterpruneCounter = 0;


	cudaError_t cuEr;
	//++++ h_candiColocations: It has all the patterns of cardinality equal to degree.
	cuEr = cudaMalloc((void**)&d_candiColocations, candiColocCounter * degree * sizeof(Integer));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_candiColocations.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	cudaMemcpy(d_candiColocations, h_candiColocations.data(), candiColocCounter * degree * sizeof(Integer), cudaMemcpyHostToDevice);
	//++++ d_countMap2: For each given pattern i, it has degree integers counting number of participating instances of each feature. size: degree * nChoosek(n, 2)
    	cuEr = cudaMalloc((void**)&d_countMap2, candiColocCounter * degree * sizeof(Integer));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_countMap2.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	cuEr = cudaMemset(d_countMap2, 0, candiColocCounter * degree * sizeof(Integer));
	//++++ d_slotCounts: for each (A, B) pattern is of size |A| and i-th elements counts the number of instances that has i-th instance of A in it. 
	cudaMalloc((void**)&d_slotCounts, instanceMax*sizeof(Integer));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_slotCounts.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	//++++ d_isCellProcessed: It makes sure each instance in each cell is counted only once in each pattern.
	cuEr = cudaMalloc((void**)&d_isCellProcessed, candiColocCounter * gridStructure.totalCells * sizeof(int));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_isCellProcessed.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	cudaMemset(d_isCellProcessed, 0, candiColocCounter * gridStructure.totalCells * sizeof(int));

	filter2_2(d_isCellProcessed, d_cellEventInstanceCount, gridStructure, maxFeaturesNum, d_featureInstanceCount, parameter.PIthreshold, 
		d_candiColocations, candiColocCounter, degree, d_countMap2);	
	
    	Integer *countMap2 = (Integer*) malloc(candiColocCounter * degree * sizeof(Integer));
	cudaMemcpy(countMap2, d_countMap2, candiColocCounter * degree * sizeof(Integer), cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();
        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&diff_Time, startPoint, endPoint);
	filter_Time += diff_Time;
	if(PRINT_TIMES)cout<<"\ncolocNum: "<<candiColocCounter<<", Degree: "<<degree<<", Filter time: "<<diff_Time;

	cudaFree(d_countMap2);
	cudaFree(d_isCellProcessed);

	cudaEventRecord(startPoint, 0);

	for (size_t i = 0; i < candiColocCounter; i++) {
		if( !*(countMap2 + i * degree) )continue;
		size_t firstFeatureIndex = h_candiColocations[i*degree];
		size_t secondFeatureIndex = h_candiColocations[i*degree + 1];
		size_t start = featureInstanceStart[firstFeatureIndex];
		size_t end = featureInstanceEnd[firstFeatureIndex];
		size_t secondstart = featureInstanceStart[secondFeatureIndex];

		size_t totalInstances = 0;
		size_t table1InstanceCount = end - start + 1;
		if(table1InstanceCount > instanceMax){
			cudaFree(d_slotCounts);
			status = CudaSafeCall(cudaMalloc((void**)&d_slotCounts, table1InstanceCount*sizeof(Integer)));
		}
		status = CudaSafeCall(cudaMemset(d_slotCounts, 0, table1InstanceCount*sizeof(Integer)));
		size_t bitmapSize = featureInstanceCount[firstFeatureIndex] + featureInstanceCount[secondFeatureIndex];
		status = CudaSafeCall(cudaMemset(d_bitmap, 0, bitmapSize*sizeof(Integer)));
		status = degree2InstanceTableSlotCounter(table1InstanceCount, d_slotCounts, start, end, secondstart, d_bitmap, secondFeatureIndex);
		Real PI = getParticipationIndex(d_bitmap, degree, i, h_candiColocations, featureInstanceStart, featureInstanceEnd, featureInstanceCount);

		if (PI < parameter.PIthreshold)continue;
		if (table1InstanceCount + pSlotCounter >= candiColocCounter * slotLimit) {
			cout<<endl<<"Not enough memory for h_prevalentColocationCount"<<endl;
			exit(1);
			//h_prevalentSlotCounts[h_prevelentColocationCount]= (Integer*) malloc(table1InstanceCount*sizeof(Integer));
		}
		h_prevalentSlotCounts[h_prevelentColocationCount]= h_prevalentSlotCounts[0] + pSlotCounter;
		pSlotCounter += table1InstanceCount;

		cudaMemcpy(h_prevalentSlotCounts[h_prevelentColocationCount], d_slotCounts, table1InstanceCount*sizeof(Integer), cudaMemcpyDeviceToHost);
		h_prevalantColocations.push_back(firstFeatureIndex);
		h_prevalantColocations.push_back(secondFeatureIndex);
		h_prevelentColocationCount++;
	}
	hasInstanceTable.clear();
	hasInstanceTable.shrink_to_fit();
	// (A B) (A C) (B C) ( D E) degree=2  h_prevelentColocationCount = 3
	// degree =3 -  >  (A B C)      hasInstanceTable[3]
	for (Integer i = 0; i < h_prevelentColocationCount; i++) {
		hasInstanceTable.push_back(0);
	}
	candidateColocationGeneral(degree+1);
	if (candiColocCounter == 0) {
		needInstanceTable = false;
		totalFilteredPatterns += filterpruneCounter;
		return;
	}
	Integer degkplus1 = degree + 1;
	// From this point on degkplus1 is 3 and h_candiColocations holds all the degree 3 possible paterns
	for (Integer i = 0; i < candiColocCounter; i++) {
		vector<Integer> combiColoc1;
		for (size_t t = 0; t < degkplus1 - 1; t++)combiColoc1.push_back(h_candiColocations[i*degkplus1 + t]); 
				//t participating feature in i-th colocation pattern of degree degkplus1   (A B C, D E F) i=1 t=1 it retures 'E'
				//i = 0 : combiColoc1 = (A B) -> 0 
				//i = 1 : combiColoc1 = (D E) -> 3 
		Integer table1Index = getIndex(combiColoc1, degkplus1 - 1);
		hasInstanceTable[table1Index] = 1;
	}
	h_prevalentInstanceTable = (Integer**)malloc(h_prevelentColocationCount*sizeof(Integer*));
	for (size_t i = 0; i < h_prevelentColocationCount; i++) {
		if (hasInstanceTable[i] == 1) {
			size_t firstFeatureIndex = h_prevalantColocations[i*degree];
			size_t secondFeatureIndex = h_prevalantColocations[i*degree + 1];
			size_t start = featureInstanceStart[firstFeatureIndex];
			size_t end = featureInstanceEnd[firstFeatureIndex];
			size_t secondstart = featureInstanceStart[secondFeatureIndex];
			size_t table1InstanceCount = end - start + 1;
			if (table1InstanceCount > instanceMax) {
				cudaFree(d_Indexes);
				//cudaFreeHost(h_Indexes);
				free(h_Indexes);
				//cudaMallocHost((void**)&h_Indexes, table1InstanceCount*sizeof(Index64));
				h_Indexes = (Index64*) malloc(table1InstanceCount*sizeof(Index64));
				status = CudaSafeCall(cudaMalloc((void**)&d_Indexes, table1InstanceCount*sizeof(Index64)));
			}

			thrust::exclusive_scan(h_prevalentSlotCounts[i], h_prevalentSlotCounts[i] + table1InstanceCount, h_Indexes);
			Integer totalInstances = getTotalInstances(h_prevalentSlotCounts[i], table1InstanceCount);

			size_t totalSize = totalInstances*degree;
			memoryTracker += (Real)(totalSize *4.0) / (1024 * 1024 * 1024);
			if (table1InstanceCount > instanceMax) {
				cudaFree(d_slotCounts);
				status = CudaSafeCall(cudaMalloc((void**)&d_slotCounts, table1InstanceCount*sizeof(Integer)));
			}
			status = CudaSafeCall(cudaMemcpy(d_Indexes, h_Indexes, table1InstanceCount*sizeof(Index64), cudaMemcpyHostToDevice));
			cudaDeviceSynchronize();
			status = scalebyConstant(degree, d_Indexes, table1InstanceCount);
			cudaDeviceSynchronize();
			status = CudaSafeCall(cudaMemcpy(d_slotCounts, h_prevalentSlotCounts[i], table1InstanceCount*sizeof(Integer), cudaMemcpyHostToDevice));
			status = degree2InstanceTableGenerator(table1InstanceCount, d_slotCounts, d_Indexes, d_intermediate, start, end, secondFeatureIndex);
			h_prevalentInstanceTableSize.push_back(totalInstances);
			//cudaMallocHost((void**)&h_prevalentInstanceTable[i], totalInstances*degree*sizeof(Integer));
			h_prevalentInstanceTable[i] = (Integer*) malloc(totalInstances*degree*sizeof(Integer));
			status = CudaSafeCall(cudaMemcpy(h_prevalentInstanceTable[i], d_intermediate, totalInstances*degree*sizeof(Integer), cudaMemcpyDeviceToHost));
		}
		else {
			h_prevalentInstanceTableSize.push_back(0);
		}
	}

	free(countMap2);
	totalFilteredPatterns += filterpruneCounter;

        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&diff_Time, startPoint, endPoint);
	refinement_Time += diff_Time;
	if(PRINT_TIMES)cout<<", Refinement time: "<<diff_Time<<endl;

}
//Functions for CUDA
StatusType colocationFinder::CudaSafeCall(cudaError_t Status)
{
	if (Status == cudaErrorInvalidValue)
	{
		printf("cudaErrorInvalidValue");
	}
	if (cudaSuccess != Status)
	{
		printf(cudaGetErrorString(Status));
		return ST_CUDAERR;
	}
	return ST_NOERR;
}



StatusType colocationFinder::initializeDeviceMemory() {
	StatusType status;
	cudaMalloc((void**)&d_instanceList, maxInstancesNum*sizeof(Integer));

	cudaMemcpy(d_instanceList, instanceList, maxInstancesNum*sizeof(Integer), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_instanceLocationX, maxInstancesNum*sizeof(Integer));

	cudaMemcpy(d_instanceLocationX, instanceLocationX, maxInstancesNum*sizeof(Integer), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_instanceLocationY, maxInstancesNum*sizeof(Integer));
	cudaMemcpy(d_instanceLocationY, instanceLocationY, maxInstancesNum*sizeof(Integer), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_featureInstanceStart, maxFeaturesNum*sizeof(Integer));
	cudaMemcpy(d_featureInstanceStart, featureInstanceStart, maxFeaturesNum*sizeof(Integer), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_featureInstanceEnd, maxFeaturesNum*sizeof(Integer));
	cudaMemcpy(d_featureInstanceEnd, featureInstanceEnd, maxFeaturesNum*sizeof(Integer), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_featureInstanceCount, maxFeaturesNum*sizeof(Integer));
	cudaMemcpy(d_featureInstanceCount, featureInstanceCount, maxFeaturesNum*sizeof(Integer), cudaMemcpyHostToDevice);

	return status;
}

StatusType colocationFinder::initializeDevicConstantMemory() {
	StatusType status;
	status = initializeDeviceMemConst(maxInstancesNum, maxFeaturesNum, parameter, d_instanceList, d_featureInstanceStart, d_featureInstanceEnd, d_featureInstanceCount, d_instanceLocationX, d_instanceLocationY, d_cellEventInstanceCount, gridStructure, d_cellEventInstanceStart, d_cellEventInstanceEnd, d_instanceCellIDs, d_cellBasedSortedIndex);
	return status;
}
/*
 *  A[0 1 2] , B[0 1] , C[0 1 2 3 4] , D[0 1]
 *  degree = 2: 
 *
 *
 *
 */
//Functions for CUDA END
void colocationFinder::tableGenRequired(Integer degree) {
	Integer lastIndex = degree - 2;
	bool flag;
	needInstanceTable = false;
	for (size_t i = 0; i < candiColocCounter; i++) {
		flag = true;
		for (size_t j = i + 1; j < candiColocCounter; j++) {
			for (size_t k = 0; k < lastIndex; k++) {
				if (h_candiColocations[i*(degree - 1) + k] != h_candiColocations[j*(degree - 1) + k]) {
					flag = false;
					break;
				}
			}
			if (flag && h_candiColocations[i*(degree - 1) + lastIndex]< h_candiColocations[j*(degree - 1) + lastIndex]) {
				vector<Integer> inter;
				for (Integer t = 0; t < degree - 1; t++) {
					inter.push_back(h_candiColocations[i*(degree - 1) + t]);
				}
				inter.push_back(h_candiColocations[j*(degree - 1) + lastIndex]);
				//module to generate k-1 degree subset
				flag = generateandCheckSubsets2(inter, degree);
				clearSubsetVectors();
				if (!flag) {
					break;
				}
				else {
					needInstanceTable = true;
					return;
				}
			}
			else {
				break;
			}
		}
	}
	totalCandidatePatterns += candiColocCounter;
}

void colocationFinder::candidateColocationGeneral(Integer degree) {
	Integer lastIndex = degree - 2;
	bool flag;
	candiColocCounter = 0;
	h_candiColocations.clear();
	h_candiColocations.shrink_to_fit();
	for (size_t i = 0; i < h_prevelentColocationCount; i++) {
		flag = true;
		for (size_t j = i + 1; j < h_prevelentColocationCount; j++) {
			for (size_t k = 0; k < lastIndex; k++) {
				if (h_prevalantColocations[i*(degree - 1) + k] != h_prevalantColocations[j*(degree - 1) + k]) {
					flag = false;
					break;
				}
			}
			if (flag && h_prevalantColocations[i*(degree - 1) + lastIndex]< h_prevalantColocations[j*(degree - 1) + lastIndex]) {
				vector<Integer> inter;
				for (Integer t = 0; t < degree - 1; t++) {
					inter.push_back(h_prevalantColocations[i*(degree - 1) + t]);
				}
				inter.push_back(h_prevalantColocations[j*(degree - 1) + lastIndex]);
				//module to generate k-1 degree subset
				flag = generateandCheckSubsets(inter, degree);
				clearSubsetVectors();
				if (!flag) {
					break;
				}
				else {
					for (size_t l = 0; l < degree; l++) {
						h_candiColocations.push_back(inter[l]);
					}
					candiColocCounter++;
				}
			}
			else {
				break;
			}
		}
	}
	totalCandidatePatterns += candiColocCounter;
}
void colocationFinder::kplus2ColocationGeneral(Integer degree) {
	Integer lastIndex = degree - 2;
	bool flag;
	candiColocCounter = 0;
	kplus2CandiColocation.clear();
	kplus2CandiColocation.shrink_to_fit();
	for (size_t i = 0; i < h_prevelentColocationCount2; i++) {
		flag = true;
		for (size_t j = i + 1; j < h_prevelentColocationCount2; j++) {
			for (size_t k = 0; k < lastIndex; k++) {
				if (h_prevalantColocations2[i*(degree - 1) + k] != h_prevalantColocations2[j*(degree - 1) + k]) {
					flag = false;
					break;
				}
			}
			if (flag && h_prevalantColocations2[i*(degree - 1) + lastIndex]< h_prevalantColocations2[j*(degree - 1) + lastIndex]) {
				vector<Integer> inter;
				for (Integer t = 0; t < degree - 1; t++) {
					inter.push_back(h_prevalantColocations2[i*(degree - 1) + t]);
				}
				inter.push_back(h_prevalantColocations2[j*(degree - 1) + lastIndex]);
				//module to generate k-1 degree subset
				flag = generateandCheckSubsetskplus2(inter, degree);
				clearSubsetVectors();
				if (!flag) {
					break;
				}
				else {
					for (size_t l = 0; l < degree; l++) {
						kplus2CandiColocation.push_back(inter[l]);
					}
				}
			}
			else {
				break;
			}
		}
	}
}

void colocationFinder::subsetGen(vector<Integer> &inter, Integer k, Integer n, Integer idx) {
	if (idx == n)
		return;

	if (k == 1) {
		for (size_t i = idx; i<n; i++)
		{
			subset.push_back(inter[i]);
			subsetList.push_back(subset);
			subset.pop_back();
		}
	}

	for (size_t j = idx; j<n; j++) {
		subset.push_back(inter[j]);
		subsetGen(inter, k - 1, n, j + 1);
		subset.pop_back();
	}
}

bool colocationFinder::checkSubset(vector<Integer> subsetElem) {
	Integer degree = subsetElem.size();
	Integer flag = true;
	for (size_t i = 0; i < h_prevelentColocationCount; i++) {
		for (size_t j = 0; j < degree; j++) {
			if (h_prevalantColocations[i*degree + j] != subsetElem[j]) {
				flag = false;
				break;
			}
			flag = true;
		}
		if (flag) {
			return flag;
		}
	}
	return flag;
}
bool colocationFinder::checkSubset2(vector<Integer> subsetElem) {
	Integer degree = subsetElem.size();
	Integer flag = true;
	for (size_t i = 0; i < candiColocCounter; i++) {
		for (size_t j = 0; j < degree; j++) {
			if (h_candiColocations[i*degree + j] != subsetElem[j]) {
				flag = false;
				break;
			}
			flag = true;
		}
		if (flag) {
			return flag;
		}
	}
	return flag;
}
bool colocationFinder::checkSubsetkplus2(vector<Integer> subsetElem) {
	Integer degree = subsetElem.size();
	Integer flag = true;
	for (size_t i = 0; i < h_prevelentColocationCount2; i++) {
		for (size_t j = 0; j < degree; j++) {
			if (h_prevalantColocations2[i*degree + j] != subsetElem[j]) {
				flag = false;
				break;
			}
			flag = true;
		}
		if (flag) {
			return flag;
		}
	}
	return flag;
}
bool colocationFinder::generateandCheckSubsets(vector<Integer> &inter, Integer degree) {
	subsetGen(inter, degree - 1, degree, 0);
	bool flag = true;
	for (size_t i = 2; i < degree; i++) {
		flag = checkSubset(subsetList[i]);
		if (!flag) {
			return flag;
		}
	}
	return flag;
}
bool colocationFinder::generateandCheckSubsets2(vector<Integer> &inter, Integer degree) {
	subsetGen(inter, degree - 1, degree, 0);
	bool flag = true;
	for (size_t i = 2; i < degree; i++) {
		flag = checkSubset2(subsetList[i]);
		if (!flag) {
			return flag;
		}
	}
	return flag;
}
bool colocationFinder::generateandCheckSubsetskplus2(vector<Integer> &inter, Integer degree) {
	subsetGen(inter, degree - 1, degree, 0);
	bool flag = true;
	for (size_t i = 2; i < degree; i++) {
		flag = checkSubsetkplus2(subsetList[i]);
		if (!flag) {
			return flag;
		}
	}
	return flag;
}

Integer colocationFinder::getIndex(vector<Integer> inner, Integer degree) {
	bool flag = false;
	for (size_t i = 0; i < h_prevelentColocationCount; i++) {
		for (size_t j = 0; j < degree; j++) {
			if (inner[j] != h_prevalantColocations[i*degree + j]) {
				flag = false;
				break;
			}
			flag = true;
		}
		if (flag) {
			return i;
		}
	}
	return -1;
}
Integer colocationFinder::getIndex2(vector<Integer> inner, Integer degree) {
	bool flag = false;
	for (size_t i = 0; i < h_prevelentColocationCount2; i++) {
		for (size_t j = 0; j < degree; j++) {
			if (inner[j] != h_prevalantColocations2[i*degree + j]) {
				flag = false;
				break;
			}
			flag = true;
		}
		if (flag) {
			return i;
		}
	}
	return -1;
}

void colocationFinder::generatePrevalentPatternsGeneral(Integer degree) {
	cudaEventRecord(startPoint, 0);

	Integer lastIndex = degree - 2;
	h_prevelentColocationCount2 = 0;
	Integer filterpruneCounter = 0;
	Integer pSlotCounter = 0, expFactor = 100;
	//if (estimatePrevalent < candiColocCounter) {
	delete [] h_prevalentSlotCounts;
	h_prevalentSlotCounts = (Integer**)malloc(candiColocCounter * sizeof(Integer*));
	h_prevalentSlotCounts[0] = (Integer*)malloc(expFactor * candiColocCounter * slotLimit * sizeof(Integer));
	estimatePrevalent = candiColocCounter;
	//}
	cudaError_t cuEr = cudaMalloc((void**)&d_candiColocations, candiColocCounter * degree * sizeof(Integer));
	cuEr = cudaMemcpy(d_candiColocations, h_candiColocations.data(), candiColocCounter * degree * sizeof(Integer), cudaMemcpyHostToDevice);
	//++++ d_isCellProcessed: It makes sure each instance in each cell is counted only once in each pattern.
	cuEr = cudaMalloc((void**)&d_isCellProcessed, gridStructure.totalCells * candiColocCounter * sizeof(int));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_isCellProcessed.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	cudaMemset(d_isCellProcessed, 0, gridStructure.totalCells * candiColocCounter * sizeof(int));
	//++++ d_countMap2: For each given pattern i, it has degree integers counting number of participating instances of each feature. size: degree * nChoosek(n, 2)
    	cuEr = cudaMalloc((void**)&d_countMap2, candiColocCounter * degree * sizeof(Integer));
	if(cuEr != cudaSuccess){printf("\nError in allocating %d GPU memory for d_candiColocations.\n", candiColocCounter * degree * sizeof(Integer));exit(1);}
	cuEr = cudaMemset(d_countMap2, 0, candiColocCounter * degree * sizeof(Integer));

	filter2(d_isCellProcessed, d_cellEventInstanceCount, gridStructure, maxFeaturesNum, d_featureInstanceCount, parameter.PIthreshold, 
		d_candiColocations, candiColocCounter, degree, d_countMap2);

    	Integer *countMap2 = (Integer*) malloc(candiColocCounter * degree * sizeof(Integer));
	cudaMemcpy(countMap2, d_countMap2, candiColocCounter * degree * sizeof(Integer), cudaMemcpyDeviceToHost);

	//cudaDeviceSynchronize();
        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&diff_Time, startPoint, endPoint);
	filter_Time += diff_Time;
	if(PRINT_TIMES)cout<<"\ncolocNum: "<<candiColocCounter<<", Degree: "<<degree<<", Filter time: "<<diff_Time;

	cudaFree(d_countMap2);
	cudaFree(d_isCellProcessed);


        cudaEventRecord(startPoint, 0);


	for (size_t i = 0; i < candiColocCounter; i++) {
		if( !*(countMap2 + i * degree) )continue;
		Integer table1Index;
		Integer table2Index;
		vector<Integer> combiColoc1;    //k combining pattern to make a k+1 candidate pattern 
		vector<Integer> combiColoc2;
		Integer instaceCountTable1;
		for (size_t t = 0; t < degree; t++) {
			if (t < degree - 2) {
				combiColoc1.push_back(h_candiColocations[i*degree + t]);
				combiColoc2.push_back(h_candiColocations[i*degree + t]);
			}
		}
		combiColoc1.push_back(h_candiColocations[i*degree + (degree - 2)]);
		combiColoc2.push_back(h_candiColocations[i*degree + (degree - 1)]);

		table1Index = getIndex(combiColoc1, degree - 1);
		instaceCountTable1 = h_prevalentInstanceTableSize[table1Index];///(degree-1);


		if (instaceCountTable1 > instanceMax) {
			cudaFree(d_slotCounts);
			//cout<<"\nBAD LUCK I\n";
			cudaMalloc((void**)&d_slotCounts, instaceCountTable1*sizeof(Integer));
		}
		cudaMemset(d_slotCounts, 0, instaceCountTable1*sizeof(Integer));
		if (instaceCountTable1*(degree - 1)> table1Max) {
			cudaFree(d_table1);
			cudaMalloc((void**)&d_table1, instaceCountTable1*(degree - 1)*sizeof(Integer));
			table1Max = instaceCountTable1*(degree - 1);
			//cout<<"\nBAD LUCK II\n";
		}
		cudaMemcpy(d_table1, h_prevalentInstanceTable[table1Index], instaceCountTable1*(degree - 1)*sizeof(Integer), cudaMemcpyHostToDevice);
		Integer bitmapSize = 0;
		Integer fType;
		for (size_t Idx = 0; Idx < degree; Idx++) {
			//fType = h_coloc[Idx];
			fType = h_candiColocations[i * degree + Idx];
			bitmapSize += featureInstanceCount[fType];
		}
		cudaMemset(d_bitmap, 0, bitmapSize*sizeof(Integer));
		//Integer lastFeatureID = h_coloc[degree - 1];
		Integer lastFeatureID = h_candiColocations[i * degree + degree - 1];
		generalInstanceTableSlotCounter(instaceCountTable1, d_slotCounts, degree, d_table1, d_bitmap, d_candiColocations + i * degree, lastFeatureID);

		Real PI = getParticipationIndex(d_bitmap, degree, i, h_candiColocations, featureInstanceStart, featureInstanceEnd, featureInstanceCount);

		if (PI < parameter.PIthreshold)continue;
		if (instaceCountTable1 + pSlotCounter >= expFactor * candiColocCounter * slotLimit) {
			cout<<endl<<"Not enough memory for h_prevalentColocationCount: "<<i<<" out of "<<candiColocCounter<<"  , instanceCountTable1: "<<instaceCountTable1<<"  vs  "<<expFactor*candiColocCounter * slotLimit<<endl;
			exit(1);
			//h_prevalentSlotCounts[h_prevelentColocationCount]= (Integer*) malloc(table1InstanceCount*sizeof(Integer));
		}
		h_prevalentSlotCounts[h_prevelentColocationCount2]= h_prevalentSlotCounts[0] + pSlotCounter;
		pSlotCounter += instaceCountTable1;

		cudaMemcpy(h_prevalentSlotCounts[h_prevelentColocationCount2], d_slotCounts, instaceCountTable1*sizeof(Integer), cudaMemcpyDeviceToHost);

		for (Integer Idx = 0; Idx < degree; Idx++) {
			h_prevalantColocations2.push_back(h_candiColocations[i * degree + Idx]);
		}
		h_prevelentColocationCount2++;
	}
	totalFilteredPatterns += filterpruneCounter;
	cudaFree(d_candiColocations);
	free(countMap2);

        cudaEventRecord(endPoint, 0);
        cudaEventSynchronize(endPoint);
        cudaEventElapsedTime(&diff_Time, startPoint, endPoint);
	refinement_Time += diff_Time;
	if(PRINT_TIMES)cout<<", Refinement time: "<<diff_Time<<endl;

	return;
}

void colocationFinder::generateInstanceTableGeneral(Integer degree) {
	Integer lastIndex = degree - 2;
	h_prevalentInstanceTable2 = (Integer**)malloc((h_prevelentColocationCount2)*sizeof(Integer*));
	StatusType status;
	for (size_t i = 0; i < h_prevelentColocationCount2; i++) {
		if (hasInstanceTable[i]==1)
		{
			Integer instaceCountTable1;
			Integer table1Index;
			Integer table2Index;
			vector<Integer> combiColoc1;    //k combining pattern to make a k+1 candidate pattern 
			vector<Integer> combiColoc2;

			for (size_t t = 0; t < degree - 2; t++) {
				combiColoc1.push_back(h_prevalantColocations2[i*degree + t]);
				combiColoc2.push_back(h_prevalantColocations2[i*degree + t]);
			}
			combiColoc1.push_back(h_prevalantColocations2[i*degree + (degree - 2)]);
			combiColoc2.push_back(h_prevalantColocations2[i*degree + (degree - 1)]);

			table1Index = getIndex(combiColoc1, degree - 1);
			instaceCountTable1 = h_prevalentInstanceTableSize[table1Index];
			size_t totalInstances = 0;
			if (instaceCountTable1 > instanceMax) {
				//cudaFreeHost(h_Indexes);
				free(h_Indexes);
				cudaFree(d_Indexes);
				//cudaMallocHost((void**)&h_Indexes, instaceCountTable1*sizeof(Index64));
				h_Indexes = (Index64*) malloc(instaceCountTable1*sizeof(Index64));
				status = CudaSafeCall(cudaMalloc((void**)&d_Indexes, instaceCountTable1*sizeof(Index64)));
			}
			thrust::exclusive_scan(h_prevalentSlotCounts[i], h_prevalentSlotCounts[i] + instaceCountTable1, h_Indexes);
			totalInstances = getTotalInstances(h_prevalentSlotCounts[i], instaceCountTable1);
			status = CudaSafeCall(cudaMemcpy(d_Indexes, h_Indexes, instaceCountTable1*sizeof(Index64), cudaMemcpyHostToDevice));
			status = scalebyConstant(degree, d_Indexes, instaceCountTable1);
			if (instaceCountTable1 > instanceMax) {
				cudaFree(d_slotCounts);
				status = CudaSafeCall(cudaMalloc((void**)&d_slotCounts, instaceCountTable1*sizeof(Integer)));
			}
			Integer lastFeatureID = h_prevalantColocations2[i*degree + (degree - 1)];
			status = CudaSafeCall(cudaMemcpy(d_slotCounts, h_prevalentSlotCounts[i], instaceCountTable1*sizeof(Integer), cudaMemcpyHostToDevice));
			status = CudaSafeCall(cudaMemcpy(d_table1, h_prevalentInstanceTable[table1Index], instaceCountTable1*(degree - 1)*sizeof(Integer), cudaMemcpyHostToDevice));
			status = calcInstanceTableGeneral(instaceCountTable1, d_slotCounts, d_Indexes, d_intermediate, degree, d_table1, 0, lastFeatureID);
			h_prevalentInstanceTableSize2.push_back(totalInstances);
			size_t mSize = totalInstances*degree;
			//cudaMallocHost((void**)&h_prevalentInstanceTable2[i], mSize*sizeof(Integer));
			h_prevalentInstanceTable2[i] = (Integer*) malloc(mSize*sizeof(Integer));
			status = CudaSafeCall(cudaMemcpy(h_prevalentInstanceTable2[i], d_intermediate, mSize*sizeof(Integer), cudaMemcpyDeviceToHost));
		}
		else {
			h_prevalentInstanceTableSize2.push_back(0);
		}
	}
}

void colocationFinder::copyPrevalentColocations() {
	h_prevalantColocations = h_prevalantColocations2;
	h_prevalantColocations2.clear();
	//h_prevalantColocations2.shrink_to_fit();
	h_prevalentInstanceTable = h_prevalentInstanceTable2;
	h_prevelentColocationCount = h_prevelentColocationCount2;
	h_prevelentColocationCount2 = 0;
	h_prevalentInstanceTableSize = h_prevalentInstanceTableSize2;
	h_prevalentInstanceTableSize2.clear();
	//h_prevalentInstanceTableSize2.shrink_to_fit();
}
//Cleaning Functions start
void colocationFinder::resetPrevalentData(Integer degree) {

}

void colocationFinder::clearMemory() {
	delete[] h_prevalentInstanceTable;

	h_prevalantColocations.clear();
	h_prevalantColocations.shrink_to_fit();

	h_prevalantColocations2.clear();
	h_prevalantColocations2.shrink_to_fit();

	h_prevalentInstanceTableSize.clear();
	h_prevalentInstanceTableSize.shrink_to_fit();

	h_prevalentInstanceTableSize2.clear();
	h_prevalentInstanceTableSize2.shrink_to_fit();

	cudaFree(d_coloc);
	delete[] h_coloc;
	cudaFree(d_Indexes);
	cudaFree(d_slotCounts);
	cudaFree(d_intermediate);
	cudaFree(d_table1);
}

void colocationFinder::clearSubsetVectors() {
	subset.clear();
	subset.shrink_to_fit();
	subsetList.clear();
	subsetList.shrink_to_fit();
}

//Cleaning Functions ends

//Log Function starts
void colocationFinder::savetoFile2(Integer degree) {
	std::string degStr = std::to_string(degree);
	std::string FileName = location + "ColocationDegree_" + degStr + ".txt";
	std::ofstream fout(FileName.c_str());
	for (size_t i = 0; i < h_prevelentColocationCount; i++) {
		fout << "( ";
		for (size_t j = 0; j < degree; j++) {
			size_t index = h_prevalantColocations[i*degree + j];
			fout << featureTypes[index] << " ";
			if (j != degree - 1) {
				fout << "| ";
			}
			else {
				fout << ")" << "\n";
			}
		}
	}
	fout.close();
}
void colocationFinder::savetoFileGen(Integer degree) {
	std::string degStr = std::to_string(degree);
	std::string FileName = location + "ColocationDegree_" + degStr + ".txt";
	std::ofstream fout(FileName.c_str());
	for (size_t i = 0; i < h_prevelentColocationCount2; i++) {
		fout << "( ";
		for (size_t j = 0; j < degree; j++) {
			size_t index = h_prevalantColocations2[i*degree + j];
			fout << featureTypes[index] << " ";
			if (j != degree - 1) {
				fout << "| ";
			}
			else {
				fout << ")" << "\n";
			}
		}
	}
	fout.close();
}

void colocationFinder::timeLog(Integer degree, std::string function, std::string eventType, Integer duration) {
	time_t currentTime;
	struct tm *localTime;
	time(&currentTime);                   // Get the current time
	localTime = localtime(&currentTime);  // Convert the current time to the local time
	timeLogStream.open(logFileName.c_str(), ofstream::app);
	timeLogStream << degree << " " << function << " " << eventType << " " << localTime->tm_year + 1900 << "\\" << localTime->tm_mon + 1 << "\\" << localTime->tm_mday << " " << localTime->tm_hour << ":" << localTime->tm_min << ":" << localTime->tm_sec << "\n";
	if (eventType == "end") {
		timeLogStream << degree << " " << function << " " << "Duration: " << " " << duration << "seconds \n";
	}
	timeLogStream.close();
}
void colocationFinder::compLog(Integer degree, std::string function) {
	compLogStream.open(compLogFileName.c_str(), ofstream::app);
	compLogStream << "Degree: " << degree << ", " << function << "\n";
	compLogStream.close();
}
void colocationFinder::compLog(Integer i, Integer degree, Integer candicolocNumber, std::string Remark, std::string PIType, Real PI) {
	compLogStream.open(compLogFileName.c_str(), ofstream::app);
	if (Remark != "") {
		compLogStream << "Degree: " << degree << ", Candidate " << i << "/" << candicolocNumber << PIType << PI << " (" << Remark << ") \n";
	}
	else {
		compLogStream << "Degree: " << degree << ", Candidate " << i << "/" << candicolocNumber << PIType << PI << "\n";
	}
	compLogStream.close();
}
void colocationFinder::compLog(Integer degree, Integer i, std::string totalmem, Integer candiColocNum) {
	compLogStream.open(compLogFileName.c_str(), ofstream::app);
	compLogStream << "Degree: " << degree << " pattern No.: " << i << "/" << candiColocNum << " Total memRequired =" << totalmem << "GB\n";
	compLogStream.close();
}
void colocationFinder::loggingFileOpenFunction() {
	logFileName = location + "GPU_TimeLog.txt";
	compLogFileName = location + "GPU_ComputationLog.txt";
}
void colocationFinder::loggingFileCloseFunction() {
	timeLogStream.open(logFileName.c_str(), ofstream::app);
	timeLogStream << "distance Threshold: " << parameter.thresholdDistance << "\n";
	timeLogStream << "prevalance Threshold: " << parameter.PIthreshold << "\n";
	timeLogStream.close();
}
//Log Functions ends
void colocationFinder::calcInstanceTablePatternIndexes(Integer degree) {
	hasInstanceTable.clear();
	hasInstanceTable.shrink_to_fit();
	for (Integer i = 0; i < h_prevelentColocationCount2; i++) {
		hasInstanceTable.push_back(0);
	}
	Integer degreePlus1 = degree + 1;
	kplus2ColocationGeneral(degree+1);
	Integer kplu2PatternsCount = kplus2CandiColocation.size()/ degreePlus1;
	if (kplu2PatternsCount == 0) {
		needInstanceTable = false;
		return;
	}
	for (Integer i = 0; i < kplu2PatternsCount; i++) {
		vector<Integer> combiColoc1;
		for (size_t t = 0; t < degreePlus1 - 1; t++) {
				combiColoc1.push_back(kplus2CandiColocation[i*degreePlus1 + t]);
				//combiColoc2.push_back(h_candiColocations[i*degree + t]);
		}
		//combiColoc2.push_back(h_candiColocations[i*degree + (degree - 1)]);

		Integer table1Index = getIndex2(combiColoc1, degreePlus1 - 1);
		hasInstanceTable[table1Index] = 1;
	}
}

void colocationFinder::Begin(Integer argc, char**argv) {
        
	cudaEvent_t start, end;
        float e2e_Time;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);
	
	std::string datasetFilename;
	if (argc > 1) {
		std::string configurationFile = argv[1];
		ifstream in(configurationFile.c_str());
		std::string line;
		std::string parameterFilename;
		getline(in, line);
		location = line;
		getline(in, line);
		parameterFilename = line;
		getline(in, line);
		datasetFilename = line;
		getline(in, line);
		outputLocation = line;
		getline(in, line);
		outputFile = line;
		in.close();
		LoadParameter(parameterFilename);
		populateData(datasetFilename);

	}
	else {
		std::cout << "Configuration file missing...";
		exit(0);
	}

	if(PRINT_TIMES)cout<<endl<<"Dataset: "<<datasetFilename<<"\t Number of points: "<<maxInstancesNum<<"  \t Number of features:"<<maxFeaturesNum<<"  \t , Distance threshold: "<<parameter.thresholdDistance<<endl;

	degree2Processing();

	Integer degree = 3;
	Integer featureCount = featureTypes.size();
	
	cudaEvent_t s1, s2;
	Real d, t1=0;
	cudaEventCreate(&s1);
	cudaEventCreate(&s2);

	while (candiColocCounter > 0 && degree <= featureCount) {
		tableGenRequired(degree+1);
		generatePrevalentPatternsGeneral(degree);

		cudaEventRecord(s1,0);
		if (needInstanceTable) {
			calcInstanceTablePatternIndexes(degree);
			generateInstanceTableGeneral(degree);
		}
		copyPrevalentColocations();
		candidateColocationGeneral(degree + 1);

		cudaEventRecord(s2,0);
		cudaEventSynchronize(s2);
        	cudaEventElapsedTime(&d, s1, s2); 
		t1 += d;

		if (candiColocCounter == 0) {
			clearMemory();//clean memory
			break;
		}
		degree++;

	}

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&e2e_Time, start, end); 

	//DatasetName  |  ExecutableName | prevalanceThreshold | Transfer Time | PreProcessing Time | Filter Time | Refine Time | Total Time
	cout<<endl<<datasetFilename<<" , "<<"GPU_Danial"<<" , "<<parameter.PIthreshold<<" , "<<io_Time<<" , "<<transfer_Time<<" , "<<grid_Time<<" , "<<filter_Time<<" , "<<refinement_Time<<" , "<<t1<<" , "<<e2e_Time;

	return;
}


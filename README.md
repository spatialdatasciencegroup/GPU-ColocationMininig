# GPU Colocation
Colocation patterns refer to subsets of spatial features whose instances are frequently located together. Mining colocation patterns is important in many applications such as identifying relationships between diseases and environmental factors, but is computationally challenging due to the large number of instances and candidate patterns. Existing algorithms are mostly sequential, and thus can be insufficient for big spatial event data. Recently, parallel colocation mining algorithms have been developed based on the Map-reduce framework, which is economically expensive. Another work proposed a GPU algorithm based on iCPI tree, but assumes that the number of neighbors for each instance is within a small constant, and thus cannot be used when instances are dense and unevenly distributed. To address these limitations, we recently proposed grid-based GPU colocation mining algorithms that include a novel cell-aggregate-based upper bound filter, and two refinement algorithms. In this paper, we provide theoretical analysis of running time. Furthermore using GPU profiling, we identify our recent GPU implementation, GPU-grid-join, as a memory bound problem and to address its bottlenecks, we proposes GPU-grid-join+, an optimized GPU algorithm. Our experimental results on real world data shows that GPU-grid-join+ achieves 4 to 12-fold speedup over GPU-grid-join both running on Nvidia P100 GPU as well as 56 to 126-fold speedup over OpenMP implementation over Intel(R) Xeon(R) CPU with 12 cores. Also for synthetic data, the speedup is in ranges 3 to 7-fold and 9 to 42-fold respectively.

This is a GPU-based program to for parallel colocation mining [1,2]

### How to Compile

1. Install boost. (https://www.boost.org/users/download/) 
2. Change your current directory to be the directory containing main.cpp  and colocationFinder.cpp
3. In the console, compile with: **nvcc DeviceProcess.cu -I (boost_installation_path) main.cpp colocationFinder.cpp -o GPU-Colocation

### How to Run

1. We have a data folder for you to test out.
2. To run the compiled GPU-Colocation program: ./GPU-Colocation /home/TestCase/config.txt

### Input Data Description
1. Please see the Technical Document pdf file. 

### References

[1] Arpan Man Sainju, Danial Aghajarian, Zhe Jiang, & Sushil K Prasad, (2018). Parallel grid-based colocation mining algorithms on GPUs for big spatial event data. IEEE Transactions on Big Data.

[2] Arpan Ma Sainju, and Zhe Jiang. "Grid-based colocation mining algorithms on gpu for big spatial event data: A summary of results." International Symposium on Spatial and Temporal Databases. Springer, Cham, 2017.

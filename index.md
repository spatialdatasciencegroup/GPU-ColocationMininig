## Introduction

Given a set of spatial features and their instances, co-location mining aims to find subsets of features whose instances are frequently located together. Examples of colocation patterns include symbiotic relationships between species such as Nile Crocodiles and Egyptian Plover, as well as environmental factors and disease events (e.g., air pollution and lung cancer). 

Colocation mining is important in many applications that aim to find  associations between different spatial events or factors. For example, in public safety, law enforcement agencies are interested in finding relationships between different crime event types and potential crime generators. In ecology, scientists analyze common spatial footprints of various species to capture their interactions and spatial distributions. In public health, identifying relationships between human disease and potential environmental causes is an important problem. In climate science, colocation patterns help reveal relationships between the occurrence of different climate extreme events. In location based service, colocation patterns help identify travelers that share the same favourite locations to promote effective tour recommendation. 

Mining colocation patterns from big spatial event data poses several computational challenges. First, in order to evaluate if a candidate colocation pattern is prevalent, we need to generate its instances. This is computationally expensive due to checking spatial neighborhood relationships between different instances,  particularly when the number of instances is large and instances are clumpy (e.g., many instances are within the same spatial neighborhoods). Second, the number of candidate colocation patterns are exponential to the number of spatial features. Evaluating a large number of candidate patterns can be computationally prohibitive. Finally, the distribution of event instances in the space may be uneven, making it hard to design parallel data structure and algorithms.

Colocation pattern mining has been studied extensively in the literature, including early work on spatial association rule mining and colocation patterns based on event-centric model. Various algorithms have been proposed to efficiently identify colocation patterns, including Apriori generator and multi-resolution upper bound filter, partial join and joinless approach, iCPI tree based colocation mining algorithms. There are also works on identifying regional or zonal colocation patterns, and statistically significant colocation patterns, top-K prevalent colocation patterns or prevalent patterns without thresholding. 
Existing algorithms are mostly sequential, and can be insufficient when the number of event instances is very large (e.g., several millions). 
Parallel colocation mining algorithms are relatively less studied. 

We propose GPU colocation mining algorithms based on a grid index, including a cell-aggregate-based upper bound filter and two refinement algorithms. Proposed cell-aggregate-based filter is easier to implement in parallel on GPU and is also insensitive to pattern clumpiness (the average number of overlaying colocation instances for a given colocation instance) compared with the existing multi-resolution filter. We also conducted further GPU optimization, as well as detailed theoretical analysis and experimental evaluations.

## Source Codes
- GPU Colocation Mining: [Github Repository](https://github.com/spatialdatasciencegroup/GPU-ColocationMininig)
- CPU Sequential Implementation: [Github Repository](https://github.com/spatialdatasciencegroup/CPU_Colocation)
- CPU Multicore (OpenMP) Version: [Github Repository](https://github.com/spatialdatasciencegroup/OpenMP_Colocation)

## Publications
- Arpan Man Sainju, Zhe Jiang. "Grid-based Co-location Mining Algorithms on GPU for Big Spatial Event Data: A Summary of Results". International Symposium on Spatial and Temporal Databases (SSTD), 2017 [PDF](https://www.jiangteam.org/papers/SSTD2017.pdf)
- Arpan Man Sainju, Daniel Aghajarian, Zhe Jiang, Sushil Prasad, "Parallel Grid-based Colocation Mining Algorithms on GPUs for Big Spatial Event Data", IEEE Transactions on Big Data (TBD), 2018 [PDF](https://www.jiangteam.org/papers/TBD2018.pdf)


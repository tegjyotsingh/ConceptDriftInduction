# ConceptDriftInduction
Datasets for concept drift detection

The repository presents datasets used in the paper: 

Sethi, Tegjyot Singh, and Mehmed Kantardzic. "On the Reliable Detection of Concept Drift from Streaming Unlabeled Data." Expert Systems with Applications (2017).

The datasets are presented in two parts:

real_world_data: These are datasets from cybersecurity domains, as explained in the paper. 

induced_datasets: These represent datasets which are induced based on the feature ranking methodology as described in the paper. 
The script generate_shuffled_synthetic.py is used to generate these datasets from original UCI classification datasets. data_infra.py is a simple library file for routine operations. 

The datasets are obtianed from sources as mentioned in paper. Most of them are from UCI repository. The processed versions, for replication of experiments, is presented here. 

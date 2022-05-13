# Cancer Gene Expression Microarray Data Analysis

## Problem Statement

ICMR wants to analyze different types of cancers, such as breast cancer, renal cancer, colon cancer, lung cancer, and prostate cancer becoming a cause of worry in recent years. They would like to identify the probable cause of these cancers in terms of genes responsible for each cancer type. This would lead us to early identification of each type of cancer reducing the fatality rate.

## Dataset Details

The input dataset contains 802 samples for the corresponding 802 people who have been detected with different types of cancer. Each sample contains expression values of more than 20K genes. Samples have one of the types of tumors: BRCA, KIRC, COAD, LUAD, and PRAD.

## Tasks

### Week 1

 1. Plot the merged dataset as a hierarchically-clustered heatmap to see if the five classes of genes are shown distinctively to see their existence

 2. Apply the feature selection algorithms, filter the actual dataset for selected columns and save it as a new DataFrame to represent the feature selection data

### Week 2

1. Each sample has expression values for around 20K genes. However, it may not be necessary to include all 20K genes’ expression values to analyze each cancer type. Therefore, identify a smaller set of attributes which will then be used to fit multiclass classification models using dimensionality reduction techniques.

2. Filter the actual dataset for the columns (or genes) suggested by this approach and save it as a new DataFrame to represent the dimensionality reduction data.


### Week 3

Identify groups of genes that behave similarly across samples and identify the distribution of samples corresponding to each cancer type.

1. First, apply the given clustering technique on all genes to identify:
   - Genes whose expression values are similar across all samples
   - Genes whose expression values are similar across samples of each cancer type 

2. Next, apply the given clustering technique on all samples to identify:
   - Samples of the same class (cancer type) which also correspond to the same cluster
   - Samples identified to be belonging to another cluster but also to the same class (cancer type)

### Week 4

1. Build a robust classification model for identifying each type of cancer. Try variants of SVM, Random Forest and Neural Networks on original, selected and extracted features and evaluated on AUC score.

2. Write an observation based on your analysis of the best models considered in the previous step. 

## Solution

#### [See Project Thesis](https://lookupinthesky.github.io/Purdue-Simplilearn-AI-ML/gene-expression/Report.pdf)

#### Notebooks

Week 1 - [EDA](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%201%20-%20EDA.ipynb) , [Feature Selection](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%201%20-%20Feature%20Selection.ipynb)

Week 2 - [Dimensionality Reduction](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%202%20-Dimensionality%20Reduction.ipynb)

Week 3 - [Clustering Genes](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%203%20-%20Clustering%20Genes.ipynb) , [Clustering Samples](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%203%20-%20Clustering%20Samples.ipynb)

Week 4 - [Model Building Part 1](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%204%20-%20Model%20Building%20-%201.ipynb), [Model Building Part 2 and Analysis](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/blob/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis/Week%204%20-%20Model%20Building%20-%202.ipynb)


[View the entire project on Github](https://github.com/lookupinthesky/Purdue-Simplilearn-AI-ML/tree/main/Cancer%20Gene%20Expression%20%20Microarray%20Data%20Analysis)


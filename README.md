# SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data
===========================================================================

[![license](https://img.shields.io/badge/scGPT-black)](https://github.com/bowang-lab/scGPT)
[![license](https://img.shields.io/badge/LIANA+-pink)](https://github.com/saezlab/liana-py)
[![license](https://img.shields.io/badge/python_-3.8.0_-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/torch_-1.12.0_-red)](https://pytorch.org/)
[![license](https://img.shields.io/badge/scanpy_-1.9.0_-green)](https://scanpy.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/anndata_-0.8.0_-black)](https://anndata-tutorials.readthedocs.io/en/latest/index.html/)
[![license](https://img.shields.io/badge/R_-4.2.2_-pink)](https://www.r-project.org/)

Drawing parallels between linguistic constructs and cellular biology, large language models (LLMs) have achieved success in diverse downstream applications for single-cell data analysis. However, to date, it still lacks methods to take advantage of LLMs to infer ligand-receptor (LR)-mediated cell-cell communications for spatially resolved transcriptomic data. Here, we propose SpaCCC to facilitate the inference of spatially resolved cell-cell communications, which relies on our fine-tuned single-cell LLM and functional gene interaction network to embed ligand and receptor genes expressed in interacting individual cells into a unified latent space. The LR pairs with a significant closer distance in latent space are taken to be more likely to interact with each other. After that, the molecular diffusion and permutation test strategies are respectively employed to calculate the communication strength and filter out communications with low specificities. The benchmarked performance of SpaCCC is evaluated on real single-cell spatial transcriptomic datasets with superiority over other methods. SpaCCC also infers known LR pairs concealed by existing aggregative methods and then identifies communication patterns for specific cell types and their signaling pathways. Furthermore, SpaCCC provides various cell-cell communication visualization results at both single-cell and cell type resolution. In summary, SpaCCC provides a sophisticated and practical tool allowing researchers to decipher spatially resolved cell-cell communications and related communication patterns and signaling pathways based on spatial transcriptome data.


![Image text](https://github.com/jiboyalab/SpaCCC/blob/main/IMG/workflow.png)

The overview of the workflow for developing SpaCCC. **(a)** The input, intermediate process for inferring spatially resolved cell-cell communications and output of SpaCCC. **(b)** The detailed process for determining the statistical significance of ligand-receptor pairs and calculating the cell-cell communication strength at single-cell and cell-type resolution. **(c)** Numerous visualization types of SpaCCC to decipher the spatially resolved cell-cell communications (see section 3.1 in the manuscript for details). 


## Table of Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Contributing](#contributing)
- [Cite](#cite)
- [Contacts](#contacts)
- [License](#license)


## Installation

scDCA is tested to work under:

```
* Python 3.8.0
* Torch 1.12.0
* Scanpy 1.9.0
* Anndata 0.8.0
* R 4.2.2
* Numpy 1.23.5
* Other basic python and r toolkits
```
### Installation of other dependencies 
**Notes:** These dependencies, in order to infer ligand–receptor (L-R) pairs from single-cell RNA sequencing data, can skip the installation process if you already have the LR result files (e.g. LR_P76.csv and LR_P915.csv provided in the data folder).
* Install [CellPhoneDB v3](https://github.com/ventolab/CellphoneDB) using ` pip install cellphonedb ` if you encounter any issue. 
* Install [CellChat v1.6.0](https://github.com/sqjin/CellChat/tree/master) using ` devtools::install_github("sqjin/CellChat") ` in the R environment if you encounter any issue.
* Install [NicheNet v1.1.0](https://github.com/saeyslab/nichenetr) using ` devtools::install_github("saeyslab/nichenetr") ` in the R environment if you encounter any issue.
* Install [ICELLNET](https://github.com/soumelis-lab/ICELLNET) using ` install_github("soumelis-lab/ICELLNET",ref="master", subdir="icellnet") ` in the R environment if you encounter any issue.


# Quick start
To reproduce our results:


**Notes:** We provide readers with 3 sets of data as detailed in the following data descriptions. Note that in order to reduce the computational overhead and to make it easier for readers to reproduce our code, we will use the smaller test data in the following tutorials. Processing of other single-cell RNA-Seq data follows the same core pipeline as the test data. Due to the large size of the data, we also uploaded them to the [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link).

## Data description

| File name  | Description |
| ------------- | ------------- |
| RCC_scRNA_P76_matrix.txt  | The single-cell gene expression matrix for patient 76 with advanced renal cell carcinoma. The origional data can be downloaded from the [Paper](https://singlecell.broadinstitute.org/single_cell/study/SCP1288/tumor-and-immune-reprogramming-during-immunotherapy-in-advanced-renal-cell-carcinoma?hiddenTraces=P55_scRNA%2CP90_scRNA%2CP906_scRNA%2CP912_scRNA%2CP913_scRNA%2CP916_scRNA%2CP76_scRNA#study-summary.) and the processed data by us can be obtained from the [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| RCC_scRNA_P76_metadata.txt  | The single-cell metadata, including cell type annotations, for patient 76 with advanced renal cell carcinoma. The origional data can be downloaded from the [Paper](https://singlecell.broadinstitute.org/single_cell/study/SCP1288/tumor-and-immune-reprogramming-during-immunotherapy-in-advanced-renal-cell-carcinoma?hiddenTraces=P55_scRNA%2CP90_scRNA%2CP906_scRNA%2CP912_scRNA%2CP913_scRNA%2CP916_scRNA%2CP76_scRNA#study-summary.) and the processed data by us can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| LR_P76.csv  | The integrated ligand-receptor results for patient 76 with advanced renal cell carcinoma obtained form 4 cell–cell communication analysis tools. The data can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| RCC_scRNA_P915_matrix.txt  | The single-cell gene expression matrix for patient 915 with advanced renal cell carcinoma. The origional data can be downloaded from the [Paper](https://singlecell.broadinstitute.org/single_cell/study/SCP1288/tumor-and-immune-reprogramming-during-immunotherapy-in-advanced-renal-cell-carcinoma?hiddenTraces=P55_scRNA%2CP90_scRNA%2CP906_scRNA%2CP912_scRNA%2CP913_scRNA%2CP916_scRNA%2CP76_scRNA#study-summary.) and the processed data by us can be obtained from the [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| RCC_scRNA_P915_metadata.txt  | The single-cell metadata, including cell type annotations, for patient 915 with advanced renal cell carcinoma. The origional data can be downloaded from the [Paper](https://singlecell.broadinstitute.org/single_cell/study/SCP1288/tumor-and-immune-reprogramming-during-immunotherapy-in-advanced-renal-cell-carcinoma?hiddenTraces=P55_scRNA%2CP90_scRNA%2CP906_scRNA%2CP912_scRNA%2CP913_scRNA%2CP916_scRNA%2CP76_scRNA#study-summary.) and the processed data by us can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| LR_P915.csv  | The integrated ligand-receptor results for patient 915 with advanced renal cell carcinoma obtained form 4 cell–cell communication analysis tools. The data can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| ScRNA_test_data_matrix.txt  | The single-cell gene expression matrix for test data to reduce the computational overhead and to make it easier for readers to reproduce our code. The origional data can be downloaded from the GSE175510 and the processed data by us can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| ScRNA_test_data_metadata.txt  | The single-cell metadata for test data including cell type annotations. The origional data can be downloaded from the GSE175510 and the processed data by us can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| LR_test_data.csv  | The integrated ligand-receptor results for test data. The data can be obtained from the [Github/data](https://github.com/jiboyalab/scDCA/tree/main/data) or [Google Drive](https://drive.google.com/drive/folders/18sUpPlPuT9SBg-2rvurJ-IHjiKnhivI-?usp=drive_link)|
| P76_malignant_cell_states_gsva_mat.txt  | The activity scores calculated by gene set variation analysis (gsva) for 14 functional state in malignant cells of patient P76, which 14 functional state signatures of malignant cells were obtained from the [CancerSEA](http://biocc.hrbmu.edu.cn/CancerSEA/)|
| malignant_cell_states_gsva.txt  | The activity scores calculated by gene set variation analysis (gsva) for 14 functional state in malignant cells of test data, which 14 functional state signatures of malignant cells were obtained from the [CancerSEA](http://biocc.hrbmu.edu.cn/CancerSEA/)|

## 1，Infer ligand–receptor (L-R) pairs from single-cell RNA sequencing data
**Notes:** If you already have an LR result file or want to specify the LR yourself (e.g. LR_P76.csv, LR_P915.csv and LR_test_data.csv provided in the data folder), skip this step.
```
# The following program needs to be run in the cellphonedb environment, see [Cellphonedb](https://github.com/Teichlab/cellphonedb) for details on how to use it:

cellphonedb method statistical_analysis ./data/ScRNA_test_data_metadata.txt ./data/ScRNA_test_data_matrix.txt --counts-data=gene_name --iterations=10 --threads=100 --output-path=./output/
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **counts-data** | [ensembl or gene_name or hgnc_symbol] Type of gene identifiers in the counts data |
| **iterations** | Number of iterations for the statistical analysis [1000] |
| **threads** | Number of threads to use. >=1 [4] |
| **output-path** | Directory where the results will be allocated (the directory must exist). |


```
[ ][CORE][15/08/23-10:17:35][INFO] Initializing SqlAlchemy CellPhoneDB Core
[ ][CORE][15/08/23-10:17:35][INFO] Using custom database at /home/jby2/.cpdb/releases/v4.0.0/cellphone.db
[ ][APP][15/08/23-10:17:35][INFO] Launching Method cpdb_statistical_analysis_local_method_launcher
[ ][APP][15/08/23-10:17:35][INFO] Launching Method _set_paths
[ ][APP][15/08/23-10:17:35][WARNING] Output directory (/home/jby2/HoloNet/github) exist and is not empty. Result can overwrite old results
[ ][APP][15/08/23-10:17:35][INFO] Launching Method _load_meta_counts
[ ][APP][15/08/23-10:17:37][INFO] Launching Method _check_counts_data
[ ][CORE][15/08/23-10:17:37][INFO] Launching Method cpdb_statistical_analysis_launcher
[ ][CORE][15/08/23-10:17:37][INFO] Launching Method _counts_validations
[ ][CORE][15/08/23-10:17:38][INFO] Launching Method get_interactions_genes_complex
[ ][CORE][15/08/23-10:17:38][INFO] [Cluster Statistical Analysis] Threshold:0.1 Iterations:10 Debug-seed:-1 Threads:100 Precision:3
[ ][CORE][15/08/23-10:17:39][INFO] Running Real Analysis
[ ][CORE][15/08/23-10:17:39][INFO] Running Statistical Analysis
[ ][CORE][15/08/23-10:17:43][INFO] Building Pvalues result
[ ][CORE][15/08/23-10:17:43][INFO] Building results
```


```
# The following program needs to be run in the R environment, see [CellChat](https://github.com/sqjin/CellChat), [NicheNet](https://github.com/saeyslab/nichenetr) and [ICELLNET](https://github.com/soumelis-lab/ICELLNET) for details on how to use it:

Rscript ./tools/run_cellchat.R --count ./data/ScRNA_test_data_matrix.txt --meta ./data/ScRNA_test_data_metadata.txt  --output ./output/
```
```
[1] "############ ------------- cellchat --------------- ############"
[1] ">>> loading library and data <<< [2023-08-15 10:48:39]"
[1] ">>> start CellChat workflow <<< [2023-08-15 10:48:50]"
[1] "Create a CellChat object from a data matrix"
The cell barcodes in 'meta' is  P2@CSF-0703-A1-1_GGTAATCA P2@CSF-0703-A1-1_CCTTCAAG P2@CSF-0703-A1-1_CACACTGA P2@CSF-0703-A1-2_GGTGGACT P2@CSF-0703-A1-2_TCAACGAC P2@CSF-0703-A2-1_GGTAATCA 
Set cell identities for the new CellChat object 
The cell groups used for CellChat analysis are  B CD8T Malignant Mono/Macro 
[1] ">>> Infer CCI network <<< [2023-08-15 10:49:05]"
triMean is used for calculating the average gene expression per cell group. 
[1] ">>> Run CellChat on sc/snRNA-seq data <<< [2023-08-15 10:49:05]"
  |======================================================================| 100%
[1] ">>> CellChat inference is done. Parameter values are stored in `object@options$parameter` <<< [2023-08-15 10:49:46]"
[1] ">>> saving results <<< [2023-08-15 10:49:46]"
[1] ">>> done <<< [2023-08-15 10:49:48]"
Warning message:
In createCellChat(object = as.matrix(data.norm), group.by = "group",  :
  The cell barcodes in 'meta' is different from those in the used data matrix.
              We now simply assign the colnames in the data matrix to the rownames of 'mata'!
```


```
# The used ligand-target matrix, lr network and weighted networks of interacting cells for nichenet can be downloaded from [Zenodo](https://zenodo.org/record/7074291) or our Google Drive.

Rscript ./tools/run_nichenet.R --count ./data/ScRNA_test_data_matrix.txt --meta ./data/ScRNA_test_data_metadata.txt  --output ./output/
```
```
[1] "############ ------------- nichenet --------------- ############"
[1] ">>> loading library and data <<< [2023-08-15 11:08:14]"
[1] ">>> generate seurat object <<< [2023-08-15 11:08:35]"
[1] ">>> start Nichenet workflow for each cell types <<< [2023-08-15 11:08:36]"
[1] ">>> B_CD8T start <<< [2023-08-15 11:08:36]"
[1] ">>> B_CD8T write <<< [2023-08-15 11:08:41]"
[1] ">>> B_CD8T finish <<< [2023-08-15 11:08:41]"
...
[1] ">>> Mono/Macro_Malignant strict write <<< [2023-08-15 11:09:46]"
[1] ">>> Mono/Macro_Malignant strict finish <<< [2023-08-15 11:09:46]"
There were 11 warnings (use warnings() to see them)
```


```
# The used ICELLNETdb for icellnet can be downloaded from our Google Drive.

Rscript ./tools/run_icellnet.R --count ./data/ScRNA_test_data_matrix.txt --meta ./data/ScRNA_test_data_metadata.txt  --output ./output/
```
```
[1] "############ ------------- icellnet --------------- ############"
[1] ">>> loading data <<< [2023-08-15 11:17:58]"
[1] ">>> generate Seurat object <<< [2023-08-15 11:18:10]"
[1] ">>> start ICELLNET workflow (sc.data.cleaning) <<< [2023-08-15 11:18:11]"
[1] "Filling in intermediate table: percentage of expressing cell per cluster per gene, and mean of expression"
[1] "Intermediate table were saved as scRNAseq_statsInfo_for_ICELLNET.csv."
[1] "Filtering done"
[1] ">>> Go through each cell types <<< [2023-08-15 11:18:29]"
Note: Check that PC.data and/or CC.data contains rownames. Ignore this note if this is the case 
Note: lr contains only 78 after filtering interaction highest than theshold 
[1] ">>> Bvs others finished <<< [2023-08-15 11:18:30]"
Note: Check that PC.data and/or CC.data contains rownames. Ignore this note if this is the case 
Note: lr contains only 117 after filtering interaction highest than theshold 
[1] ">>> CD8Tvs others finished <<< [2023-08-15 11:18:30]"
Note: Check that PC.data and/or CC.data contains rownames. Ignore this note if this is the case 
Note: lr contains only 120 after filtering interaction highest than theshold 
[1] ">>> Malignantvs others finished <<< [2023-08-15 11:18:31]"
Note: Check that PC.data and/or CC.data contains rownames. Ignore this note if this is the case 
Note: lr contains only 101 after filtering interaction highest than theshold 
[1] ">>> Mono/Macrovs others finished <<< [2023-08-15 11:18:31]"
There were 50 or more warnings (use warnings() to see the first 50)
```

**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **output** | Directory where the results will be allocated. |

```
# Finally, obtain the intersection of LR pairs output by 4 cellular communication tools, which are required to be found by at least 2 tools and have expression in scRNA-seq data.

python ./tools/process_final_lr.py --lr_cellphonedb ./output/process_cellphonedb_lr.csv --lr_cellchat ./output/process_cellchat_lr.csv --lr_nichenet ./output/process_nichenet_lr.csv --lr_icellnet ./output/process_icellchat_lr.csv --count ./data/ScRNA_test_data_matrix.txt --output ./output/LR_test_data.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **lr_cellphonedb** | The results of LR pairs output by cellphonedb. |
| **lr_cellchat** | The results of LR pairs output by cellchat. |
| **lr_nichenet** | The results of LR pairs output by nichenet. |
| **lr_icellnet** | The results of LR pairs output by icellnet. |
| **count** | Count matrix / normalized count matrix path. |
| **output** | The final results of LR pairs. |

## 2，Prioritize the dominant cell communication assmebly that regulates the target gene expression pattern
```
cd ./src/tutorials1/ && python main.py --count /home/jby2/ScRNA_test_data_matrix.txt --meta /home/jby2/ScRNA_test_data_metadata.txt --gene HCST --lr_file /home/jby2/LR_test_data.csv --device cuda:1 --facked_LR 200 --repeat_num 50 --max_epoch 200 --learning_rate 1e-1 --display_loss True --ccc_ratio_result /home/jby2/ccc_ratio_result.csv --dca_rank_result /home/jby2/dca_rank_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name (Please ensure that the gene is highly variable, we detect the highly variable genes by running sc.pp.highly_variable_genes with default parameters). |
| **device** | The device for model training (cuda or cpu, default is cpu). |
| **facked_LR** | The faked ligand and receptor genes number for removing the edges with low specificities (default is 200). |
| **repeat_num** | The repeat number for model training (default is 50). |
| **max_epoch** | The max epoch for model training (default is 200). |
| **learning_rate** | The learning rate for model training (default is 1e-1). |
| **display_loss** | Display training loss for model training (default is True).|
| **dca_rank_result** | The result filename of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result filename of ratio of different cell types affected by cellular communication. |

```
############ ------------- scDCA --------------- ############
>>> arguments <<<
Namespace(ccc_ratio_result='/home/jby2/ccc_ratio_result.csv', count='/home/jby2/ScRNA_test_data_matrix.txt', dca_rank_result='/home/jby2/dca_rank_result.csv', device='cuda:1', display_loss='False', facked_LR='200', gene='HCST', learning_rate='0.1', lr_file='/home/jby2/LR_test_data.csv', max_epoch='200', meta='/home/jby2/ScRNA_test_data_metadata.txt', repeat_num='50')
>>> loading library and data <<<  Tue Aug 15 23:04:05 2023
>>> construct an AnnData object from a count file and a metadata file <<<  Tue Aug 15 23:04:05 2023
>>> load the provided dataframe with the information on ligands and receptors <<<  Tue Aug 15 23:04:08 2023
>>> calculate the CE tensor by considering the expression levels of ligand and receptor genes <<<  Tue Aug 15 23:04:08 2023

    Note:
        This function calculates the CE tensor by considering the expression levels of ligand and receptor genes.
        If the data is large, it may require substantial memory for computation.
        We're working on improving this piece of code.
    
>>> filter the edge in calculated CE tensor, removing the edges with low specificities <<<  Tue Aug 15 23:04:12 2023

    Notes:
        This process will take a long time, if you want to reduce the calculation time, please reduce the facked_LR number, the default value is 200
    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [15:56<00:00,  6.21s/it]
>>> construct a cell type adjacency tensor based on the cell type and the summed LR-CE tensor. <<<  Tue Aug 15 23:20:10 2023
cell type: B
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:31<00:00,  7.75s/it]
cell type: CD8T
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:53<00:00, 17.96s/it]
cell type: Malignant
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [01:52<00:00, 56.26s/it]
cell type: Mono/Macro
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.15s/it]
>>> detect the highly variable genes <<<  Tue Aug 15 23:23:29 2023
>>> start training the multi-view graph convolutional neural network <<<  Tue Aug 15 23:23:30 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:22<00:00,  1.65s/it]
>>> calculate the generated expression profile of the target gene. <<<  Tue Aug 15 23:24:52 2023
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 113.09it/s]
The mean squared error of original and predicted gene expression profiles: 0.03500232
The Pearson correlation of original and predicted gene expression profiles: 0.34182112010369733
>>> the dominant cell communication assmebly that regulates the target gene expression pattern is stored at: <<<  /home/jby2/dca_rank_result.csv Tue Aug 15 23:24:52 2023
>>> the ratio of different cell types affected by cellular communication is stored at: <<<  /home/jby2/ccc_ratio_result.csv Tue Aug 15 23:24:52 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 37.04it/s]
```
A sample output result file is as follows:


**dca_rank_result.csv** （The first column represents the serial number of cell type pairs, ordered by attention weight; the second column represents the cell type pair name; the third column represents the average attention weight for 50 model repetitions of training）:
| | Cell_type_Pair |MGC_layer_attention|
| --- | --- | --- |
| 8 | CD8T:CD8T |0.51397777|
| 4 | Malignant:Mono/Macro |0.44408146|
| 7 | Malignant:Malignant |0.43624955|
| 6 | CD8T:Mono/Macro |0.41747302|
| 5 | CD8T:Malignant |0.4010921|
| 1 | B:CD8T |0.3871707|
| 2 | B:Malignant |0.35314357|
| 9 | Mono/Macro:Mono/Macro |0.3088738|
| 3 | B:Mono/Macro |0.27923074|
| 0 | B:B |0.21194793|


A visualization sample of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDCA/blob/main/IMG/cd8arank.png" alt="Editor" width="500">
</div>

===========================================================================


**ccc_ration_result.csv** (The first column represents the serial number of cell type; the second column represents the extent to which the expression of a target gene is affected by cellular communication in the cell type; the third column represents the extent to which the expression of a target gene is affected by the cell type itself; the fourth column represents the ratio of the two (Delta_e/(Delta_e+E0)), indicating the extent to which that the cell type is affected by cellular communication).
| | Delta_e |E0|Delta_e_proportion | Cell_type|
| --- | --- | --- | --- | --- |
| 0 |89.06278 |25.143023|0.77984464|B|
| 1 | 2561.6501 |136.53331|0.9493981|CD8T|
| 2 | 5678.756 |685.81494|0.8922449|Malignant|
| 3 | 544.611 |30.595951|0.9468088|Mono/Macro|

A visualization sample of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDCA/blob/main/IMG/cd8adeltae.png" alt="Editor" width="400">
</div>

## 3，Prioritize the dominant cell communication assmebly that regulates the key factors in specific cell type
```
cd ./src/tutorials2/ && python main.py --count /home/jby2/ScRNA_test_data_matrix.txt --meta /home/jby2/ScRNA_test_data_metadata.txt --gene GZMK --cell_type CD8T --lr_file /home/jby2/LR_test_data.csv --device cuda:1 --facked_LR 200 --repeat_num 50 --max_epoch 200 --learning_rate 1e-1 --display_loss False --ccc_ratio_result /home/jby2/ccc_ratio_result.csv --dca_rank_result /home/jby2/dca_rank_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name (Please ensure that the gene is highly variable, we detect the highly variable genes by running sc.pp.highly_variable_genes with default parameters). |
| **cell_type** | The specific cell type. |
| **device** | The device for model training (cuda or cpu, default is cpu). |
| **facked_LR** | The faked ligand and receptor genes number for removing the edges with low specificities (default is 200). |
| **repeat_num** | The repeat number for model training (default is 50). |
| **max_epoch** | The max epoch for model training (default is 200). |
| **learning_rate** | The learning rate for model training (default is 1e-1). |
| **display_loss** | Display training loss for model training (default is True).|
| **dca_rank_result** | The result filename of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result filename of ratio of different cell types affected by cellular communication. |

```
############ ------------- scDCA (the key factors in specific cell type)--------------- ############
>>> arguments <<<  Namespace(ccc_ratio_result='/home/jby2/ccc_ratio_result.csv', cell_type='CD8T', count='/home/jby2/HoloNet/github/ScRNA_test_data_matrix.txt', dca_rank_result='/home/jby2/dca_rank_result.csv', device='cuda:1', display_loss='False', facked_LR='200', gene='GZMK', learning_rate='0.1', lr_file='/home/jby2/HoloNet/github/LR_test_data.csv', max_epoch='200', meta='/home/jby2/HoloNet/github/ScRNA_test_data_metadata.txt', repeat_num='50')
>>> loading library and data <<<  Wed Aug 16 16:52:58 2023
>>> construct an AnnData object from a count file and a metadata file <<<  Wed Aug 16 16:52:58 2023
>>> load the provided dataframe with the information on ligands and receptors <<<  Wed Aug 16 16:53:01 2023
>>> calculate the CE tensor by considering the expression levels of ligand and receptor genes <<<  Wed Aug 16 16:53:01 2023

    Notes:
        This function calculates the CE tensor by considering the expression levels of ligand and receptor genes.
        If the data is large, it may require substantial memory for computation.
        We're working on improving this piece of code.
    
>>> filter the edge in calculated CE tensor, removing the edges with low specificities <<<  Wed Aug 16 16:53:06 2023

    Notes:
        This process will take a long time, if you want to reduce the calculation time, please reduce the facked_LR number, the default value is 200
    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [12:42<00:00,  4.95s/it]
>>> construct a cell type adjacency tensor based on the specific cell type and the summed LR-CE tensor. <<<  Wed Aug 16 17:05:50 2023
cell type: B
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:03<00:00, 389.17it/s]
cell type: CD8T
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:06<00:00, 211.45it/s]
cell type: Malignant
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:23<00:00, 59.02it/s]
cell type: Mono/Macro
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:03<00:00, 429.62it/s]
>>> detect the highly variable genes <<<  Wed Aug 16 17:06:28 2023
>>> start training the multi-view graph convolutional neural network <<<  Wed Aug 16 17:06:29 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:22<00:00,  1.65s/it]
>>> calculate the generated expression profile of the target gene. <<<  Wed Aug 16 17:07:51 2023
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 225.80it/s]
The mean squared error of original and predicted gene expression profiles: 0.022984229
The Pearson correlation of original and predicted gene expression profiles: 0.730637538204903
>>> the dominant cell communication assmebly that regulates the target gene expression pattern is stored at: <<<  /home/jby2/dca_rank_result.csv Wed Aug 16 17:07:51 2023
>>> the ratio of different cell types affected by cellular communication is stored at: <<<  /home/jby2/ccc_ratio_result.csv Wed Aug 16 17:07:51 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 81.21it/s]
```

A sample output result file is as follows:

**dca_rank_result.csv** （The first column represents the serial number of cell type pairs, ordered by attention weight; the second column represents the cell type pair name; the third column represents the average attention weight for 50 model repetitions of training）:
| | Cell_type_Pair |MGC_layer_attention|
| --- | --- | --- |
| 2 | Malignant:CD8T |0.5214142|
| 0 | B:CD8T |0.46100155|
| 3 | Mono/Macro:CD8T |0.43015426|
| 1 |  CD8T:CD8T|0.3521795|



A visualization sample of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDCA/blob/main/IMG/folr2tam.png" alt="Editor" width="500">
</div>

## 4，Prioritize the dominant cell communication assmebly that affected functional states of malignant cells
```
# First, we calculate functional states of malignant cells by GSVA, R package needs to be installed in advance: org.Hs.eg.db, clusterProfiler, GSVA
cd ./src/tutorials3/ && Rscript malignant_cell_states_gsva.R --count ./data/ScRNA_test_data_matrix.txt --meta ./data/ScRNA_test_data_metadata.txt --output_file_name ./output/malignant_cell_states_gsva.txt
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **output_file_name** | The output file name of results. |

```
[1] "############ ------------- TODO: calculate functional states of malignant cells --------------- ############"
[1] "############ ------------- Author:  Boya Ji && Liwen Xu  date(2023-06) --------------- ############"
[1] "############ ------------- R package needs to be installed in advance: org.Hs.eg.db, clusterProfiler, GSVA) --------------- ############"
[1] ">>> CancerSEA signature gene list ID convert <<< [2023-08-16 19:07:48]"

clusterProfiler v4.6.2  For help: https://yulab-smu.top/biomedical-knowledge-mining-book/
...
[1] "Running for human genes ..."
Loading required package: AnnotationDbi
Loading required package: stats4
Loading required package: BiocGenerics
...
[1] ">>> Extract malignant cell subsets <<< [2023-08-16 19:07:59]"
[1] 10
[1] ">>> Calculation of malignant cell state activity based on GSVA <<< [2023-08-16 19:08:12]"
Estimating GSVA scores for 14 gene sets.
Estimating ECDFs with Gaussian kernels
  |======================================================================| 100%

Warning message:
useNames = NA is deprecated. Instead, specify either useNames = TRUE or useNames = TRUE.
```

```
# Second, run tutorials3 to prioritize the dominant cell communication assmebly that affected functional states of malignant cells
cd ./src/tutorials3/ && python main.py --count /home/jby2/ScRNA_test_data_matrix.txt --meta /home/jby2/ScRNA_test_data_metadata.txt --lr_file /home/jby2/LR_test_data.csv --dca_rank_result /home/jby2/dca_rank_result.csv --facked_LR 200 --device cuda:1 --repeat_num 50 --max_epoch 200 --cell_type Malignant --cell_state_file_path /home/jby2/malignant_cell_states_gsva.txt --learning_rate 1e-1 --display_loss False --cell_state EMT
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **cell_type** | The specific cell type (Malignant here). |
| **cell_state_file_path** | The file path of functional state of malignant cells. |
| **cell_state** | The 14 kinds of functional state of malignant cells ( EMT, Metastasis, Hypoxia, Invasion, Apoptosis, DNArepair, CellCycle, DNAdamage, Stemness, Proliferation, Quiescence, Angiogenesis, Differentiation, Inflammation, Metastasis ). |
| **device** | The device for model training (cuda or cpu, default is cpu). |
| **facked_LR** | The faked ligand and receptor genes number for removing the edges with low specificities (default is 200). |
| **repeat_num** | The repeat number for model training (default is 50). |
| **max_epoch** | The max epoch for model training (default is 200). |
| **learning_rate** | The learning rate for model training (default is 1e-1). |
| **display_loss** | Display training loss for model training (default is True).|
| **dca_rank_result** | The result filename of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |

```
############ ------------- scDCA (functional states of malignant cells)--------------- ############
>>> arguments <<<  Namespace(cell_state='EMT', cell_state_file_path='/home/jby2/malignant_cell_states_gsva.txt', cell_type='Malignant', count='/home/jby2/ScRNA_test_data_matrix.txt', dca_rank_result='/home/jby2/dca_rank_result.csv', device='cuda:1', display_loss='False', facked_LR='200', learning_rate='1e-1', lr_file='/home/jby2/LR_test_data.csv', max_epoch='200', meta='/home/jby2/ScRNA_test_data_metadata.txt', repeat_num='50')
>>> loading library and data <<<  Wed Aug 16 20:01:43 2023
>>> construct an AnnData object from a count file and a metadata file <<<  Wed Aug 16 20:01:43 2023
>>> load the provided dataframe with the information on ligands and receptors <<<  Wed Aug 16 20:01:46 2023
>>> calculate the CE tensor by considering the expression levels of ligand and receptor genes <<<  Wed Aug 16 20:01:46 2023

    Notes:
        This function calculates the CE tensor by considering the expression levels of ligand and receptor genes.
        If the data is large, it may require substantial memory for computation.
        We're working on improving this piece of code.
    
>>> filter the edge in calculated CE tensor, removing the edges with low specificities <<<  Wed Aug 16 20:01:51 2023

    Notes:
        This process will take a long time, if you want to reduce the calculation time, please reduce the facked_LR number, the default value is 200
    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [12:33<00:00,  4.89s/it]
>>> construct a cell type adjacency tensor based on the specific cell type and the summed LR-CE tensor. <<<  Wed Aug 16 20:14:26 2023
cell type: B
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:14<00:00, 99.36it/s]
cell type: CD8T
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:24<00:00, 58.54it/s]
cell type: Malignant
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [01:13<00:00, 19.13it/s]
cell type: Mono/Macro
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1412/1412 [00:14<00:00, 100.68it/s]
>>> get the functional states of malignant cells in the dataset <<<  Wed Aug 16 20:16:32 2023
>>> start training the multi-view graph convolutional neural network <<<  Wed Aug 16 20:16:32 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:22<00:00,  1.65s/it]
>>> calculate the functional states of malignant cells. <<<  Wed Aug 16 20:17:55 2023
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 63.22it/s]
The mean squared error of original and predicted the functional states of malignant cells: 0.024902295
The Pearson correlation of original and predicted the functional states of malignant cells: 0.8482257796249801
>>> the dominant cell communication assmebly that affected the functional states of malignant cells is stored at: <<<  /home/jby2/dca_rank_result.csv Wed Aug 16 20:17:56 2023
```

A visualization sample of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDCA/blob/main/IMG/cellstate.png" alt="Editor" width="500">
</div>

## 5，Clinical intervertion altered effect of cell communication on gene expression
```
# run tutorials1 for untreated P76_scRNA data and P915_scRNA data, which was received the aPD-1 + aCTLA-4 with no tyrosine kinase inhibitors exposed and showed partial response.
cd ./src/tutorials1/ && python main.py --count /home/jby2/RCC_scRNA_P76_matrix.txt --meta /home/jby2/RCC_scRNA_P76_metadata.txt --gene CD8A --lr_file /home/jby2/LR_P76.csv --device cuda:1 --facked_LR 200 --repeat_num 50 --max_epoch 200 --learning_rate 1e-1 --display_loss True --ccc_ratio_result /home/jby2/P76_ccc_ratio_result.csv --dca_rank_result /home/jby2/P76_dca_rank_result.csv

cd ./src/tutorials1/ && python main.py --count /home/jby2/RCC_scRNA_P915_matrix.txt --meta /home/jby2/RCC_scRNA_P915_metadata.txt --gene CD8A --lr_file /home/jby2/LR_P915.csv --device cuda:1 --facked_LR 200 --repeat_num 50 --max_epoch 200 --learning_rate 1e-1 --display_loss True --ccc_ratio_result /home/jby2/P915_ccc_ratio_result.csv --dca_rank_result /home/jby2/P915_dca_rank_result.csv
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **count** | Count matrix / normalized count matrix path. |
| **meta** | Meta data (celltypes annotation) path. |
| **lr_file** | The final results of LR pairs. |
| **gene** | The specific target gene name (Please ensure that the gene is highly variable, we detect the highly variable genes by running sc.pp.highly_variable_genes with default parameters). |
| **device** | The device for model training (cuda or cpu, default is cpu). |
| **facked_LR** | The faked ligand and receptor genes number for removing the edges with low specificities (default is 200). |
| **repeat_num** | The repeat number for model training (default is 50). |
| **max_epoch** | The max epoch for model training (default is 200). |
| **learning_rate** | The learning rate for model training (default is 1e-1). |
| **display_loss** | Display training loss for model training (default is True).|
| **dca_rank_result** | The result filename of prioritize the dominant cell communication assmebly that regulates the target gene expression pattern. |
| **ccc_ratio_result** | The result filename of ratio of different cell types affected by cellular communication. |

A visualization sample of results:
<div align="center">
  <img src="https://github.com/jiboyalab/scDCA/blob/main/IMG/cd8arankchange.png" alt="Editor" width="500">
</div>

===========================================================================





# Contributing

All authors were involved in the conceptualization of the scDCA method.  LWX and SLP conceived and supervised the project. BYJ and LWX designed the study and developed the approach. XQW and XW collected the data. BYJ and LWX analyzed the results. BYJ, XQW, XW, LWX and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.

# Cite
<p align="center">
  <a href="https://clustrmaps.com/site/1bpq2">
     <img width="200"  src="https://clustrmaps.com/map_v2.png?cl=ffffff&w=268&t=m&d=4hIDPHzBcvyZcFn8iDMpEM-PyYTzzqGtngzRP7_HkNs" />
   </a>
</p>

<p align="center">
  <a href="#">
     <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fjiboyalab%2FscDCA&labelColor=%233499cc&countColor=%2370c168" />
   </a>
</p>


# Contacts
If you have any questions or comments, please feel free to email: byj@hnu.edu.cn.

# License

[MIT © Richard McRichface.](../LICENSE)

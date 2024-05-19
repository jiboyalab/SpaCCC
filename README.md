# SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data
===========================================================================

[![license](https://img.shields.io/badge/scGPT-black)](https://github.com/bowang-lab/scGPT)
[![license](https://img.shields.io/badge/LIANA+-pink)](https://github.com/saezlab/liana-py)
[![license](https://img.shields.io/badge/Node2vec+-red)](https://github.com/krishnanlab/node2vecplus_benchmarks?tab=readme-ov-file)
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

SpaCCC is tested to work under:

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
**Notes:** These dependencies, in order to obtain hidden embeddings of cells, ligands and receptors, are publicly available on their Github README.
* Install [scGPT](https://github.com/bowang-lab/scGPT) using ` pip install scgpt "flash-attn<1.0.5" ` if you encounter any issue. 
* Install [Node2vec+](https://github.com/krishnanlab/PecanPy) using ` pip install pecanpy ` if you encounter any issue.
* Install [LIANA+](https://github.com/saezlab/liana-py) using ` pip install liana ` if you encounter any issue.



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

## 1，Fine-tuning on Pre-trained Model for Cell-type Annotation
**Notes:** If you already have the cell type labels in your dataset (e.g. adata.obs.cell_type in the h5ad file that we provide), skip this step.
```
# The following program needs to be run in the scGPT environment, see [scGPT](https://github.com/bowang-lab/scGPT) for details on how to use it:

python /home/jby2/SpaCCC/cell_type_anno_finetune.py --filename /home/jby2/SpaCCC/data/BRCA_Visium_10x_tmp.h5ad --dataset_name BRCA_Visium_10x_tmp --load_model /home/jby2/SpaCCC/scGPT_their_example/scGPT_human --save_dir /home/jby2/SpaCCC/results
```
**Arguments**:

| **Arguments** | **Detail** |
| --- | --- |
| **filename** | The file path for single-cell RNA-seq data, requires h5ad file format |
| **dataset_name** | Dataset name |
| **load_model** | The folder path of pretained scGPT model (you can download it form [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y))|
| **save_dir** | The folder path for saving the results (the directory will automatically be created). |


```
/home/jby2/anaconda3/envs/scgptt/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
  warn(f"Failed to load image Python extension: {e}")
Global seed set to 0
############ ------------- SpaCCC --------------- ############
>>> arguments <<< 
 Namespace(filename='/home/jby2/SpaCCC/data/BRCA_Visium_10x_tmp.h5ad', dataset_name='BRCA_Visium_10x_tmp', load_model='/home/jby2/SpaCCC/scGPT_their_example/scGPT_human', save_dir='/home/jby2/SpaCCC/results')
>>> loading hyperparameter and data <<<  Sun May 19 11:17:01 2024
wandb: Currently logged in as: jby236. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.4
wandb: Run data is saved locally in /home/jby2/data/wandb/run-20240519_111703-w8koygt1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run bright-frog-221
wandb: ⭐️ View project at https://wandb.ai/jby236/scGPT
wandb: 🚀 View run at https://wandb.ai/jby236/scGPT/runs/w8koygt1
{'seed': 0, 'dataset_name': 'BRCA_Visium_10x_tmp', 'do_train': True, 'load_model': '/home/jby2/SpaCCC/scGPT_their_example/scGPT_human', 'mask_ratio': 0.0, 'epochs': 10, 'n_bins': 51, 'MVC': False, 'ecs_thres': 0.0, 'dab_weight': 0.0, 'lr': 0.0001, 'batch_size': 12, 'layer_size': 128, 'nlayers': 4, 'nhead': 4, 'dropout': 0.2, 'schedule_ratio': 0.9, 'save_eval_interval': 5, 'fast_transformer': True, 'pre_norm': False, 'amp': True, 'include_zero_gene': False, 'freeze': False, 'DSBN': False}
>>> settings for input and preprocessing <<<  Sun May 19 11:17:14 2024
>>> input/output representation <<<  Sun May 19 11:17:14 2024
>>> settings for training <<<  Sun May 19 11:17:14 2024
save to /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-11-17
>>> Load and pre-process data <<<  Sun May 19 11:17:14 2024
scGPT - INFO - match 18097/22240 genes in vocabulary of size 60697.
scGPT - INFO - Resume model from /home/jby2/SpaCCC/scGPT_their_example/scGPT_human/best_model.pt, the model args will override the config /home/jby2/SpaCCC/scGPT_their_example/scGPT_human/args.json.
>>> set up the preprocessor, use the args to config the workflow <<<  Sun May 19 11:17:14 2024
scGPT - INFO - Normalizing total counts ...
scGPT - INFO - Binning data ...
scGPT - INFO - Normalizing total counts ...
scGPT - INFO - Binning data ...
scGPT - INFO - train set number of samples: 2734, 
         feature length: 3001
scGPT - INFO - valid set number of samples: 304, 
         feature length: 3001
>>> Load the pre-trained scGPT model <<<  Sun May 19 11:17:22 2024
scGPT - INFO - Loading params encoder.embedding.weight with shape torch.Size([60697, 512])
scGPT - INFO - Loading params encoder.enc_norm.weight with shape torch.Size([512])
scGPT - INFO - Loading params encoder.enc_norm.bias with shape torch.Size([512])
scGPT - INFO - Loading params value_encoder.linear1.weight with shape torch.Size([512, 1])
scGPT - INFO - Loading params value_encoder.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params value_encoder.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params value_encoder.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params value_encoder.norm.weight with shape torch.Size([512])
scGPT - INFO - Loading params value_encoder.norm.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.0.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.0.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.0.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.0.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.0.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.0.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.1.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.1.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.1.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.1.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.1.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.1.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.2.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.2.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.2.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.2.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.2.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.2.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.3.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.3.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.3.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.3.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.3.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.3.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.4.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.4.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.4.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.4.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.4.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.4.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.5.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.5.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.5.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.5.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.5.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.5.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.6.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.6.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.6.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.6.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.6.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.6.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.7.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.7.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.7.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.7.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.7.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.7.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.8.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.8.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.8.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.8.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.8.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.8.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.9.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.9.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.9.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.9.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.9.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.9.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.10.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.10.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.10.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.10.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.10.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.10.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.self_attn.Wqkv.weight with shape torch.Size([1536, 512])
scGPT - INFO - Loading params transformer_encoder.layers.11.self_attn.Wqkv.bias with shape torch.Size([1536])
scGPT - INFO - Loading params transformer_encoder.layers.11.self_attn.out_proj.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.11.self_attn.out_proj.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.linear1.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.11.linear1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.linear2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params transformer_encoder.layers.11.linear2.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.norm1.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.norm1.bias with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.norm2.weight with shape torch.Size([512])
scGPT - INFO - Loading params transformer_encoder.layers.11.norm2.bias with shape torch.Size([512])
scGPT - INFO - Loading params decoder.fc.0.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params decoder.fc.0.bias with shape torch.Size([512])
scGPT - INFO - Loading params decoder.fc.2.weight with shape torch.Size([512, 512])
scGPT - INFO - Loading params decoder.fc.2.bias with shape torch.Size([512])
scGPT - INFO - Loading params decoder.fc.4.weight with shape torch.Size([1, 512])
scGPT - INFO - Loading params decoder.fc.4.bias with shape torch.Size([1])
--------------------
name: encoder.embedding.weight
--------------------
name: encoder.enc_norm.weight
--------------------
name: encoder.enc_norm.bias
--------------------
name: value_encoder.linear1.weight
--------------------
name: value_encoder.linear1.bias
--------------------
name: value_encoder.linear2.weight
--------------------
name: value_encoder.linear2.bias
--------------------
name: value_encoder.norm.weight
--------------------
name: value_encoder.norm.bias
--------------------
name: transformer_encoder.layers.0.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.0.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.0.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.0.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.0.linear1.weight
--------------------
name: transformer_encoder.layers.0.linear1.bias
--------------------
name: transformer_encoder.layers.0.linear2.weight
--------------------
name: transformer_encoder.layers.0.linear2.bias
--------------------
name: transformer_encoder.layers.0.norm1.weight
--------------------
name: transformer_encoder.layers.0.norm1.bias
--------------------
name: transformer_encoder.layers.0.norm2.weight
--------------------
name: transformer_encoder.layers.0.norm2.bias
--------------------
name: transformer_encoder.layers.1.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.1.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.1.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.1.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.1.linear1.weight
--------------------
name: transformer_encoder.layers.1.linear1.bias
--------------------
name: transformer_encoder.layers.1.linear2.weight
--------------------
name: transformer_encoder.layers.1.linear2.bias
--------------------
name: transformer_encoder.layers.1.norm1.weight
--------------------
name: transformer_encoder.layers.1.norm1.bias
--------------------
name: transformer_encoder.layers.1.norm2.weight
--------------------
name: transformer_encoder.layers.1.norm2.bias
--------------------
name: transformer_encoder.layers.2.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.2.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.2.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.2.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.2.linear1.weight
--------------------
name: transformer_encoder.layers.2.linear1.bias
--------------------
name: transformer_encoder.layers.2.linear2.weight
--------------------
name: transformer_encoder.layers.2.linear2.bias
--------------------
name: transformer_encoder.layers.2.norm1.weight
--------------------
name: transformer_encoder.layers.2.norm1.bias
--------------------
name: transformer_encoder.layers.2.norm2.weight
--------------------
name: transformer_encoder.layers.2.norm2.bias
--------------------
name: transformer_encoder.layers.3.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.3.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.3.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.3.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.3.linear1.weight
--------------------
name: transformer_encoder.layers.3.linear1.bias
--------------------
name: transformer_encoder.layers.3.linear2.weight
--------------------
name: transformer_encoder.layers.3.linear2.bias
--------------------
name: transformer_encoder.layers.3.norm1.weight
--------------------
name: transformer_encoder.layers.3.norm1.bias
--------------------
name: transformer_encoder.layers.3.norm2.weight
--------------------
name: transformer_encoder.layers.3.norm2.bias
--------------------
name: transformer_encoder.layers.4.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.4.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.4.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.4.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.4.linear1.weight
--------------------
name: transformer_encoder.layers.4.linear1.bias
--------------------
name: transformer_encoder.layers.4.linear2.weight
--------------------
name: transformer_encoder.layers.4.linear2.bias
--------------------
name: transformer_encoder.layers.4.norm1.weight
--------------------
name: transformer_encoder.layers.4.norm1.bias
--------------------
name: transformer_encoder.layers.4.norm2.weight
--------------------
name: transformer_encoder.layers.4.norm2.bias
--------------------
name: transformer_encoder.layers.5.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.5.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.5.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.5.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.5.linear1.weight
--------------------
name: transformer_encoder.layers.5.linear1.bias
--------------------
name: transformer_encoder.layers.5.linear2.weight
--------------------
name: transformer_encoder.layers.5.linear2.bias
--------------------
name: transformer_encoder.layers.5.norm1.weight
--------------------
name: transformer_encoder.layers.5.norm1.bias
--------------------
name: transformer_encoder.layers.5.norm2.weight
--------------------
name: transformer_encoder.layers.5.norm2.bias
--------------------
name: transformer_encoder.layers.6.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.6.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.6.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.6.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.6.linear1.weight
--------------------
name: transformer_encoder.layers.6.linear1.bias
--------------------
name: transformer_encoder.layers.6.linear2.weight
--------------------
name: transformer_encoder.layers.6.linear2.bias
--------------------
name: transformer_encoder.layers.6.norm1.weight
--------------------
name: transformer_encoder.layers.6.norm1.bias
--------------------
name: transformer_encoder.layers.6.norm2.weight
--------------------
name: transformer_encoder.layers.6.norm2.bias
--------------------
name: transformer_encoder.layers.7.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.7.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.7.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.7.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.7.linear1.weight
--------------------
name: transformer_encoder.layers.7.linear1.bias
--------------------
name: transformer_encoder.layers.7.linear2.weight
--------------------
name: transformer_encoder.layers.7.linear2.bias
--------------------
name: transformer_encoder.layers.7.norm1.weight
--------------------
name: transformer_encoder.layers.7.norm1.bias
--------------------
name: transformer_encoder.layers.7.norm2.weight
--------------------
name: transformer_encoder.layers.7.norm2.bias
--------------------
name: transformer_encoder.layers.8.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.8.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.8.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.8.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.8.linear1.weight
--------------------
name: transformer_encoder.layers.8.linear1.bias
--------------------
name: transformer_encoder.layers.8.linear2.weight
--------------------
name: transformer_encoder.layers.8.linear2.bias
--------------------
name: transformer_encoder.layers.8.norm1.weight
--------------------
name: transformer_encoder.layers.8.norm1.bias
--------------------
name: transformer_encoder.layers.8.norm2.weight
--------------------
name: transformer_encoder.layers.8.norm2.bias
--------------------
name: transformer_encoder.layers.9.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.9.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.9.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.9.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.9.linear1.weight
--------------------
name: transformer_encoder.layers.9.linear1.bias
--------------------
name: transformer_encoder.layers.9.linear2.weight
--------------------
name: transformer_encoder.layers.9.linear2.bias
--------------------
name: transformer_encoder.layers.9.norm1.weight
--------------------
name: transformer_encoder.layers.9.norm1.bias
--------------------
name: transformer_encoder.layers.9.norm2.weight
--------------------
name: transformer_encoder.layers.9.norm2.bias
--------------------
name: transformer_encoder.layers.10.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.10.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.10.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.10.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.10.linear1.weight
--------------------
name: transformer_encoder.layers.10.linear1.bias
--------------------
name: transformer_encoder.layers.10.linear2.weight
--------------------
name: transformer_encoder.layers.10.linear2.bias
--------------------
name: transformer_encoder.layers.10.norm1.weight
--------------------
name: transformer_encoder.layers.10.norm1.bias
--------------------
name: transformer_encoder.layers.10.norm2.weight
--------------------
name: transformer_encoder.layers.10.norm2.bias
--------------------
name: transformer_encoder.layers.11.self_attn.Wqkv.weight
--------------------
name: transformer_encoder.layers.11.self_attn.Wqkv.bias
--------------------
name: transformer_encoder.layers.11.self_attn.out_proj.weight
--------------------
name: transformer_encoder.layers.11.self_attn.out_proj.bias
--------------------
name: transformer_encoder.layers.11.linear1.weight
--------------------
name: transformer_encoder.layers.11.linear1.bias
--------------------
name: transformer_encoder.layers.11.linear2.weight
--------------------
name: transformer_encoder.layers.11.linear2.bias
--------------------
name: transformer_encoder.layers.11.norm1.weight
--------------------
name: transformer_encoder.layers.11.norm1.bias
--------------------
name: transformer_encoder.layers.11.norm2.weight
--------------------
name: transformer_encoder.layers.11.norm2.bias
--------------------
name: decoder.fc.0.weight
--------------------
name: decoder.fc.0.bias
--------------------
name: decoder.fc.2.weight
--------------------
name: decoder.fc.2.bias
--------------------
name: decoder.fc.4.weight
--------------------
name: decoder.fc.4.bias
--------------------
name: cls_decoder._decoder.0.weight
--------------------
name: cls_decoder._decoder.0.bias
--------------------
name: cls_decoder._decoder.2.weight
--------------------
name: cls_decoder._decoder.2.bias
--------------------
name: cls_decoder._decoder.3.weight
--------------------
name: cls_decoder._decoder.3.bias
--------------------
name: cls_decoder._decoder.5.weight
--------------------
name: cls_decoder._decoder.5.bias
--------------------
name: cls_decoder.out_layer.weight
--------------------
name: cls_decoder.out_layer.bias
scGPT - INFO - Total Pre freeze Params 51336202
scGPT - INFO - Total Post freeze Params 51336202
>>> Finetune scGPT with task-specific objectives <<<  Sun May 19 11:17:24 2024
random masking at epoch   1, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   1 | 100/228 batches | lr 0.0001 | ms/batch 368.21 | loss  1.59 | cls  1.59 | err  0.50 | 
scGPT - INFO - | epoch   1 | 200/228 batches | lr 0.0001 | ms/batch 361.07 | loss  1.24 | cls  1.24 | err  0.40 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   1 | time: 85.86s | valid loss/mse 1.1115 | err 0.3553
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 1.1115
random masking at epoch   2, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   2 | 100/228 batches | lr 0.0001 | ms/batch 367.99 | loss  1.03 | cls  1.03 | err  0.35 | 
scGPT - INFO - | epoch   2 | 200/228 batches | lr 0.0001 | ms/batch 364.89 | loss  0.88 | cls  0.88 | err  0.31 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   2 | time: 86.29s | valid loss/mse 0.7274 | err 0.2664
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.7274
random masking at epoch   3, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   3 | 100/228 batches | lr 0.0001 | ms/batch 370.11 | loss  0.81 | cls  0.81 | err  0.28 | 
scGPT - INFO - | epoch   3 | 200/228 batches | lr 0.0001 | ms/batch 365.53 | loss  0.76 | cls  0.76 | err  0.27 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   3 | time: 86.58s | valid loss/mse 0.6702 | err 0.2632
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.6702
random masking at epoch   4, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   4 | 100/228 batches | lr 0.0001 | ms/batch 370.92 | loss  0.63 | cls  0.63 | err  0.22 | 
scGPT - INFO - | epoch   4 | 200/228 batches | lr 0.0001 | ms/batch 366.96 | loss  0.66 | cls  0.66 | err  0.23 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   4 | time: 86.87s | valid loss/mse 0.5782 | err 0.2007
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.5782
random masking at epoch   5, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   5 | 100/228 batches | lr 0.0001 | ms/batch 374.61 | loss  0.57 | cls  0.57 | err  0.20 | 
scGPT - INFO - | epoch   5 | 200/228 batches | lr 0.0001 | ms/batch 367.27 | loss  0.54 | cls  0.54 | err  0.18 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   5 | time: 87.29s | valid loss/mse 0.5129 | err 0.1875
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.5129
random masking at epoch   6, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   6 | 100/228 batches | lr 0.0001 | ms/batch 371.06 | loss  0.47 | cls  0.47 | err  0.17 | 
scGPT - INFO - | epoch   6 | 200/228 batches | lr 0.0001 | ms/batch 366.60 | loss  0.44 | cls  0.44 | err  0.16 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   6 | time: 86.81s | valid loss/mse 0.5146 | err 0.1908
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   7, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   7 | 100/228 batches | lr 0.0001 | ms/batch 371.34 | loss  0.39 | cls  0.39 | err  0.14 | 
scGPT - INFO - | epoch   7 | 200/228 batches | lr 0.0001 | ms/batch 366.52 | loss  0.35 | cls  0.35 | err  0.12 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   7 | time: 86.88s | valid loss/mse 0.6900 | err 0.2105
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   8, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   8 | 100/228 batches | lr 0.0000 | ms/batch 371.22 | loss  0.37 | cls  0.37 | err  0.13 | 
scGPT - INFO - | epoch   8 | 200/228 batches | lr 0.0000 | ms/batch 366.34 | loss  0.30 | cls  0.30 | err  0.11 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   8 | time: 86.80s | valid loss/mse 0.7080 | err 0.2039
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   9, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   9 | 100/228 batches | lr 0.0000 | ms/batch 370.82 | loss  0.33 | cls  0.33 | err  0.11 | 
scGPT - INFO - | epoch   9 | 200/228 batches | lr 0.0000 | ms/batch 367.14 | loss  0.30 | cls  0.30 | err  0.11 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   9 | time: 86.83s | valid loss/mse 0.7576 | err 0.1908
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch  10, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch  10 | 100/228 batches | lr 0.0000 | ms/batch 370.83 | loss  0.24 | cls  0.24 | err  0.08 | 
scGPT - INFO - | epoch  10 | 200/228 batches | lr 0.0000 | ms/batch 365.80 | loss  0.26 | cls  0.26 | err  0.08 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch  10 | time: 86.76s | valid loss/mse 0.9935 | err 0.1941
scGPT - INFO - -----------------------------------------------------------------------------------------
>>> Inference with fine-tuned scGPT model <<<  Sun May 19 11:31:51 2024
scGPT - INFO - Accuracy: 0.820, Precision: 0.777, Recall: 0.676, Macro F1: 0.707
WARNING: saving figure to file /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-11-17/showtest_cell_type_results.pdf
WARNING: saving figure to file /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-11-17/showprediction_cell_type_results.pdf
WARNING: saving figure to file /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-11-17/showcell_type_results.pdf
>>> Save the model into the save_dir <<<  Sun May 19 11:32:04 2024
wandb: \ 0.282 MB of 0.282 MB uploaded
wandb: Run history:
wandb:                        epoch ▁▂▃▃▄▅▆▆▇██
wandb: info/post_freeze_param_count ▁
wandb:  info/pre_freeze_param_count ▁
wandb:                test/accuracy ▁
wandb:                test/macro_f1 ▁
wandb:               test/precision ▁
wandb:                  test/recall ▁
wandb:                    train/cls █▆██▅▄▅▃▅▅▄▄▄▅▂▅▄▂▂▃▃▃▂▂▂▅▄▄▄▂▂▁▂▂▂▄▃▁▁▁
wandb:                    valid/dab ▁▁▁▁▁▁▁▁▁▁▁
wandb:                    valid/err █▄▄▂▁▁▂▂▁▁▂
wandb:                    valid/mse █▄▃▂▁▁▃▃▄▇▂
wandb:            valid/sum_mse_dab █▄▃▂▁▁▃▃▄▇▂
wandb: 
wandb: Run summary:
wandb:                        epoch 10
wandb: info/post_freeze_param_count 51336202
wandb:  info/pre_freeze_param_count 51336202
wandb:                test/accuracy 0.82074
wandb:                test/macro_f1 0.70775
wandb:               test/precision 0.77764
wandb:                  test/recall 0.67672
wandb:                    train/cls 0.00378
wandb:                    valid/err 0.18526
wandb: 
wandb: 🚀 View run bright-frog-221 at: https://wandb.ai/jby236/scGPT/runs/w8koygt1
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240519_111703-w8koygt1/logs
```






## 2，Fine-tuning on Pre-trained Model for Obtaining Gene embeddings
```
# The following program needs to be run in the scGPT environment, see [scGPT](https://github.com/bowang-lab/scGPT) for details on how to use it:

python /home/jby2/SpaCCC/fine_tuning_gene_embeddings.py --filename /home/jby2/SpaCCC/data/BRCA_Visium_10x_tmp.h5ad --dataset_name BRCA_Visium_10x_tmp --load_model /home/jby2/SpaCCC/scGPT_their_example/scGPT_human --save_dir /home/jby2/SpaCCC/results
python /home/jby2/SpaCCC/grn_get_embedding.py --filename /home/jby2/SpaCCC/data/BRCA_Visium_10x_tmp.h5ad --model_dir /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-15-17
```
**Arguments**:


| **Arguments** | **Detail** |
| --- | --- |
| **filename** | The file path for single-cell RNA-seq data, requires h5ad file format |
| **dataset_name** | Dataset name |
| **load_model** | The folder path of pretained scGPT model (you can download it form [link](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y))|
| **save_dir** | The folder path for saving the results (the directory will automatically be created). |
| **model_dir** | The folder path of fine tuned scGPT model. |
```
/home/jby2/anaconda3/envs/scgptt/lib/python3.9/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
  warn(f"Failed to load image Python extension: {e}")
Global seed set to 0
############ ------------- SpaCCC --------------- ############
>>> arguments <<< 
 Namespace(filename='/home/jby2/SpaCCC/data/BRCA_Visium_10x_tmp.h5ad', dataset_name='BRCA_Visium_10x_tmp', load_model='/home/jby2/SpaCCC/scGPT_their_example/scGPT_human', save_dir='/home/jby2/SpaCCC/results')
>>> loading hyperparameter and data <<<  Sun May 19 15:17:11 2024
wandb: Currently logged in as: jby236. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.4
wandb: Run data is saved locally in /home/jby2/data/wandb/run-20240519_151714-2plynghc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run apricot-hill-224
wandb: ⭐️ View project at https://wandb.ai/jby236/scGPT
wandb: 🚀 View run at https://wandb.ai/jby236/scGPT/runs/2plynghc
{'seed': 0, 'dataset_name': 'BRCA_Visium_10x_tmp', 'do_train': True, 'load_model': '/home/jby2/scGPT_their_example/scGPT_human', 'mask_ratio': 0.0, 'epochs': 10, 'n_bins': 51, 'MVC': False, 'ecs_thres': 0.0, 'dab_weight': 0.0, 'lr': 0.0001, 'batch_size': 12, 'layer_size': 128, 'nlayers': 4, 'nhead': 4, 'dropout': 0.2, 'schedule_ratio': 0.9, 'save_eval_interval': 5, 'fast_transformer': True, 'pre_norm': False, 'amp': True, 'include_zero_gene': False, 'freeze': False, 'DSBN': False}
>>> settings for input and preprocessing <<<  Sun May 19 15:17:26 2024
>>> input/output representation <<<  Sun May 19 15:17:26 2024
>>> settings for training <<<  Sun May 19 15:17:26 2024
save to /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-15-17
>>> Load and pre-process data <<<  Sun May 19 15:17:26 2024
scGPT - INFO - match 18097/22240 genes in vocabulary of size 60697.
scGPT - INFO - Resume model from /home/jby2/SpaCCC/scGPT_their_example/scGPT_human/best_model.pt, the model args will override the config /home/jby2/SpaCCC/scGPT_their_example/scGPT_human/args.json.
>>> set up the preprocessor, use the args to config the workflow <<<  Sun May 19 15:17:26 2024
scGPT - INFO - Normalizing total counts ...
scGPT - INFO - Binning data ...
scGPT - INFO - Normalizing total counts ...
scGPT - INFO - Binning data ...
scGPT - INFO - train set number of samples: 2734, 
         feature length: 3001
scGPT - INFO - valid set number of samples: 304, 
         feature length: 3001
>>> Load the pre-trained scGPT model <<<  Sun May 19 15:17:34 2024
scGPT - INFO - Loading params encoder.embedding.weight with shape torch.Size([60697, 512])
scGPT - INFO - Loading params encoder.enc_norm.weight with shape torch.Size([512])
...
name: cls_decoder._decoder.5.bias
--------------------
name: cls_decoder.out_layer.weight
--------------------
name: cls_decoder.out_layer.bias
scGPT - INFO - Total Pre freeze Params 51336202
scGPT - INFO - Total Post freeze Params 51336202
>>> Finetune scGPT with task-specific objectives <<<  Sun May 19 15:17:36 2024
random masking at epoch   1, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   1 | 100/228 batches | lr 0.0001 | ms/batch 371.36 | loss  1.59 | cls  1.59 | err  0.50 | 
scGPT - INFO - | epoch   1 | 200/228 batches | lr 0.0001 | ms/batch 363.57 | loss  1.24 | cls  1.24 | err  0.40 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   1 | time: 86.51s | valid loss/mse 1.1115 | err 0.3553
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 1.1115
random masking at epoch   2, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   2 | 100/228 batches | lr 0.0001 | ms/batch 370.45 | loss  1.03 | cls  1.03 | err  0.35 | 
scGPT - INFO - | epoch   2 | 200/228 batches | lr 0.0001 | ms/batch 365.53 | loss  0.88 | cls  0.88 | err  0.31 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   2 | time: 86.66s | valid loss/mse 0.7274 | err 0.2664
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.7274
random masking at epoch   3, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   3 | 100/228 batches | lr 0.0001 | ms/batch 371.72 | loss  0.81 | cls  0.81 | err  0.28 | 
scGPT - INFO - | epoch   3 | 200/228 batches | lr 0.0001 | ms/batch 366.72 | loss  0.76 | cls  0.76 | err  0.27 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   3 | time: 86.95s | valid loss/mse 0.6702 | err 0.2632
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.6702
random masking at epoch   4, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   4 | 100/228 batches | lr 0.0001 | ms/batch 372.03 | loss  0.63 | cls  0.63 | err  0.22 | 
scGPT - INFO - | epoch   4 | 200/228 batches | lr 0.0001 | ms/batch 367.04 | loss  0.66 | cls  0.66 | err  0.23 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   4 | time: 87.02s | valid loss/mse 0.5782 | err 0.2007
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.5782
random masking at epoch   5, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   5 | 100/228 batches | lr 0.0001 | ms/batch 374.14 | loss  0.57 | cls  0.57 | err  0.20 | 
scGPT - INFO - | epoch   5 | 200/228 batches | lr 0.0001 | ms/batch 367.74 | loss  0.54 | cls  0.54 | err  0.18 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   5 | time: 87.26s | valid loss/mse 0.5129 | err 0.1875
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - Best model with score 0.5129
random masking at epoch   6, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   6 | 100/228 batches | lr 0.0001 | ms/batch 372.35 | loss  0.47 | cls  0.47 | err  0.17 | 
scGPT - INFO - | epoch   6 | 200/228 batches | lr 0.0001 | ms/batch 367.12 | loss  0.44 | cls  0.44 | err  0.16 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   6 | time: 87.06s | valid loss/mse 0.5146 | err 0.1908
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   7, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   7 | 100/228 batches | lr 0.0001 | ms/batch 372.07 | loss  0.39 | cls  0.39 | err  0.14 | 
scGPT - INFO - | epoch   7 | 200/228 batches | lr 0.0001 | ms/batch 366.75 | loss  0.35 | cls  0.35 | err  0.12 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   7 | time: 86.95s | valid loss/mse 0.6900 | err 0.2105
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   8, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   8 | 100/228 batches | lr 0.0000 | ms/batch 371.68 | loss  0.37 | cls  0.37 | err  0.13 | 
scGPT - INFO - | epoch   8 | 200/228 batches | lr 0.0000 | ms/batch 366.29 | loss  0.30 | cls  0.30 | err  0.11 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   8 | time: 86.85s | valid loss/mse 0.7080 | err 0.2039
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch   9, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch   9 | 100/228 batches | lr 0.0000 | ms/batch 371.61 | loss  0.33 | cls  0.33 | err  0.11 | 
scGPT - INFO - | epoch   9 | 200/228 batches | lr 0.0000 | ms/batch 367.19 | loss  0.30 | cls  0.30 | err  0.11 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch   9 | time: 86.92s | valid loss/mse 0.7576 | err 0.1908
scGPT - INFO - -----------------------------------------------------------------------------------------
random masking at epoch  10, ratio of masked values in train:  0.0000
scGPT - INFO - | epoch  10 | 100/228 batches | lr 0.0000 | ms/batch 371.57 | loss  0.24 | cls  0.24 | err  0.08 | 
scGPT - INFO - | epoch  10 | 200/228 batches | lr 0.0000 | ms/batch 365.88 | loss  0.26 | cls  0.26 | err  0.08 | 
scGPT - INFO - -----------------------------------------------------------------------------------------
scGPT - INFO - | end of epoch  10 | time: 86.82s | valid loss/mse 0.9935 | err 0.1941
scGPT - INFO - -----------------------------------------------------------------------------------------
>>> Inference with fine-tuned scGPT model <<<  Sun May 19 15:32:05 2024
>>> Save the model into the save_dir <<<  Sun May 19 15:32:18 2024

```
```
############ ------------- SpaCCC --------------- ############
>>> Load pre-trained model <<<  Sun May 19 16:27:10 2024
>>> Retrieve model parameters from config files <<<  Sun May 19 16:27:10 2024
Resume model from /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-15-17/best_model.pt, the model args will override the config /home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-15-17/args.json.
Loading params encoder.embedding.weight with shape torch.Size([60697, 512])
Loading params encoder.enc_norm.weight with shape torch.Size([512])
...
Loading params cls_decoder.out_layer.weight with shape torch.Size([1, 512])
Loading params cls_decoder.out_layer.bias with shape torch.Size([1])
scGPT - INFO - Binning data ...
>>> Retrieve scGPT's gene embeddings <<<  Sun May 19 16:27:23 2024
Retrieved gene embeddings for 17674 genes.
>>> Save gene embeddings to <<<  Sun May 19 16:27:40 2024
/home/jby2/SpaCCC/results/dev_BRCA_Visium_10x_tmp-May19-15-17/all_gene_embedding.csv
```



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

All authors were involved in the conceptualization of the SpaCCC method.  LWX and SLP conceived and supervised the project. BYJ and LWX designed the study and developed the approach. BYJ and DBQ collected the data. BYJ and LWX analyzed the results. BYJ, DBQ, LWX and SLP contributed to the review of the manuscript before submission for publication. All authors read and approved the final manuscript.

# Views
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

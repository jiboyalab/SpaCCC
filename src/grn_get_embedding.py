import copy
import json
import os
from pathlib import Path
import sys
import warnings
import argparse,time
import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins
###########################-main-##############################
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data.")
    parser.add_argument("--filename", required=True, help="The file path for single-cell RNA-seq data, requires h5ad file format")
    
    parser.add_argument("--model_dir", required=True, help="The folder path of pretained scGPT model")
    args = parser.parse_args()
    filename=args.filename
    model_dir=args.model_dir
    print('############ ------------- SpaCCC --------------- ############')
    
    # 1.1 Load pre-trained model
    print('>>> Load pre-trained model <<< ', time.ctime())
    # Specify model path; here we load the pre-trained scGPT blood model
    model_dir = Path(model_dir)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    print('>>> Retrieve model parameters from config files <<< ', time.ctime())
    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    gene2idx = vocab.get_stoi()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)



    # 1.2 Load dataset of interest


    # Specify data path; here we load the Immune Human dataset

    adata = sc.read(filename)
    ori_batch_col = "batch"
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata.obs['batch'] = 0
    data_is_raw = False
    # make the batch category column
    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()




    # Preprocess the data following the scGPT data pre-processing pipeline
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=False,  # step 1
        filter_cell_by_counts=False,  # step 2
        # normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        normalize_total=False, #数据集已经做过预处理lognomallized
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        # subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        subset_hvg=False,
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="str_batch")

    print('>>> Retrieve scGPT\'s gene embeddings <<< ', time.ctime())
    # Step 2: Retrieve scGPT’s gene embeddings
    # Retrieve the data-independent gene embeddings from scGPT
    gene_ids = np.array([id for id in gene2idx.values()])
    gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()


    # Filter on the intersection between the Immune Human HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
    gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
    print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))
    import csv
    print('>>> Save gene embeddings to <<< ', time.ctime())
    # 指定要保存的文件名
    csv_file =  model_dir / "all_gene_embedding.csv"
    print(csv_file)
    print("\n")
    # 获取字典的所有键和对应的值
    keys = list(gene_embeddings.keys())
    values = list(gene_embeddings.values())
    # 获取数组的长度，假设所有数组长度相同
    keys_length=len(keys)
    array_length = len(values[0])

    # 打开 CSV 文件进行写入
    with open(csv_file, 'w', newline='') as csvfile:
        # 创建 CSV writer 对象
        csv_writer = csv.writer(csvfile)

        # 写入每一行的数据
        for i in range(keys_length):
                row_data = [keys[i]] + [str(values[i][j]) for j in range(array_length)]
                csv_writer.writerow(row_data)





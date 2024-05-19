import os,torch
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from anndata._core.anndata import AnnData
from scipy.spatial.distance import squareform, pdist
from typing import List
from copy import deepcopy
from tqdm import tqdm
import argparse
def default_w_visium(adata: AnnData,
                     min_cell_distance: int = 100,
                     cover_distance: int = 255,
                     obsm_spatial_slot: str = 'spatial',
                     ) -> float:
    """\
    Calculate a recommended value for the distance parameter in the edge weighting function.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_cell_distance
        The min distance between spots is 100 micrometers in 10x Visium technology.
    cover_distance
        Ligands cover a region with 255 micrometers diameter at a fixed concentration by default.
        The diameter of sender spot is 55 micrometers, and the ligands spread 100 micrometers.
    obsm_spatial_slot
        The slot name storing the spatial position information of each spot

    Returns
    -------
    A integer for a recommended value for the distance parameter in the edge weighting function.
    """

    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))

    # ligands cover 255 micrometers by default, and the min value of distance between spot is 100 micrometers
    w_best = cover_distance * (dist_mat[dist_mat > 0].min() / min_cell_distance) / np.sqrt(np.pi)

    return w_best

def compute_ce_tensor(adata: AnnData,
                      lr_df: pd.DataFrame,
                      w_best: float,
                      distinguish: bool = True,
                      ) -> torch.Tensor:
    """\
    Calculate CE matrix for measuring the strength of communication between any pairs of cells, according to
    the edge weighting function.

    Parameters
    ----------
    adata
        Annotated data matrix.
    lr_df
        A preprocessed LR-gene dataframe.
        must contain three columns: 'Ligand_gene_symbol', 'Receptor_gene_symbol' and 'LR_pair'.
    w_best
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    distinguish:
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    
    Returns
    -------
    A CE tensor (LR_pair_num * cell_num * cell_num)
    
    """

    dist_factor_tensor = distinguish_dist_factor_calculate(adata=adata,
                                                           lr_df=lr_df,
                                                           w_best=w_best,
                                                           distinguish=distinguish)

    expressed_ligand = lr_df.loc[:, 'ligand'].tolist()
    expressed_receptor = lr_df.loc[:, 'receptor'].tolist()

    expressed_ligand_tensor = get_gene_expr_tensor(adata, expressed_ligand)
    expressed_receptor_tensor = get_gene_expr_tensor(adata, expressed_receptor).permute(0, 2, 1)

    ce_tensor = expressed_ligand_tensor.mul(dist_factor_tensor).mul(expressed_receptor_tensor)
    ce_tensor = ce_tensor / ce_tensor.mean((1, 2)).unsqueeze(1).unsqueeze(2)

    return ce_tensor.to(torch.float32)

def distinguish_dist_factor_calculate(adata: AnnData,
                                      lr_df: pd.DataFrame,
                                      w_best: float,
                                      distinguish=False,
                                      ) -> torch.Tensor:
    if distinguish:
        w_best2 = w_best * 2
        dist_factor1 = dist_factor_calculate(adata=adata, w_best=w_best, )
        dist_factor2 = dist_factor_calculate(adata=adata, w_best=w_best2, )
        dist_factor_tensor = dist_factor1.repeat(lr_df.shape[0], 1, 1)
        secreted_index = lr_df[lr_df.loc[:,"secreted"]==True].index
        dist_factor_tensor[secreted_index, :, :] = dist_factor2
    else:
        dist_factor_tensor = dist_factor_calculate(adata, w_best=w_best, )

    return dist_factor_tensor

def dist_factor_calculate(adata: AnnData,
                          w_best: float,
                          obsm_spatial_slot: str = 'spatial',
                          ) -> torch.Tensor:
    """

    Parameters
    ----------
    adata :
        Annotated data matrix.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    obsm_spatial_slot :
        The slot name storing the spatial position information of each spot。

    Returns
    -------
    A tensor describing the distance factor.

    """
    position_mat = adata.obsm[obsm_spatial_slot]
    dist_mat = squareform(pdist(position_mat, metric='euclidean'))
    dist_factor = dist_mat / w_best

    dist_factor = np.exp((-1) * dist_factor * dist_factor)
    dist_factor = torch.tensor(dist_factor).to(torch.float32)

    return dist_factor


def get_gene_expr_tensor(adata: AnnData,
                         gene_name: List[str],
                         ) -> torch.Tensor:
    gene_expr_mat = adata[:, gene_name].X.toarray().astype(np.float32)
    gene_expr_tensor = torch.tensor(np.expand_dims(gene_expr_mat, 2)).permute(1, 2, 0)

    return gene_expr_tensor

def filter_ce_tensor(ce_tensor: torch.Tensor,
                     adata: AnnData,
                     lr_df: pd.DataFrame,
                     w_best: float,
                     n_pairs: int = 200,
                     thres: float = 0.05,
                     distinguish: bool = True,
                     copy: bool = True,
                     ) -> torch.Tensor:
    """
    Filter the edge in calculated CE tensor, removing the edges with low specificities.

    For each LR pair, select faked ligand and receptor genes, which have similar expression levels
    with the ligand and receptor gene in the dataset. Then calculate the background CE tensor using faked LR genes,

    Using permutation tests, require filtered edges with communication event strength larger than
    a proportion of background strengthes.

    Parameters
    ----------
    ce_tensor :
        Calculated CE tensor (LR_pair_num * cell_num * cell_num) by "compute_sc_CE_tensor" function
    adata :
        Annotated data matrix.
    lr_df :
        A preprocessed LR-gene dataframe.
    w_best :
        A distance parameter in edge weighting function controlling the covering region of ligands.
        'default_w_visium' function provides a recommended value of w_best.
    n_pairs :
        The number of faked ligand and receptor genes.
    thres :
        We require filtered edges with communicatin event strength larger than a proportion of background strengthes.
        The parameter is the proportion.
    distinguish :
        If True, set the different w_best for secreted ligands and plasma-membrane-binding ligands.
    copy :
        If False, change the input ce_tensor and save memory consumption

    Returns
    -------
    A CE tensor which removed the edges with low specificities (LR_pair_num * cell_num * cell_num)

    """
    if copy:
        if ce_tensor.is_sparse:
            ce_tensor = deepcopy(ce_tensor.to_dense())
        else:
            ce_tensor = deepcopy(ce_tensor)
    all_genes = [item for item in adata.var_names.tolist()
                 if not (item.startswith("MT-") or item.startswith("MT_"))]
    means = adata.to_df()[all_genes].mean().sort_values()

    for i in tqdm(range(lr_df.shape[0])):
        dist_factor_tensor = distinguish_dist_factor_calculate(adata=adata,
                                                               lr_df=lr_df.iloc[i:i + 1, :].reset_index(drop=True),
                                                               w_best=w_best,
                                                               distinguish=distinguish)

        lr1 = lr_df.ligand[i]
        lr2 = lr_df.receptor[i]
        i1, i2 = means.index.get_loc(lr1), means.index.get_loc(lr2)
        i1, i2 = np.sort([i1, i2])
        im = np.argmin(abs(means.values - means.iloc[i1:i2].median()))

        selected = (
            abs(means - means.iloc[im])
                .sort_values()
                .drop([lr1, lr2])[: n_pairs * 2]
                .index.tolist()
        )
        faked_ligand = selected[-n_pairs:]
        faked_receptor = selected[:n_pairs]

        faked_expressed_ligand_tensor = get_gene_expr_tensor(adata, faked_ligand)
        faked_expressed_receptor_tensor = get_gene_expr_tensor(adata, faked_receptor).permute(0, 2, 1)
        faked_ce_tensor = faked_expressed_ligand_tensor.mul(dist_factor_tensor).mul(faked_expressed_receptor_tensor)

        expressed_ligand_tensor = get_gene_expr_tensor(adata, lr1)
        expressed_receptor_tensor = get_gene_expr_tensor(adata, lr2).permute(0, 2, 1)
        true_ce_tensor = expressed_ligand_tensor.mul(dist_factor_tensor).mul(expressed_receptor_tensor)

        tmp = (true_ce_tensor > faked_ce_tensor).sum(0) > n_pairs * (1 - thres)
        ce_tensor[i, :, :] = ce_tensor[i, :, :].mul(tmp)

    return ce_tensor

def construct_lr_adj(adata,Sum_LR_CE_tensor,LR_df):
    #计算每对lr在不同的cell type对下的通讯强度
    #因为运行时间太久，而且不需要这多种细胞类型的组合
    # cell_type=np.unique(adata.obs["cell_type"].values)
    # num=-1
    # index1=[]
    # cell_type_name=[]
    # for i in range(len(cell_type)):#所有的细胞类型的组合构建
    #     for j in range(len(cell_type)):
    #         num+=1
    #         index1.append(num)
    #         cell_type_name.append(str(cell_type[i])+":"+str(cell_type[j]))
    cell_type=np.unique(adata.obs["cell_type"].values)
    num=-1
    index1=[]
    cell_type_name=[]
    for i in range(len(cell_type)):#所有的细胞类型的组合构建
            num+=1
            index1.append(num)
            cell_type_name.append("Tcell:"+str(cell_type[i]))
            cell_type_name.append(str(cell_type[i])+":"+"Tcell")
    cell_type_name=list(set(cell_type_name))
    
    kong=pd.DataFrame(columns=['LR','cell_type_pair','strength']) #创建一个空的pandas

    for i in range(LR_df.shape[0]):
        print(i)
        #先一个一个lr的循环
        Sum_LR_CE_tensor_tep=Sum_LR_CE_tensor[i]
        for j in range(len(cell_type_name)):
            #再循环所有的细胞类型组合
            celltype1,celltype2=cell_type_name[j].split(":")
            strength=0
            for n in range(Sum_LR_CE_tensor_tep.shape[0]):#循环行
                    if adata.obs["cell_type"][n] == celltype1:
                        
                        for m in range(Sum_LR_CE_tensor_tep.shape[1]):#循环列
                            if adata.obs["cell_type"][m] == celltype2:
                                strength+=float(Sum_LR_CE_tensor_tep[n][m])

            new_data = pd.DataFrame({'LR': LR_df.loc[i,"ligand"]+":"+LR_df.loc[i,"receptor"], 'cell_type_pair': cell_type_name[j], 'strength': strength},index=[0])
            kong = pd.concat([kong,new_data],ignore_index=True)
            
        
    
    return kong


def construct_cell_type_adj(adata,Sum_LR_CE_tensor):
    #构造cell type组合的pandas
    cell_type=np.unique(adata.obs["cell_type"].values)
    num=-1
    index1=[]
    cell_type_name=[]
    for i in range(len(cell_type)):#所有的细胞类型的组合构建
        for j in range(len(cell_type)):
            num+=1
            index1.append(num)
            cell_type_name.append(str(cell_type[i])+":"+str(cell_type[j]))
    
    
    kong=pd.DataFrame(columns=['cell_type_pair','strength'],index=index1) #创建一个空的pandas
    for i in range(len(cell_type_name)):
        kong["cell_type_pair"][i]=cell_type_name[i]
    #计算细胞种类一共有多少个组合，作为multi-view个数
    for i in range(kong.shape[0]):
        print(i)
        celltype1,celltype2=kong.loc[i,"cell_type_pair"].split(":")
        strength=0
        for n in range(Sum_LR_CE_tensor.shape[0]):#循环行
                if adata.obs["cell_type"][n] == celltype1:
                    
                    for m in range(Sum_LR_CE_tensor.shape[1]):#循环列
                        if adata.obs["cell_type"][m] == celltype2:
                            strength+=float(Sum_LR_CE_tensor[n][m])
        kong.loc[i,"strength"]=strength

    
    return kong

if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="it's usage tip.", description="SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data.")
    parser.add_argument("--filename", required=True, help="The file path for single-cell RNA-seq data, requires h5ad file format")
    parser.add_argument("--LR_result", required=True, help="The saved file path of significant ligand and receptor pairs")
    parser.add_argument("--save_dir", required=True, help="The folder path for saving the results (the directory will automatically be created).")
    args = parser.parse_args()
    filename = args.filename
    LR_df = args.LR_result
    save_dir = args.save_dir
    adata = sc.read(filename)

    LR_df = pd.read_csv(LR_df,delimiter=",",index_col=0).reset_index(drop=True)

    w_best = default_w_visium(adata)

    CE_tensor = compute_ce_tensor(adata, lr_df=LR_df, w_best=w_best)

    CE_tensor_filtered = filter_ce_tensor(CE_tensor, adata,lr_df=LR_df, w_best=w_best,n_pairs=200)
    save_file=save_dir+"CE_tensor_filtered.pt"
    torch.save(CE_tensor_filtered, save_file)


    #计算每对lr在不同的cell type对下的通讯强度
    lr_pair_df=construct_lr_adj(adata,CE_tensor_filtered,LR_df)
    #对strength进行缩放
    max_value = lr_pair_df["strength"].max()
    lr_pair_df["strength"]=lr_pair_df["strength"]/max_value
    save_file=save_dir+"lr_strength_in_celltype_pair_df.csv"
    lr_pair_df.to_csv(save_file)

    #同时提供cell-type的CCC
    # CE_tensor_filtered=torch.load("/home/jby2/scGPTPPI/ce_tensor_filter/CE_tensor_filtered.pt")
    Sum_CE_tensor_filtered=torch.sum(CE_tensor_filtered, dim=0)
    cell_type_pair_df=construct_cell_type_adj(adata,Sum_CE_tensor_filtered)

    #对strength进行缩放
    max_value = cell_type_pair_df["strength"].max()
    cell_type_pair_df["strength"]=cell_type_pair_df["strength"]/max_value
    save_file=save_dir+"Sum_CE_tensor_filtered_type_pair_df.csv"
    cell_type_pair_df.to_csv(save_file)




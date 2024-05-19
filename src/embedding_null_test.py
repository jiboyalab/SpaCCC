import pandas as pd
import numpy as np
import scanpy as sc
from anndata._core.anndata import AnnData
import scipy
from scipy.spatial.distance import cdist
import argparse

def null_test(df_nn: pd.DataFrame, 
            candidates, 
            filter_zeros=True, 
            pval=0.05):
    '''nonparametric left tail test to have enriched pairs'''
    if ('dist' ) not in df_nn.columns:
        raise IndexError('require resulted dataframe with column \'dist\' ')

    else:
        dist_test = df_nn[df_nn.index.isin(candidates)].copy()
        # filter pairs with correspondence_score zero
        if filter_zeros:
            mask = df_nn['dist'] != 0
        else:
            mask = np.ones(len(df_nn), dtype=bool)
        
        
        dist_null = df_nn[(~df_nn.index.isin(candidates)) & (mask)]
        dist_test['p_val'] = dist_test['dist'].apply(
            lambda x: scipy.stats.percentileofscore(dist_null['dist'], x) / 100)
        df_enriched = dist_test[dist_test['p_val'] < pval].sort_values(by=['dist'])
        print(f'\nTotal enriched: {len(df_enriched)} / {len(df_nn)}')
        df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1

    return df_enriched

def null_test_my(all_gene_embedding_df: pd.DataFrame,
    embedding_dist_df: pd.DataFrame, 
            pval=0.05,faked_pairs=200):
    '''nonparametric left tail test to have enriched pairs'''
    if ('dist' ) not in embedding_dist_df.columns:
        raise IndexError('require resulted dataframe with column \'dist\' ')

    else:
        # dist_test = embedding_dist_df[embedding_dist_df.index.isin(candidates)].copy()
        dist_test = embedding_dist_df
        for index, row in dist_test.iterrows():
            ligand=dist_test.loc[index,"ligand"]
            receptor=dist_test.loc[index,"receptor"]
            dist=dist_test.loc[index,"dist"]
            ligand_embedding=all_gene_embedding_df.loc[ligand].tolist()
            filtered_df = all_gene_embedding_df[all_gene_embedding_df.index != receptor]
            filtered_df = filtered_df.sample(n=faked_pairs, replace=False)
            pair=[]
            ligand_embedding=np.array(ligand_embedding,dtype=float).reshape(1,-1)

            for index2, row2 in filtered_df.iterrows():
                pair.append(cdist(np.array(ligand_embedding,dtype=float).reshape(1,-1), np.array(filtered_df.loc[index2,:].tolist(),dtype=float).reshape(1,-1), metric='euclidean')[0][0])
            dist_test.loc[index,"p_val"]=scipy.stats.percentileofscore(pair, dist) / 100
                
        df_enriched = dist_test[dist_test['p_val'] < pval].sort_values(by=['dist'])
        print(f'\nTotal enriched: {len(df_enriched)} / {len(embedding_dist_df)}')
        df_enriched['enriched_rank'] = np.arange(len(df_enriched)) + 1

    return df_enriched
if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="it's usage tip.", description="SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data.")
    parser.add_argument("--embedding_file", required=True, help="The file path of ligand and receptor embeddings ")
    parser.add_argument("--LR_file", required=True, help="The file path of ligand and receptor pairs ")
    parser.add_argument("--LR_result", required=True, help="The saved file path of significant ligand and receptor pairs ")
    args = parser.parse_args()
    embedding_file=args.embedding_file
    LR_file=args.LR_file  
    LR_result = args.LR_result
    all_embedding=pd.read_csv(embedding_file,delimiter=",",header=None,index_col=0)


    LR_pair_database_path = LR_file
    LR_df = pd.read_csv(LR_pair_database_path,delimiter=",")


    df = pd.DataFrame(columns=['ligand', 'receptor', 'dist'])
    #計算篩選之後的LR的欧式距离
    for i in range(LR_df.shape[0]):
        gene_name1=LR_df.loc[i,"ligand"]
        gene_name2=LR_df.loc[i,"receptor"]
        if gene_name1 in all_embedding.index and gene_name2 in all_embedding.index:
            gene_embedding1=list(all_embedding.loc[gene_name1])
            gene_embedding1=np.array(gene_embedding1,dtype=float).reshape(1,-1)
            gene_embedding2=list(all_embedding.loc[gene_name2])
            gene_embedding2=np.array(gene_embedding2,dtype=float).reshape(1,-1)
            dist =cdist(gene_embedding1, gene_embedding2, metric='euclidean')[0][0]
            df.loc[gene_name1 + '_' + gene_name2]=[gene_name1,gene_name2,dist]

    
    df_enriched=null_test_my(all_embedding,df,pval=0.05,faked_pairs=200)

    df_enriched.to_csv(LR_result)

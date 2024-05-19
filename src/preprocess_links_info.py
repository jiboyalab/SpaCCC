import pandas as pd
import csv
import argparse,time,os
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return
def ReadMyTsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName),delimiter="\t")
    for row in csv_reader:  
        SaveList.append(row)
    return
def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return
if __name__ == "__main__":

    parser = argparse.ArgumentParser(usage="it's usage tip.", description="SpaCCC: Large language model-based cell-cell communication inference for spatially resolved transcriptomic data.")
    parser.add_argument("--protein_links", required=True, help="The file path for protein links")
    parser.add_argument("--protein_info", required=True, help="The file path for protein information")
    parser.add_argument("--save_dir", required=True, help="The folder path for saving the results (the directory will automatically be created)")
    args = parser.parse_args()
    protein_links=args.protein_links
    protein_info=args.protein_info
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    protein_links = pd.read_csv(protein_links, sep=' ',header=0)
    protein_info = pd.read_csv(protein_info, sep='\t',header=0)

    final_pair=[]
    for i in range(protein_info.shape[0]):
        pair=[]
        gene_name=protein_info.loc[i,"preferred_name"]
        pair.append(i+1)
        pair.append(gene_name)
        final_pair.append(pair)
    filename=save_dir+"gene_name.csv"
    StorFile(final_pair,filename)


    final_pair=[]
    for i in range(protein_links.shape[0]):
        print(i)
        pair=[]
        protein1=protein_links.loc[i,"protein1"]
        protein2=protein_links.loc[i,"protein2"]
        combined_score=protein_links.loc[i,"combined_score"]
        gene1=protein_info[protein_info["#string_protein_id"]==protein1]["preferred_name"].values[0]
        gene2=protein_info[protein_info["#string_protein_id"]==protein2]["preferred_name"].values[0]
        pair.append(gene1)
        pair.append(gene2)
        pair.append(combined_score)
        final_pair.append(pair)
    filename=save_dir+"gene_links.csv"
    StorFile(final_pair,filename)

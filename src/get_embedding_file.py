import pandas as pd
import csv
import argparse
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
   
    parser.add_argument("--save_dir", required=True, help="The folder path for saving the results ")
    args = parser.parse_args()
    save_dir=args.save_dir
    gene_name_file=save_dir+"gene_name.csv"
    gene_name=pd.read_csv(gene_name_file,header=None,decimal=",",index_col=0)

    embedding_file=save_dir+"PPI_n2vplus_epoch100.emb"
    with open(embedding_file) as f1:
        lines=f1.readlines()


    final=[]
    for i in range(1,len(lines)):#第一行是节点的个数以及embedding的维度，跳过
        pair=[]
        line=lines[i].strip().split(" ")
        pair.append(list(gene_name.loc[int(line[0])])[0])
        pair.extend(lines[i].strip().split(" ")[1:])
        final.append(pair)
    PPI_embedding_file=save_dir+"PPI_embedding.csv"
    StorFile(final,PPI_embedding_file)

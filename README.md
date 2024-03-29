# AGLDA

## Introduction

This is code of AGLDA (“Learning association characteristics by dynamic hypergraph and gated convolution enhanced pairwise attributes for prediction of disease-related lncRNAs”).

## Dataset

| File_name                  | Data_type       | Source                                                       |
| -------------------------- | --------------- | ------------------------------------------------------------ |
| dis_sim_matrix_process.txt | disease-disease | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html)           |
| lnc_dis_association.txt    | lncRNA-disease  | [LncRNADisease](https://www.cuilab.cn/lncrnadisease)         |
| mi_dis.txt                 | miRNA-disease   | [HMDD](https://www.cuilab.cn/hmdd)                           |
| lnc_mi.txt                 | lncRNA-miRNA    | [starBase](https://rnasysu.com/encori/)                      |
| lnc_sim.txt                | lncRNA-lncRNA   | [Chen *et al.*](https://www.nature.com/articles/srep11338)$^{1}$ |

(1) Chen, X., Clarence Yan, C., Luo, C. *et al.* Constructing lncRNA Functional Similarity Network Based on lncRNA-Disease Associations and Disease Semantic Similarity. *Sci Rep* **5**, 11338 (2015).

# File

```markdown
-train : data preprocessing and model training
-data : data set
-evaluation : experimental evaluation
-parasave : results and parameters
```

## Environment

```markdown
packages:
python == 3.9.0
torch == 1.13.0
numpy == 1.23.5
scikit-learn == 1.2.2
scipy == 1.10.1
pandas == 2.0.1
matplotlib == 3.7.1
```

# Run

```python
python ./train/preprocess.py
python ./train/train.py
python ./evaluation/plt.py
```


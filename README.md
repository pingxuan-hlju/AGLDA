# AGLDA

## Introduction

This is code of AGLDA (“Learning association characteristics by dynamic hypergraph and gated convolution enhanced pairwise attributes for prediction of disease-related lncRNAs”).

## Dataset

| File_name                  | Data_type       | Data_size |
| -------------------------- | --------------- | --------- |
| dis_sim_matrix_process.txt | disease-disease | 405,405   |
| lnc_dis_association.txt    | lncRNA-disease  | 240,405   |
| mi_dis.txt                 | miRNA-disease   | 495,405   |
| yuguoxian_lnc_mi.txt       | lncRNA-miRNA    | 240,495   |
| lnc_sim.txt                | lncRNA-lncRNA   | 240,240   |
| yuguoxian_lncRNA_name.txt  | lncRNA          | 240       |
| disease_name.txt           | disease         | 405       |

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


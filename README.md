# OTHarmonizer
Automated construction of hierarchical cell annotation relationships across single-cell transcriptomics datasets using optimal transport

# Quick Start Guide

## 1. Preprocess Data
To begin using OTHarmonizer, first load your dataset using **Scanpy** and perform preprocessing steps such as normalization, log transformation, and identification of highly variable genes.
```python
import scanpy as sc
import OTHarmonizer as oth

adata = sc.read_h5ad('path/to/.h5ad')

# Normalize the data and log-transform，and identify highly variable genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key='batch_key', subset=True)
```

## 2. Reduce Batch Effect
Use scVI from OTHarmonizer to reduce batch effects by specifying the batch key and annotation key. This step helps in aligning the data across batches while preserving biological variance.
```python
latent = oth.scVI(adata, 
                  batch_key='batch_key', 
                  annotation_key='annotation_key', 
                  n_latent=10)
```

## 3. Perform Annotation Harmonization
After batch effect correction, you can perform annotation harmonization using OTHarmonizer. You can specify a sample size, and optionally set the batch order (if needed).
```python
root = oth.do_harmonization(adata, 
                            annotation_key='annotation_key', 
                            batch_key='batch_key', 
                            sample_size=200,
#                             batch_order = ['study1', 'study2', 'study3']
                            )
```

## 4. Create Ground-Truth Tree
Define the ground-truth hierarchical tree by providing a string representation of parent-child relationships among annotations. This tree serves as a reference to compare against the harmonized tree.
```python
ref_tree = oth.create_tree_from_string("""
'root'
----'annotation-A'
--------'annotation-C'
--------'annotation-B'
----'annotation-D&annotation-E'
...
""")
```

## 5. Compare Tree
Finally, use the benchmark function to compare the harmonized tree with the ground-truth reference tree, using the provided performance metrics (TEDS, PCBS, and AH-F1).
```python
oth.benchmark(root, ref_tree)
```

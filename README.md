### Diagnostic Text-guided Representation Learning in Hierarchical Classification for Pathological Whole Slide Image

The paper is under review. See the preprint version as follows: [Arxiv](https://arxiv.org/abs/2411.10709)

The current codebase contains the following four parts:

1. WSI preprocessing
2. Tree-like text preprocessing
3. PathTree model architecture
4. PathTree training


### 1. WSI preprocessing

We use the Opensdpc and PIANO codebases to preprocess the WSI data.
See the [Opensdpc](https://github.com/WonderLandxD/opensdpc) and [PIANO](https://github.com/WonderLandxD/PIANO) for more details.


We use the BRACS dataset structure as an example. The `Data/BRACS` folder shows the path structure after preprocessing.

```
BRACS/
├── BRACS_281/
│   ├── plip/
│   │   └── piano_BRACS_281_plip.pth
├── BRACS_300/
│   ├── plip/
│   │   └── piano_BRACS_300_plip.pth
├── BRACS_735/
│   ├── plip/
│   │   └── piano_BRACS_735_plip.pth
└── ...
```

Using PIANO library, each BRACS sample folder can be contains feature embeddings extracted by different foundation models, you can choose one of them as the patch feature encoder for PathTree.

### 2. Tree-like text preprocessing

Currently, we use the BRACS dataset as an example to explain the results of our text preprocessing

- We use the `tree_text_preprocess/bracs.json` file to show the tree-like text preprocessing.

- We use the `tree_text_preprocess/bracs_tree_edge_index.pth` file to save the edge index of the tree-like graph structure, which is used to build the text prompt graph for PathTree.

### 3. PathTree model architecture

The code of PathTree model is shown in `model/pathtree.py`.

### 4. PathTree training

Run the `train.py` file to train the PathTree model. The bash command is as follows:

```bash
python train.py --dataset_dir wsi_preprocess/bracs --fold 1 --device cuda:0 --batch_size 1 --seed 3407 --output_dir output_results --epoch 100 --tree_json_path tree_text_preprocess/bracs.json --edge_index_path tree_text_preprocess/bracs_edge_index.pth --patch_encoder plip --text_encoder plip --attn_type attn
```












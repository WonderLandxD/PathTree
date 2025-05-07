import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

import piano


class PathTreeDataset(Dataset):
    def __init__(self, data_json_path, tree_json_path, patch_encoder_name='plip', text_encoder_name='plip', is_eval=False):
        """
        Args:
            data_json_path (str): Path to the tcga_pathtree_datasets.json file
            tree_json_path (str): Path to the tcga.json file containing the tree structure
        """
        super().__init__()
        
        # Load the dataset json
        with open(data_json_path, 'r') as f:
            self.data = json.load(f)
            
        # Load the tree structure json
        with open(tree_json_path, 'r') as f:
            self.tree = json.load(f)
            
        # Create mappings from tree
        self.id_to_value = {}
        self.name_to_id = {}  # Add mapping from name to id
        self.leaf_nodes = []  # Store leaf node information
        self._build_mappings(self.tree)
        
        # Create mapping for leaf nodes to consecutive indices
        self.leaf_to_label = {node_id: idx for idx, node_id in enumerate(sorted([node['id'] for node in self.leaf_nodes]))}
        self.num_classes = len(self.leaf_to_label)
        
        # Sort values by ID to ensure consistent ordering
        self.sorted_values = [self.id_to_value[i] for i in range(len(self.id_to_value))]
        
        # Initialize PIANO text preprocessor
        self.text_preprocess = piano.create_model(text_encoder_name).text_preprocess
        
        self.patch_encoder_name = patch_encoder_name

        self.is_eval = is_eval


    def _build_mappings(self, node):
        """Recursively build mappings from ID to value and name to ID in tree"""
        self.id_to_value[node['id']] = node['value']
        self.name_to_id[node['name']] = node['id']
        
        # If node has no children, it's a leaf node
        if 'children' not in node:
            self.leaf_nodes.append({
                'id': node['id'],
                'name': node['name'],
                'value': node['value']
            })
        else:
            for child in node['children']:
                self._build_mappings(child)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Get patch tensors (absolute path)
        feat_dir = item['file_path']
        slide_id = os.path.basename(feat_dir)
        patch_feat = torch.load(os.path.join(feat_dir, f'{self.patch_encoder_name}/piano_{slide_id}_{self.patch_encoder_name}.pth'), map_location='cpu')['feats']

        fine_label = item['fine_label']
        node_id = self.name_to_id[fine_label]
        
        text_tensors = self.text_preprocess(self.sorted_values)
        sc_label = self.leaf_to_label[node_id]
        
        return {
            'patch_feat': patch_feat,
            'text_tensors': text_tensors,
            'sc_label': sc_label,  # Using the new consecutive label (0 to num_classes-1)
            'slide_id': slide_id,
            'raw_texts': self.sorted_values
        }
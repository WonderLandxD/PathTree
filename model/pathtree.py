import torch
import torch.nn as nn
import numpy as np
import random
from .module import Attn_Net_Gated, Self_Attention_light, BiTreeGNN, TripletLoss
import json

import piano

class PathTree(nn.Module):
    def __init__(self, 
                 json_path=None, 
                 edge_index=None,
                 text_model_name='plip',
                 text_dim=512,
                 patch_dim=512,
                 attn_block='attn', 
                 match_type='mean', 
                 node_num=None,
                 num_class=None):
        super(PathTree, self).__init__()

        with open(json_path, 'r') as file:
            self.tree_data = json.load(file)

        self.num_class = num_class
        
        with torch.no_grad():
            self.up2down_edge_index, self.down2up_edge_index = edge_index['up2down_edge_index'], edge_index['down2up_edge_index']
            self.text_encoder = piano.create_model(text_model_name)

        self.text_fc = nn.Sequential(nn.Linear(text_dim, 512), nn.LeakyReLU())
        self.patch_fc = nn.Sequential(nn.Linear(patch_dim, 512), nn.LeakyReLU())
        
        if node_num is None:
            node_num = 2*num_class - 1
        if attn_block == 'attn':
            self.patch_tree = Attn_Net_Gated(L=512, D=256, n_classes=node_num)
        elif attn_block == 'selfattn':
            self.patch_tree = Self_Attention_light(norm_layer=nn.LayerNorm, dim=512, heads=node_num) 
        else:
            raise ValueError("Invalid attention block type. Must be either 'attn' or 'selfattn'")


        self.structure_encoder = BiTreeGNN(num_node_features=512)

        self.mse_loss = nn.MSELoss()
    
        self.random_match_loss = TripletLoss(margin=0.2)
        self.sibling_match_loss = TripletLoss(margin=0.2 * 0.5)        
        self.parent_match_loss = TripletLoss(margin=0.2 * 0.01)
    
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.random_match_type = match_type
    
    def load_features_for_node(self, tree_slide_feat, node_id):
        return tree_slide_feat[node_id].unsqueeze(0)

    def aggregate_features(self, node_id, cont_score, node_features):
        self_id = node_id
        sibling_id = self.find_sibling_nodes(self.tree_data, node_id)[0]

        C_score = torch.cat([cont_score[[self_id]], cont_score[[sibling_id]]], dim=-1).unsqueeze(0)  # [1, 2]
        C_prob = torch.softmax(C_score, dim=-1) # [1, 2]
        # choice_feat = tree_slide_feat[[node_id, sibling_id], :] # [2, 512]
        choice_feat = torch.cat([node_features[self_id], node_features[sibling_id]], dim=0)
        agg_feat = torch.matmul(C_prob, choice_feat) # [1, 512]

        return agg_feat
    
    def update_node_features(self, node, tree_slide_feat, cont_score, node_features):
        if 'children' not in node or not node['children']:
            node_features[node['id']] = self.load_features_for_node(tree_slide_feat, node['id'])
        else:
            # child_features = []
            for child in node['children']:
                node_features = self.update_node_features(child, tree_slide_feat, cont_score, node_features)
                # child_features.append(node_features[child['id']])
 
            # aggregated_feature = self.aggregate_features(child['id'], cont_score, tree_slide_feat)
            aggregated_feature = self.aggregate_features(child['id'], cont_score, node_features)
            
            node_features[node['id']] = self.load_features_for_node(tree_slide_feat, node['id']) + aggregated_feature

        return node_features



    def find_path_by_leaf_id(self, node, target_id, path=[]):
        # Add the current node's id to the path
        path.append(node['id'])
        
        # Check if the current node is the target leaf node
        if node['id'] == target_id:
            return path
        
        # If the node has children, continue searching
        if 'children' in node:
            for child in node['children']:
                # Recursive call to search in the child
                result = self.find_path_by_leaf_id(child, target_id, path.copy())
                if result is not None:
                    return result

    def find_sibling_nodes(self, node, target_id, parent=None):
        # If the current node is the target, return the sibling nodes from the parent
        if node['id'] == target_id:
            if parent and 'children' in parent:
                # Filter out the target node itself to only return its siblings
                return [child['id'] for child in parent['children'] if child['id'] != target_id]
        
        # If the node has children, continue searching
        if 'children' in node:
            for child in node['children']:
                # Recursive call to search in the child, passing the current node as the parent
                result = self.find_sibling_nodes(child, target_id, node)
                if result is not None:
                    return result
                
    def find_parent_node(self, node, target_id, parent=None):
        # If the current node is the target, return the parent
        if node['id'] == target_id:
            return parent['id'] if parent else None
        
        # If the node has children, continue searching
        if 'children' in node:
            for child in node['children']:
                # Recursive call to search in the child, passing the current node as the parent
                result = self.find_parent_node(child, target_id, node)
                if result is not None:
                    return result
    
    def find_other_leaf_nodes(self, node, exclude_id, exclude_parent_id=None, exclude_siblings_ids=[]):
        # Initialize a list to collect leaf nodes
        leaf_nodes = []
        
        # If the current node is a leaf and not in the exclude list, add it
        if 'children' not in node or not node['children']:
            if node['id'] not in exclude_siblings_ids and node['id'] != exclude_id and node['id'] != exclude_parent_id:
                return [node['id']]
        
        # If the node has children, continue searching
        if 'children' in node:
            for child in node['children']:
                # Recursive call to search in the child
                leaf_nodes += self.find_other_leaf_nodes(child, exclude_id, exclude_parent_id, exclude_siblings_ids)
        
        return leaf_nodes


    def forward(self, _patch_feat, _text_ids, sc_label=None, is_eval=True, attention_only=False):  # text_feat, edge_idx
        patch_feat = self.patch_fc(_patch_feat)    # [batch_size, patch_num, patch_dim]  e.g. [1, 732, 512]
        if attention_only:
            A = self.patch_tree(patch_feat, attention_only=True)
            return A
        tree_slide_feat = self.patch_tree(patch_feat).squeeze(0)  # [2 * node_num - 1, patch_dim]  e.g. [13, 512]

        node_feat = self.text_encoder.encode_text(_text_ids)
        node_feat = self.text_fc(node_feat)   # [2 * node_num - 1, text_dim]  e.g. [13, 512]
        tree_text_feat = self.structure_encoder(node_feat.to(tree_slide_feat.device), self.up2down_edge_index.to(tree_slide_feat.device), self.down2up_edge_index.to(tree_slide_feat.device))  # [2 * node_num - 1, text_dim]

        cont_score = torch.sum(tree_slide_feat * tree_text_feat, dim=1)  # [13]

        node_features = {}
        node_features = self.update_node_features(self.tree_data, tree_slide_feat, cont_score, node_features)
        slide_feat = node_features[self.tree_data['id']]
        
        slide_feat_norm = slide_feat / slide_feat.norm(dim=-1, keepdim=True)
        leaf_text_feat_norm = tree_text_feat[:self.num_class, :] / tree_text_feat[:self.num_class, :].norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        
        if is_eval == False:
            target_text_feat = tree_text_feat[sc_label.item()].unsqueeze(0)  # [1, 512]

            tree_path = self.find_path_by_leaf_id(self.tree_data, sc_label.item(), path=[])   # path's list
            selected_text_feat = tree_text_feat[tree_path]
            B, _ = selected_text_feat.shape
            joint_loss = self.mse_loss(slide_feat.expand(B, -1), selected_text_feat)   #  Joint Embedding Learning

            sibling_node = self.find_sibling_nodes(self.tree_data, sc_label.item(), parent=None)  # sibling node
            sibling_text_feat = tree_text_feat[sibling_node]

            parent_node = self.find_parent_node(self.tree_data, sc_label.item(), parent=None)  # parent node
            parent_text_feat = tree_text_feat[[parent_node]]

            other_leaf_nodes = self.find_other_leaf_nodes(self.tree_data, sc_label.item(), exclude_siblings_ids=sibling_node, exclude_parent_id=parent_node)
            if self.random_match_type == 'random':
                random_leaf_node = random.choice(other_leaf_nodes)
                random_text_feat = tree_text_feat[[random_leaf_node]]
            elif self.random_match_type == 'mean':
                random_text_feat = tree_text_feat[other_leaf_nodes]
                random_text_feat = random_text_feat.mean(dim=0, keepdim=True)

            match_loss = self.random_match_loss(slide_feat, target_text_feat, random_text_feat) + \
                         self.sibling_match_loss(slide_feat, target_text_feat, sibling_text_feat) + \
                         self.parent_match_loss(slide_feat, target_text_feat, parent_text_feat)

            logits = logit_scale * slide_feat_norm @ leaf_text_feat_norm.t()

            return logits, joint_loss, match_loss

        else:
            with torch.no_grad():
                logits = logit_scale * slide_feat_norm @ leaf_text_feat_norm.t()

                return logits
            

if __name__ == '__main__':
    json_path = 'tree_text_preprocess/bracs.json'
    edge_index = torch.load('tree_text_preprocess/bracs_tree_edge_index.pth')
    model = PathTree(json_path=json_path, 
                     edge_index=edge_index, 
                     text_model_name='plip',
                     text_dim=512,
                     patch_dim=512,
                     attn_block='attn', 
                     match_type='mean', 
                     num_class=7).to("cpu")

    patch_feat = torch.randn([1, 732, 512]).to("cpu")
    sc_label = torch.tensor([1]).to("cpu")
    text_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
    text_encoder = piano.create_model('plip')
    text_processor = text_encoder.text_preprocess
    text_ids = text_processor(text_list).to("cpu")
    model.train()
    output = model(patch_feat, text_ids, sc_label, is_eval=False)
    print(output)

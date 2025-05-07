import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import csv
from tqdm import tqdm
import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
from typing import List, Dict, Any
from collections import Counter

from datasets_config import PathTreeDataset
from model.pathtree import PathTree

class AutoHierarchicalMetrics:
    def __init__(self, tree_json_path: str):
        """
        Initialize automatic hierarchical metrics calculator
        
        Args:
            tree_json_path: Path to the tree structure JSON file
        """
        with open(tree_json_path, 'r') as f:
            self.tree_data = json.load(f)
            
        # Store all paths
        self.all_paths = {}
        # Store mapping from leaf nodes to labels
        self.leaf_to_label = {}
        # Store mapping from labels to paths
        self.label_to_path = {}
        
        # Initialize mappings
        self._build_mappings()
        
    def _build_mappings(self):
        """Build all necessary mapping relationships"""
        def traverse(node: Dict[str, Any], current_path: List[int]):
            node_id = node['id']
            current_path = current_path + [node_id]
            
            # If it's a leaf node
            if 'children' not in node:
                leaf_idx = len(self.leaf_to_label)
                self.leaf_to_label[node_id] = leaf_idx
                self.label_to_path[leaf_idx] = current_path
                self.all_paths[node_id] = current_path
            else:
                self.all_paths[node_id] = current_path
                for child in node['children']:
                    traverse(child, current_path)
                    
        traverse(self.tree_data, [])
        
    def _count_matches(self, pred_path: List[int], true_path: List[int]) -> int:
        """Calculate the number of matching nodes between two paths"""
        return len(set(pred_path) & set(true_path))
    
    def _get_path_from_label(self, label: int) -> List[int]:
        """Get complete path from label"""
        return self.label_to_path[label]
    
    def _get_pred_path(self, logits: torch.Tensor) -> List[int]:
        """Determine prediction path based on model output logits"""
        # Get predicted leaf node label
        pred_label = torch.argmax(logits).item()
        # Get corresponding complete path
        return self.label_to_path[pred_label]
    
    def calculate_metrics(self, logits_batch: torch.Tensor, labels_batch: torch.Tensor):
        total_matches = 0
        total_pred_nodes = 0
        total_true_nodes = 0
        
        for logits, label in zip(logits_batch, labels_batch):
            pred_path = self._get_pred_path(logits)
            true_path = self._get_path_from_label(label.item())
            
            matches = self._count_matches(pred_path, true_path)
            total_matches += matches
            total_pred_nodes += len(pred_path)
            total_true_nodes += len(true_path)
        
        # Calculate metrics
        h_precision = total_matches / total_pred_nodes if total_pred_nodes > 0 else 0
        h_recall = total_matches / total_true_nodes if total_true_nodes > 0 else 0
        h_f1 = 2 * (h_precision * h_recall) / (h_precision + h_recall) if (h_precision + h_recall) > 0 else 0
        
        return {
            'hierarchical_precision': h_precision,
            'hierarchical_recall': h_recall,
            'hierarchical_f1': h_f1
        }

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to('cpu')
    train_loader = tqdm(data_loader, desc=f'Epoch {epoch}', ncols=100, colour='red')
    
    for i, batch in enumerate(train_loader):
        patch_feat = batch['patch_feat'].to(device)
        text_tensors = batch['text_tensors'].to(device)
        label = batch['sc_label'].to(device)
        
        optimizer.zero_grad()
        logits, joint_loss, match_loss = model(patch_feat, text_tensors, label, is_eval=False)
        criterion = nn.CrossEntropyLoss()
        prob_loss = criterion(logits, label)
        loss = 1 * prob_loss + 1 * joint_loss + 1 * match_loss

        loss.backward()
        optimizer.step()

        total_loss = (total_loss * i + loss.detach().cpu()) / (i + 1)
        total_prob_loss = (prob_loss * i + prob_loss.detach().cpu()) / (i + 1)
        total_joint_loss = (joint_loss * i + joint_loss.detach().cpu()) / (i + 1)
        total_match_loss = (match_loss * i + match_loss.detach().cpu()) / (i + 1)

        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(
            epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))
    
    return total_loss.item(), total_prob_loss.item(), total_joint_loss.item(), total_match_loss.item()

@torch.no_grad()
def val_one_epoch(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        labels = torch.tensor([], device='cpu')
        preds = torch.tensor([], device='cpu')
        
        val_loader = tqdm(data_loader, file=sys.stdout, ncols=75, colour='blue')

        for batch in val_loader:
            patch_feat = batch['patch_feat'].to(device)
            text_tensors = batch['text_tensors'].to(device)
            label = batch['sc_label'].to(device)

            output = model(patch_feat, text_tensors, is_eval=True)
            labels = torch.cat([labels, label.detach().cpu()], dim=0)
            preds = torch.cat([preds, output.detach().cpu()], dim=0)

        return preds.cpu(), labels.cpu()

def planar_metrics(logits, labels, num_classes):
    # accuracy
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    balanced_acc = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())

    # macro-average AUC scores
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())

    # weighted f1-score
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')

    # quadratic weighted Kappa
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')

    # confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    return accuracy, balanced_acc, auc, f1, kappa, confusion_mat

def parse():
    parser = argparse.ArgumentParser(description='Parsers for PathTree')
    parser.add_argument('--dataset_dir', type=str, default='wsi_preprocess/bracs')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--output_dir', type=str, default='output_results')
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--tree_json_path', type=str, default='tree_text_preprocess/bracs.json', help='Path to tree json file')
    parser.add_argument('--edge_index_path', type=str, default='tree_text_preprocess/bracs_edge_index.pth', help='Path to edge index file')
    parser.add_argument('--patch_encoder', type=str, default='plip')
    parser.add_argument('--text_encoder', type=str, default='plip')
    parser.add_argument('--attn_type', type=str, default='attn', choices=['attn', 'selfattn'])

    return parser.parse_args()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize hierarchical metrics calculator
    hier_metrics = AutoHierarchicalMetrics(args.tree_json_path)

    # Initialize datasets
    raw_fold_dir = args.dataset_dir
    train_dataset = PathTreeDataset(
        data_json_path=os.path.join(raw_fold_dir, f'fold_{args.fold}', 'train.json'),
        tree_json_path=args.tree_json_path,
        patch_encoder_name=args.patch_encoder,
        text_encoder_name=args.text_encoder,
        is_eval=False
    )
    valid_dataset = PathTreeDataset(
        data_json_path=os.path.join(raw_fold_dir, f'fold_{args.fold}', 'valid.json'),
        tree_json_path=args.tree_json_path,
        patch_encoder_name=args.patch_encoder,
        text_encoder_name=args.text_encoder,
        is_eval=False
    )
    
    # Get number of classes from dataset
    num_classes = train_dataset.num_classes
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    valid_loader = data.DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    sample = next(iter(train_loader))
    patch_dim = sample['patch_feat'].shape[-1]  # Get patch feature dimension

    
    edge_index = torch.load(args.edge_index_path, weights_only=False)

    model = PathTree(
        json_path=args.tree_json_path, 
        edge_index=edge_index,
        text_model_name=args.text_encoder,
        text_dim=512,
        patch_dim=patch_dim,
        attn_block=args.attn_type, 
        match_type='mean', 
        node_num=len(train_dataset.sorted_values),
        num_class=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-4
    )

    # Create output directories
    output_dir = Path(os.path.join(
        args.output_dir,
        f'{args.fold}',
        args.attn_type,
        f'patch_{args.patch_encoder}',
        f'text_{args.text_encoder}'
    ))
    weight_dir = output_dir / "weight"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    # Initialize logging files
    with open(f'{output_dir}/test_matrix.txt', 'w') as f:
        print('test start', file=f)
    
    with open(f'{output_dir}/test_results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "test_acc", "test_bal_acc", "test_auc", "test_f1", 
                           "test_hi_precision", "test_hi_recall", "test_hi_f1"])

    # Initialize tracking variables
    max_test_accuracy = 0.0
    max_test_bal_acc = 0.0
    max_test_auc = 0.0
    max_test_f1 = 0.0
    max_test_hi_pre = 0.0
    max_test_hi_recall = 0.0
    max_test_hi_f1 = 0.0


    # Training loop
    for epoch in range(args.epoch):
        _ = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch+1
        )

        torch.cuda.empty_cache()

        test_planar_probs, test_planar_labels = val_one_epoch(
            model=model,
            data_loader=valid_loader,
            device=device
        )

        # Calculate planar metrics
        test_acc, test_bal_acc, test_auc, test_f1, _, test_mat = planar_metrics(
            test_planar_probs,
            test_planar_labels,
            num_classes=num_classes
        )

        # Calculate hierarchical metrics
        hier_results = hier_metrics.calculate_metrics(test_planar_probs, test_planar_labels)
        test_hi_pre = hier_results['hierarchical_precision']
        test_hi_recall = hier_results['hierarchical_recall']
        test_hi_f1 = hier_results['hierarchical_f1']

        print('Test\t[epoch {}] acc:{}\tbal_acc:{}\tauc:{}\tf1-score:{}'.format(
            epoch + 1, test_acc, test_bal_acc, test_auc, test_f1))
        print('Test\t[epoch {}] h_pre:{}\th_recall:{}\th_f1:{}'.format(
            epoch + 1, test_hi_pre, test_hi_recall, test_hi_f1))
        print('test matrix ......')
        print(test_mat)
        
        # Update best metrics
        max_test_accuracy = max(max_test_accuracy, test_acc)
        max_test_bal_acc = max(max_test_bal_acc, test_bal_acc)
        max_test_auc = max(max_test_auc, test_auc)
        max_test_f1 = max(max_test_f1, test_f1)
        max_test_hi_pre = max(max_test_hi_pre, test_hi_pre)
        max_test_hi_recall = max(max_test_hi_recall, test_hi_recall)
        max_test_hi_f1 = max(max_test_hi_f1, test_hi_f1)

        if max_test_accuracy == test_acc:
            print('best test acc found... save best acc weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_acc.pth")

        if max_test_hi_f1 == test_hi_f1:
            print('best test hierarchical f1 found... save best hier_f1 weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_hier_f1.pth")

        # Log results
        print('max test accuracy: {:.4f}%'.format(max_test_accuracy*100))
        print('max test balanced accuracy: {:.4f}%'.format(max_test_bal_acc*100))
        print('max test auc: {:.4f}%'.format(max_test_auc*100))
        print('max test f1: {:.4f}%'.format(max_test_f1*100))
        print('max test hierarchical precision: {:.4f}%'.format(max_test_hi_pre*100))
        print('max test hierarchical recall: {:.4f}%'.format(max_test_hi_recall*100))
        print('max test hierarchical f1: {:.4f}%'.format(max_test_hi_f1*100))

        with open(f'{output_dir}/test_matrix.txt', 'a') as f:
            print(epoch + 1, file=f)
            print(test_mat, file=f)

        with open(f'{output_dir}/test_results.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, test_acc, test_bal_acc, test_auc, test_f1,
                               test_hi_pre, test_hi_recall, test_hi_f1])

if __name__ == '__main__':
    args = parse()
    main(args)

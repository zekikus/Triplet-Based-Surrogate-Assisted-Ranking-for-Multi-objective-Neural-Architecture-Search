import os
import re
import io
import torch
import pickle
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.bcnb_dataset import BCNB_Dataset
from sklearn.metrics import confusion_matrix, accuracy_score

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

# Updated function to extract WSI ID from patch path
def extract_wsi_id_from_path(patch_path):
    """
    Extract WSI ID from patch path.
    Example: C:/Users/FSM/Desktop/MUSA/BCNB/for_nas/paper_patches/patches\217\217_3_0_768.jpg
    Returns: '217'
    """
    # Normalize path separators
    normalized_path = patch_path.replace('\\', '/')
    
    # Split path and find the WSI folder
    path_parts = normalized_path.split('/')
    
    # Look for the WSI ID in the path - it should be the folder name containing the patch
    # The WSI ID appears twice: once as folder name and once as part of filename
    for i, part in enumerate(path_parts):
        if part.isdigit() and i < len(path_parts) - 1:
            # Check if the next part (filename) starts with the same number
            filename = path_parts[i + 1]
            if filename.startswith(part + '_'):
                return part
    
    # Alternative approach: extract from filename directly
    filename = os.path.basename(patch_path)
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    
    # Fallback: look for any digit sequence in the path
    matches = re.findall(r'\b(\d+)\b', normalized_path)
    if matches:
        return matches[-2] if len(matches) > 1 else matches[-1]  # Take second-to-last or last
    
    return None

# Updated majority voting function using dataloader's patch paths
def majority_vote_test_updated(model, test_loader):
    """
    Perform majority voting using patch paths returned by dataloader.
    Dataloader returns: (image, label, patch_path)
    """
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    
    # Dictionary to store predictions for each WSI
    wsi_predictions = defaultdict(list)
    wsi_true_labels = defaultdict(list)
    wsi_patch_paths = defaultdict(list)  # Store patch paths for debugging
    
    print("\n🗳️  Performing majority voting on test data...")
    print("=" * 80)
    
    # Test batches with TQDM
    test_pbar = tqdm(test_loader, desc="Processing Test Batches", unit="batch")
    
    with torch.no_grad():
        for batch_idx, (data, target, patch_paths) in enumerate(test_pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            # Process each sample in the batch
            for i in range(len(predicted)):
                patch_path = patch_paths[i]  # Get patch path from dataloader
                wsi_id = extract_wsi_id_from_path(patch_path)
                
                if wsi_id:
                    wsi_predictions[wsi_id].append(predicted[i].cpu().numpy())
                    wsi_true_labels[wsi_id].append(target[i].cpu().numpy())
                    wsi_patch_paths[wsi_id].append(patch_path)
                else:
                    print(f"⚠️  Warning: Could not extract WSI ID from path: {patch_path}")
            
            # Update progress bar
            test_pbar.set_postfix({
                'WSIs Found': len(wsi_predictions),
                'Total Patches': sum(len(v) for v in wsi_predictions.values())
            })
    
    print(f"\n📊 Found {len(wsi_predictions)} WSIs with {sum(len(v) for v in wsi_predictions.values())} total patches")
    
    # Perform majority voting for each WSI
    final_predictions = []
    final_true_labels = []
    wsi_results = []
    
    print(f"\n📋 Majority voting results:")
    print("-" * 80)
    
    # Progress bar for majority voting
    wsi_pbar = tqdm(sorted(wsi_predictions.items(), key=lambda x: int(x[0])), 
                    desc="Majority Voting", unit="WSI")
    
    for wsi_id, predictions in wsi_pbar:
        # Majority vote for predictions
        pred_votes = np.array(predictions)
        majority_pred = np.bincount(pred_votes).argmax()
        
        # True label (should be same for all patches of a WSI)
        true_labels_for_wsi = np.array(wsi_true_labels[wsi_id])
        true_label = true_labels_for_wsi[0]  # Take first label
        
        # Verify all patches have the same true label
        if not np.all(true_labels_for_wsi == true_label):
            print(f"⚠️  Warning: WSI {wsi_id} has inconsistent labels: {np.unique(true_labels_for_wsi)}")
        
        final_predictions.append(majority_pred)
        final_true_labels.append(true_label)
        
        # Count votes
        votes_0 = np.sum(pred_votes == 0)
        votes_1 = np.sum(pred_votes == 1)
        confidence = max(votes_0, votes_1) / len(pred_votes)
        
        # Status indicator
        status = "✅" if majority_pred == true_label else "❌"
        
        # Store results for summary
        wsi_results.append({
            'wsi_id': wsi_id,
            'num_patches': len(pred_votes),
            'votes_0': votes_0,
            'votes_1': votes_1,
            'majority_pred': majority_pred,
            'true_label': true_label,
            'confidence': confidence,
            'correct': majority_pred == true_label
        })
        
        wsi_pbar.set_postfix({
            'WSI': wsi_id,
            'Patches': len(pred_votes),
            'Confidence': f'{confidence:.2f}',
            'Status': status
        })
    
    # Print detailed results
    print(f"\n📈 Detailed WSI Results:")
    print("-" * 100)
    print(f"{'WSI ID':<8} {'Patches':<8} {'Votes [0|1]':<12} {'Pred':<5} {'True':<5} {'Conf':<6} {'Status':<6} {'Sample Path'}")
    print("-" * 100)
    
    for result in wsi_results:
        wsi_id = result['wsi_id']
        sample_path = os.path.basename(wsi_patch_paths[wsi_id][0])  # Show first patch filename
        status_symbol = "✅" if result['correct'] else "❌"
        
        print(f"{wsi_id:<8} {result['num_patches']:<8} "
              f"[{result['votes_0']:>2}|{result['votes_1']:>2}]{'':>4} "
              f"{result['majority_pred']:<5} {result['true_label']:<5} "
              f"{result['confidence']:.2f}{'':>2} {status_symbol:<6} {sample_path}")
    
    return np.array(final_predictions), np.array(final_true_labels), wsi_results

# Enhanced confusion matrix function (same as before)
def create_confusion_matrix_enhanced(y_true, y_pred, accuracy, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0 (Non-Cancer)', 'Class 1 (Cancer)'], 
                yticklabels=['Class 0 (Non-Cancer)', 'Class 1 (Cancer)'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'WSI-Level Confusion Matrix (Majority Voting)\nAccuracy: {accuracy:.2f}%', 
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add detailed statistics
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (fn + tn)

    stats_text = f"""
    Total WSIs: {len(y_true)}
    
    Precision: {precision:.3f}
    Recall: {recall:.3f}
    F1-Score: {f1_score:.3f}
    Sensitivity: {sensitivity:.3f}
    Specificity: {specificity:.3f}
    PPV: {ppv:.3f}
    NPV: {npv:.3f}
    
    True Negatives: {tn}
    False Positives: {fp}
    False Negatives: {fn}
    True Positives: {tp}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return cm, stats_text

if __name__ == "__main__":
    for model_no in [843, 916, 1745, 1967, 2063, 2150, 2249, 2287, 2950]:
        
        stats_text = ""
        for seed in [0, 1234, 3074]:
            path = 'results/BCNB_MODE_Triplet'
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

            model = None
            with open(f"{path}/model_{model_no}.pkl", "rb") as f:
                model = GPU_Unpickler(f).load()

            # Load best model
            model.load_state_dict(torch.load(f"{path}/model_{model_no}_seed_{seed}.pt",map_location=device))
            model = model.to(device)

            # load the train and val data for NAS
            test_dataset = BCNB_Dataset('data/test_patches.txt', percentage=0.1, nas_stage=False)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
                

            # 2. Testing with automatic WSI grouping
            predictions, true_labels, wsi_results = majority_vote_test_updated(
                model, test_loader)

            # 3. Final evaluation
            accuracy = accuracy_score(true_labels, predictions) * 100
            print(f"Final WSI-level Accuracy: {accuracy:.2f}%")

            cm, txt = create_confusion_matrix_enhanced(true_labels, predictions, accuracy, save_path=f"model_{model_no}_seed_{seed}")
            stats_text += txt + "\n"
        
        with open(f"model_{model_no}_seed_{seed}.txt", "w") as f:
            f.write(stats_text)
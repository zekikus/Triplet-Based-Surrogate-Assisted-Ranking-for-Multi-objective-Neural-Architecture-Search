import os
import pickle
import numpy as np
from torchinfo import summary

def get_model_parameters(model):
        results = summary(model, (1, 3, 128, 128), verbose=0)
        return results.to_megabytes(results.trainable_params)

def read_model_infos():

    results_path = "results/EBHI"

    data = []
    # 2. .pkl dosyalarını sırayla yükle
    pkl_files = sorted([f for f in os.listdir(results_path) if f.endswith(".pkl")])
    for pkl_file in pkl_files:
        
        with open(os.path.join(results_path, pkl_file), "rb") as f:
            model = pickle.load(f)

        if model.isFeasible and model.fitness > 0:
            chromosome = model.chromosome
            sol_no = model.solNo
            fitness = model.fitness
            nbr_params = get_model_parameters(model)
            data.append((sol_no, chromosome, fitness, nbr_params))

    return data

def is_weakly_dominate(model1, model2):
        """
        Check if vector 'a' is weakly dominate to vector 'b'
        for a multi-objective optimization problem.

        Args:
        a (tuple or list): first vector representing a solution in the objective space
        b (tuple or list): second vector representing a solution in the objective space

        Returns:
        bool: True if 'a' is weakly dominate to 'b', False otherwise
        """
        a = [model1[2], model1[3]]
        b = [model2[2], model2[3]]

        dominates = True
        atLeastOneBetter = False

        for i in range(len(a)):
            if(a[i] > b[i]):
                dominates = False
                break
            elif a[i] < b[i]:
                atLeastOneBetter = True

        return dominates and atLeastOneBetter

def construct_triplets_with_one_objectives(architectures):
    triplets = []
    
    for anchor in architectures:
        # Find architectures dominated by anchor
        positives = [arch for arch in architectures 
                    if anchor[2] < arch[2]]
        
        # Find architectures that dominate anchor  
        negatives = [arch for arch in architectures
                    if arch[2] < anchor[2]]
        
        # Create triplets
        for pos in positives:
            for neg in negatives:
                triplets.append((anchor, pos, neg))
    
    return triplets


def construct_triplets_with_two_objectives(architectures):
    triplets = []
    
    for anchor in architectures:
        # Find architectures dominated by anchor
        positives = [arch for arch in architectures 
                    if is_weakly_dominate(anchor, arch)]
        
        # Find architectures that dominate anchor  
        negatives = [arch for arch in architectures
                    if is_weakly_dominate(arch, anchor)]
        
        # Create triplets
        for pos in positives:
            for neg in negatives:
                triplets.append((anchor, pos, neg))
    
    return triplets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class TripletDataset(Dataset):
    """Dataset class for triplet data"""
    def __init__(self, triplets, scaler=None):
        self.triplets = triplets
        self.scaler = scaler
        
        if self.scaler is not None:
            
            # Extract features from triplets and normalize
            all_features = []
            for anchor, pos, neg in triplets:
                all_features.extend([anchor[1], pos[1], neg[1]])  # Skip sol_no
            
            all_features = np.array(all_features)

            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
        
            # Normalize triplets
            self.normalized_triplets = []
            for anchor, pos, neg in triplets:
                anchor_norm = self.scaler.transform([anchor[1]])[0]
                pos_norm = self.scaler.transform([pos[1]])[0]
                neg_norm = self.scaler.transform([neg[1]])[0]
                self.normalized_triplets.append((anchor_norm, pos_norm, neg_norm))
    
    def __len__(self):
        if self.scaler is not None:
            return len(self.normalized_triplets)
        else:
            return len(self.triplets)
    
    def __getitem__(self, idx):
        if self.scaler is not None:
            anchor, pos, neg = self.normalized_triplets[idx]
        else:
            anchor, pos, neg = self.triplets[idx]

        return (torch.FloatTensor(anchor[1]), 
                torch.FloatTensor(pos[1]), 
                torch.FloatTensor(neg[1]))

class TripletNetwork(nn.Module):
    """Triplet Network for learning embeddings"""
    def __init__(self, input_dim, embedding_dim=64, hidden_dims=[128, 256, 128]):
        super(TripletNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def forward_triplet(self, anchor, positive, negative):
        anchor_emb = self.forward(anchor)
        positive_emb = self.forward(positive)
        negative_emb = self.forward(negative)
        return anchor_emb, positive_emb, negative_emb

class TripletLoss(nn.Module):
    """Triplet Loss with margin"""
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

def train_triplet_network(triplets, input_dim, epochs=100, batch_size=32, 
                         learning_rate=0.001, embedding_dim=64, margin=1.0, device=None):
    """Train the triplet network"""
    
    # Create dataset and dataloader
    dataset = TripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = TripletNetwork(input_dim, embedding_dim).to(device)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    losses = []
    model.train()
    
    print(f"Training on {device}")
    print(f"Dataset size: {len(dataset)} triplets")
    
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_emb, pos_emb, neg_emb = model.forward_triplet(anchor, positive, negative)
            
            # Calculate loss
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    return model, losses, dataset.scaler

def evaluate_embeddings(model, data, scaler, device):
    """Evaluate the learned embeddings"""
    model.eval()
    
    # Normalize data
    # features = np.array([item[1] for item in data])  # Skip sol_no
    #normalized_features = scaler.transform(features)
    features = np.array([data[1]]) 
    # Get embeddings
    with torch.no_grad():
        embeddings = model(torch.FloatTensor(features).to(device))
        embeddings = embeddings.cpu().numpy()
    
    return embeddings

def plot_training_loss(losses):
    """Plot training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Triplet Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def visualize_embeddings_2d(embeddings, data, title="Learned Embeddings"):
    """Visualize embeddings in 2D using PCA"""
    from sklearn.decomposition import PCA
    
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    plt.figure(figsize=(12, 5))
    
    # Plot by fitness
    plt.subplot(1, 2, 1)
    fitness_values = [item[2] for item in data]
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=fitness_values, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Fitness')
    plt.title(f'{title} - Colored by Fitness')
    plt.xlabel('Embedding Dim 1')
    plt.ylabel('Embedding Dim 2')
    
    # Plot by parameters (if available)
    if len(data[0]) > 2:
        plt.subplot(1, 2, 2)
        param_values = [item[2] for item in data]
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=param_values, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Parameters')
        plt.title(f'{title} - Colored by Parameters')
        plt.xlabel('Embedding Dim 1')
        plt.ylabel('Embedding Dim 2')
    
    plt.tight_layout()
    plt.show()

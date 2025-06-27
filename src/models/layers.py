import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.structure.hypergraphs import Hypergraph
_LAYER_UIDS = {}
from src.utils.helpers import try_gpu

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class HGNNConv(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:

        # Args:
        #     X (``torch.Tensor``):
        #     hg (``dhg.Hypergraph``)
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.drop(self.act(X))

        return X

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, drop_rate, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.drop_rate = drop_rate
        self.act = act

    def forward(self, X, Z):
        X = F.dropout(X, self.drop_rate, training=self.training)
        Z = F.dropout(Z, self.drop_rate, training=self.training)
        H = self.act(torch.mm(X, Z.T))
        return H

class projection(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(projection, self).__init__()
        self.linear1 = nn.Linear(hidden_channels, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
            num_hyperedges: int,
    ):
        super(Attention, self).__init__()
        self.projection = projection(hidden_channels, hidden_size)

    def forward(self, X, hg):
        he = hg.state_dict['raw_groups']['main']
        edges = list(he.keys())
        edge_w = []
        Z = None  # Initialize Z outside the loop
        
        for i in range(len(edges)):
            if len(edges[i][0]) == 0:  # Skip empty hyperedges
                continue
                
            try:
                # Convert to explicitly typed integer tensor
                index = torch.tensor(list(edges[i][0]), dtype=torch.int64)
                index = index.to(device=try_gpu())
                
                # Handle cases where index is out of bounds
                valid_indices = index < X.shape[0]
                if not torch.all(valid_indices):
                    print(f"Warning: Found invalid indices: {index[~valid_indices]}")
                    index = index[valid_indices]
                
                if len(index) == 0:  # Skip if no valid indices
                    continue
                    
                x_he = torch.index_select(X, 0, index)
                w = self.projection(x_he)
                
                # Handle NaN values in attention weights
                if torch.isnan(w).any():
                    print(f"Warning: NaN values in attention weights for hyperedge {i}")
                    continue
                    
                beta = torch.softmax(w, dim=0)
                edge_w.append(beta)
                z_batch = (beta * x_he).sum(0)
                z_batch = F.leaky_relu(z_batch)
                z_batch = z_batch.unsqueeze(0)
                
                if Z is None:
                    Z = z_batch
                else:
                    Z = torch.cat([Z, z_batch], 0)
                    
            except Exception as e:
                print(f"Error processing hyperedge {i}: {str(e)}")
                continue
                
        # Handle the case when no valid hyperedges were processed
        if Z is None:
            print("Warning: No valid hyperedges processed in Attention layer")
            # Return zeros tensor of appropriate shape
            Z = torch.zeros(0, X.shape[1], device=X.device)
        
        return torch.tanh(Z), edge_w

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy(preds, labels)

    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

class Attention_PC(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(Attention_PC, self).__init__()
        self.projections = projection(hidden_channels, hidden_size)

    def forward(self, x1, x2):
        edge_w = []
        for i in range(x1.shape[0]):
            x_he1 = x1[i, :]
            x_he2 = x2[i, :]
            x_he1 = x_he1.unsqueeze(0)
            x_he2 = x_he2.unsqueeze(0)
            x_he = torch.cat([x_he1, x_he2], 0)
            w = self.projections(x_he)
            beta = torch.softmax(w, dim=0)
            edge_w.append(beta)
            z_batch = (beta * x_he)
            z_batch = z_batch.reshape(1, -1)
            z_batch = z_batch.unsqueeze(0)
            if (i > 0):
                Z = torch.cat([Z, z_batch], 0)
            else:
                Z = z_batch
        return torch.tanh(Z), edge_w

class Concentration(nn.Module):
    def __init__(
            self,
            hidden_channels: int,
            hidden_size: int,
    ):
        super(Concentration, self).__init__()

    def forward(self, X, GP_info):
        edges = GP_info
        Z_bath = torch.tensor(())
        Z_bath = Z_bath.to(device=try_gpu())
        for i in range(len(edges)):
            Z = torch.mean(X[edges[i]], dim=0).reshape(1, -1)
            Z_bath = torch.cat((Z_bath, Z), 0)
        return Z_bath

class Classifier_DNN(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
    ):
        super(Classifier_DNN, self).__init__()

        self.fc1 = nn.Linear(in_features, int(in_features / 1))
        self.bn1 = nn.BatchNorm1d(int(in_features / 1))

        self.fc2 = nn.Linear(int(in_features / 1), int(in_features / 2))
        self.bn2 = nn.BatchNorm1d(int(in_features / 2))

        self.fc3 = nn.Linear(int(in_features / 2), int(in_features / 4))
        self.bn3 = nn.BatchNorm1d(int(in_features / 4))

        self.fc5 = nn.Linear(int(in_features / 4), out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.sigmoid(self.fc5(x))

        return x

def multi_network_loss_function(preds_list, labels_list, mu, logvar, n_nodes, norm_list, pos_weight_list):
    """Loss function for multiple networks with a shared bottleneck using averaging."""
    # Reconstruction loss for each network
    reconstruction_loss = 0
    num_valid_networks = 0
    
    for i, (preds, labels, norm, pos_weight) in enumerate(zip(preds_list, labels_list, norm_list, pos_weight_list)):
        # Add numerical stability
        epsilon = 1e-10
        preds = torch.clamp(preds, min=epsilon, max=1-epsilon)
        
        # Create weight tensor that applies pos_weight only to positive examples
        weight = torch.ones_like(labels)
        weight[labels > 0] = pos_weight
        
        # Use mean reduction instead of sum
        bce_loss = F.binary_cross_entropy(
            preds, 
            labels, 
            weight=weight,
            reduction='mean'  # Use mean instead of sum
        )
        
        # Scale by norm factor but use a reasonable maximum to prevent explosion
        norm = min(norm, 10.0)  # Cap the norm value to prevent extreme scaling
        weighted_loss = norm * bce_loss
        
        # Only include valid (finite) losses
        if torch.isfinite(weighted_loss).all():
            reconstruction_loss += weighted_loss
            num_valid_networks += 1
    
    # Average the reconstruction loss across valid networks
    if num_valid_networks > 0:
        reconstruction_loss /= num_valid_networks
    else:
        # Fallback if no valid networks
        reconstruction_loss = torch.tensor(0.1, device=mu.device, requires_grad=True)
    
    # KL divergence - use mean instead of sum to scale with batch size
    # Also clamp to prevent extreme values
    kl_per_element = 1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2)
    kl_per_element = torch.clamp(kl_per_element, min=-100, max=100)
    KLD = -0.5 * torch.mean(kl_per_element)
    
    # Balance the KL term with the reconstruction term
    beta = 0.1  # Start with a smaller beta to focus on reconstruction first
    total_loss = reconstruction_loss + beta * KLD
    
    # Final safety check
    if not torch.isfinite(total_loss):
        print("Warning: Non-finite loss detected, using fallback value")
        return torch.tensor(1.0, device=mu.device, requires_grad=True)
        
    return total_loss

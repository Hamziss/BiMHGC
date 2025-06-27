import torch.nn as nn
from .layers import Concentration, Classifier_DNN,InnerProductDecoder,Attention,HGNNConv
import torch
import torch.nn.functional as F

def identity(x):
    return x
class ParallelHGVAE(nn.Module):
    """Parallel Hypergraph Variational Autoencoder with a shared bottleneck."""
    def __init__(
            self,
            in_channels: int,
            hid_channels: int,
            num_classes: int,
            num_hyperedges_list: list,  # List of hyperedge counts for each network
            use_bn: bool = False,
            drop_rate: float = 0.5,
            num_networks: int = 2,      # Number of parallel PPI networks
    ) -> None:
        super().__init__()
        
        # Create parallel encoders for each network
        self.encoders = nn.ModuleList()
        for i in range(num_networks):
            encoder = nn.ModuleDict({
                'layer1': HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate),
                'layer2': HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True),
                'layer3': HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
            })
            self.encoders.append(encoder)
            
        # Shared bottleneck fusion layer
        self.fusion = nn.Linear(num_classes * num_networks, num_classes)
        
        # Create parallel attention modules for each network
        self.attentions = nn.ModuleList()
        for i in range(num_networks):
            self.attentions.append(Attention(num_classes, 50, num_hyperedges_list[i]))
            
        # Create parallel decoders for each network
        self.decoders = nn.ModuleList()
        for i in range(num_networks):
            self.decoders.append(InnerProductDecoder(drop_rate, act=identity))

    def encode_single(self, x, hg, encoder_idx):
        """Encode a single network."""
        encoder = self.encoders[encoder_idx]
        hidden1 = encoder['layer1'](x, hg)
        return encoder['layer2'](hidden1, hg), encoder['layer3'](hidden1, hg)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, X, hg_list):
        # Print shapes for debugging      
        
        # Encode each network separately
        mu_list, logvar_list = [], []
        X_encoded_list = []
        
        for i, hg in enumerate(hg_list):
            try:               
                mu, logvar = self.encode_single(X, hg, i)                
                mu_list.append(mu)
                logvar_list.append(logvar)
                X_encoded = self.reparameterize(mu, logvar)
                X_encoded_list.append(X_encoded)
            except Exception as e:
                print(f"Error in encoding network {i+1}: {str(e)}")
                # Create dummy encodings if network fails
                dummy_size = mu_list[0].shape if mu_list else (X.shape[0], self.num_classes)
                mu_list.append(torch.zeros(dummy_size, device=X.device))
                logvar_list.append(torch.zeros(dummy_size, device=X.device))
                X_encoded_list.append(torch.zeros(dummy_size, device=X.device))
        
        # Combine encodings in the shared bottleneck
       
        combined_encoding = torch.cat(X_encoded_list, dim=1)
        shared_bottleneck = self.fusion(combined_encoding)   
        
        # Process each network through attention and decoder
        Z_list, H_list, edge_w_list = [], [], []
        
        for i, hg in enumerate(hg_list):
            try:             
                Z, edge_w = self.attentions[i](shared_bottleneck, hg)
             
                H = self.decoders[i](shared_bottleneck, Z)
                H = torch.sigmoid(H)
          
                
                Z_list.append(Z)
                H_list.append(H)
                edge_w_list.append(edge_w)
            except Exception as e:
                print(f"Error in processing network {i+1}: {str(e)}")
                # Create dummy outputs if network fails
                Z_list.append(torch.zeros(0, shared_bottleneck.shape[1], device=X.device))
                dummy_H = torch.zeros((hg.num_vertices, hg.num_vertices), device=X.device)
                H_list.append(dummy_H)
                edge_w_list.append([])
        
        # Compute combined mu and logvar for KL divergence
        combined_mu = torch.cat(mu_list, dim=1)
        combined_logvar = torch.cat(logvar_list, dim=1)
        
        return shared_bottleneck, Z_list, H_list, combined_mu, combined_logvar, edge_w_list

class PCpredict(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_class: int,
            use_bn: bool = False,
            drop_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = drop_rate
        self.concentration = Concentration(in_channels,in_channels)
        self.classfier = Classifier_DNN(in_channels, num_class)

    def forward(self, x1,GP_info):
        Z  = self.concentration(x1, GP_info)
        out = self.classfier(Z)
        return out


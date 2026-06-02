import torch
from torch import nn
from einops import rearrange
import copy
# from ultralytics import YOLO

# =========================================================
# MODEL Transformers
# =========================================================


class TransfPolicy(nn.Module):
    def __init__(self, image_encoder, image_embed_dim, pose_dim, pose_embed_dim,
                 action_dim, seq_len, hidden_dim=256, n_heads=4, n_layers=3):
        """
        Args:
            image_encoder: a CNN (e.g. timm model) producing features of dim `image_embed_dim`
            image_embed_dim: output dim of CNN encoder
            pose_dim: dimension of raw pose input
            pose_embed_dim: embedding dim for pose
            action_dim: output action dimension
            seq_len: maximum sequence length
        """
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Image encoders (shared weights or separate)
        self.img_enc1 = copy.deepcopy(image_encoder)
        self.img_enc2 = copy.deepcopy(image_encoder)  # could also use distinct encoder instance

        # Pose encoder
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.ReLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim)
        )

        # Projection to common dimension for attention
        self.proj_img = nn.Linear(image_embed_dim, hidden_dim)
        self.proj_pose = nn.Linear(pose_embed_dim, hidden_dim)

        self.fusion_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

        # Transformer Encoder for temporal fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final MLP to actions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, cam1, cam2, pose):
        """
        cam1, cam2: (B,T,3,H,W) RGB images
        pose: (B,T,P) pose vectors
        Returns:
            actions: (B,T,action_dim)
        """
        B, T, C, H, W = cam1.shape
        # Flatten batch and time to encode images
        cam1_flat = rearrange(cam1, 'b t c h w -> (b t) c h w')
        cam2_flat = rearrange(cam2, 'b t c h w -> (b t) c h w')
        # CNN encode (outputs (B*T, image_embed_dim))
        f1 = self.img_enc1(cam1_flat)
        f2 = self.img_enc2(cam2_flat)
        # Restore time dimension
        f1 = rearrange(f1, '(b t) d -> b t d', b=B, t=T)
        f2 = rearrange(f2, '(b t) d -> b t d', b=B, t=T)

        # Pose encoding
        f_pose = self.pose_mlp(pose)  # (B,T, pose_embed_dim)

        # Project to hidden dim
        img_feat1 = self.proj_img(f1)     # (B,T,hidden_dim)
        img_feat2 = self.proj_img(f2)     # (B,T,hidden_dim)
        pose_feat = self.proj_pose(f_pose)  # (B,T,hidden_dim)

        # Feature fusion (concatenate along feature axis)
        fused = torch.cat([img_feat1, img_feat2, pose_feat], dim=-1)  # (B,T, 3*hidden_dim)
        fused = self.fusion_proj(fused)  # project back to hidden_dim

        # Positional encoding (optional) can be added here
        fused = fused + self.pos_embed[:, :T, :]
        
        # Transformer expects (T,B,E) 
        fused = fused.permute(1, 0, 2)  # -> (T, B, hidden_dim)
        traj = self.transformer(fused)  # (T, B, hidden_dim)
        traj = traj.permute(1, 0, 2)    # (B, T, hidden_dim)

        # Predict actions
        actions = self.action_head(traj)  # (B, T, action_dim)
        return actions




# =========================================================
# MODEL CNN
# =========================================================

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2), nn.Tanh(),
            nn.Conv2d(32, 64, 3, stride=2), nn.Tanh(),
            nn.Conv2d(64, 128, 3, stride=2), nn.Tanh(),
            nn.Flatten(),
            nn.Linear(596608, out_dim)  # adapt if resolution changes
        )

    def forward(self, x):
        return self.net(x)


class CnnPolicy(nn.Module):
    def __init__(self, pose_dim, action_dim, in_channels = 3, hidden_dim=128, pretrained = True, pc = True):
        super().__init__()

        if pretrained:
            
            if not pc:

                self.f1_net = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 128),
                    nn.Tanh()
                )
                self.f2_net = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 128),
                    nn.Tanh()
                )
                self.f3_net = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.GELU(),
                    nn.Linear(1024, 128),
                    nn.Tanh()
                )

                self.forward = self.forward_pre
            else:
                self.forward = self.forward_pc



        if not pretrained:
            self.cnn1 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn2 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn3 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            
            self.forward = self.forward_cnn




        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.Tanh()
        )


        # 🔥 Proper fusion layer (fixed)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim)
        )

        # Optional novelty: gating
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward_cnn(self, cam, cam_ext, cam_front, pose):
        f1 = self.cnn1(cam)
        f2 = self.cnn2(cam_ext)
        f3 = self.cnn3(cam_front)
        f_pose = self.pose_mlp(pose)

        fused_raw = torch.cat([f1, f2, f3, f_pose], dim=-1)

        gate = torch.sigmoid(self.gate(fused_raw))
        fused = self.fusion(fused_raw) * gate

        return self.head(fused)
    
    def forward_pre(self, f1, f2, f3, pose):

        f1 = self.f1_net(f1)
        f2 = self.f1_net(f2)
        f3 = self.f1_net(f3)

        f_pose = self.pose_mlp(pose)

        fused_raw = torch.cat([f1, f2, f3, f_pose], dim=-1)

        gate = torch.sigmoid(self.gate(fused_raw))
        fused = self.fusion(fused_raw) * gate

        return self.head(fused)
    
    def forward_pc(self, pc, pose):

        f_pose = self.pose_mlp(pose)

        fused_raw = torch.cat([pc, f_pose], dim=-1)

        # gate = torch.tanh(self.gate(fused_raw))
        fused = self.fusion(fused_raw)# * gate

        return self.head(fused)
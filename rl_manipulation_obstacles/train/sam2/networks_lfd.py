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



# =========================================================
# DCT FEATIRE REDUCER
# =========================================================
class FastDCTFeatureReducer:
    """
    Fast deterministic DCT-based feature reducer.

    Ensures fixed output size K < D by truncating frequency components.
    """

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dct_matrix = self.create_dct_matrix(self.input_dim)

    @staticmethod
    def create_dct_matrix(D, device="cuda"):
        n = torch.arange(D, device=device).float()
        k = torch.arange(D, device=device).float()

        M = torch.cos(
            torch.pi / D * (n[None, :] + 0.5) * k[:, None]
        )

        M[0] *= 1.0 / torch.sqrt(torch.tensor(D, device=device))
        M[1:] *= torch.sqrt(torch.tensor(2.0 / D, device=device))

        return M

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D) -> z: (B, K)
        """

        if x.dim() != 2:
            raise ValueError(f"Expected (B, D), got {x.shape}")

        B, D = x.shape
        K = self.output_dim
      #  if D==K: return x

        coeffs = x @ self.dct_matrix.T

        return coeffs[..., :K]

    def decode(self, z: torch.Tensor, original_dim: int = None) -> torch.Tensor:
        """
        Approximate reconstruction (optional).
        """
        K = z.shape[-1]
        D = self.input_dim

        if original_dim is None:
            original_dim = D
    
#        if D==K: return z
        
        coeffs = torch.zeros( *z.shape[:-1],D,device=z.device,dtype=z.dtype)
        coeffs[..., :K] = z
        x_rec = coeffs @ self.dct_matrix


class CnnPolicy(nn.Module):
    def __init__(self, pose_dim, action_dim, in_channels = 3, hidden_dim=128, pred_horizon = 5, pretrained = True, pc = True):
        super().__init__()

        self.action_dim = action_dim
        self.pred_horizon = pred_horizon

        if pretrained:
            
            if not pc:
                inc = 4

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
                    nn.LayerNorm(hidden_dim),

                    nn.Linear(1024, 128),
                    nn.Tanh(),
                    nn.LayerNorm(hidden_dim),
                )

                self.forward = self.forward_pre
            else:
                inc = 2
                H = 5

                self.mlp = nn.Sequential(
                    nn.Linear(3,128), # falta poner el tamaño
                    nn.LayerNorm(128), # falta poner el tamaño
                    nn.ReLU(),
                    nn.Linear(128,256), # falta poner el tamaño
                    nn.LayerNorm(256), # falta poner el tamaño
                    nn.ReLU(),
                    nn.Linear(256,512), # falta poner el tamaño
                    nn.LayerNorm(512), # falta poner el tamaño
                    nn.ReLU(),
                    nn.Linear(512,1024), # falta poner el tamaño
                    nn.LayerNorm(1024), # falta poner el tamaño
                    nn.ReLU(),
                    
                )
                self.after_mean = nn.Sequential(
                    nn.Linear(512,hidden_dim), # falta poner el tamaño
                    nn.LayerNorm(hidden_dim), # falta poner el tamaño)
                )
                self.after_max = nn.Sequential(
                    nn.Linear(512,hidden_dim), # falta poner el tamaño
                    nn.LayerNorm(hidden_dim), # falta poner el tamaño)
                )

                # self.forward = self.forward_pc

        else:
            self.cnn1 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn2 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn3 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            
            self.forward = self.forward_cnn


        self.dct = FastDCTFeatureReducer(input_dim=524288, output_dim=hidden_dim) # 128


        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )


        # 🔥 Proper fusion layer (fixed)
        self.fusion = nn.Sequential(
            nn.Linear(inc * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.gru_1 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.gru_2 = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )

        # # Optional novelty: gating
        # self.gate = nn.Linear(inc * hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            # nn.Dropout(0.15),
            nn.Linear(hidden_dim, action_dim),
            nn.LayerNorm(action_dim)
        )

        # self.forward = self.forward_temporal_DCT
        self.forward = self.forward_temporal_DCT_raw

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

        f_pc = self.mlp(pc)
        max_feat = torch.max(f_pc, dim=1)[0]
        mean_feat = torch.mean(f_pc, dim=1)

        max_feat = self.after_max(max_feat)
        mean_feat = self.after_mean(mean_feat)

        feat = torch.cat([max_feat, mean_feat], dim=-1)
        
        f_pose = self.pose_mlp(pose)

        fused_raw = torch.cat([feat, f_pose], dim=-1)

        # gate = torch.tanh(self.gate(fused_raw))
        fused = self.fusion(fused_raw)# * gate

        return self.head(fused).view(pc.shape[0], -1, self.action_dim)
    
    def forward_temporal(self, pc_seq, pose_seq):

        """
        pc_seq   : [B,T,512,3]
        pose_seq : [B,T,pose_dim]
        """

        B, T, N, _ = pc_seq.shape

        # -----------------------------------------------------
        # Flatten batch and time
        # -----------------------------------------------------

        pc = pc_seq.reshape(B * T, N, 3)
        pose = pose_seq.reshape(B * T, -1)

        # -----------------------------------------------------
        # PointNet
        # -----------------------------------------------------

        f_pc = self.mlp(pc)

        max_feat = torch.max(f_pc, dim=1)[0]
        mean_feat = torch.mean(f_pc, dim=1)

        max_feat = self.after_max(max_feat)
        mean_feat = self.after_mean(mean_feat)

        feat_pc = torch.cat(
            [max_feat, mean_feat],
            dim=-1
        )

        # -----------------------------------------------------
        # Pose encoder
        # -----------------------------------------------------

        f_pose = self.pose_mlp(pose)

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        fused = torch.cat(
            [feat_pc, f_pose],
            dim=-1
        )

        fused = self.fusion(fused)

        # -----------------------------------------------------
        # Restore time dimension
        # -----------------------------------------------------

        fused = fused.reshape(
            B,
            T,
            -1
        )

        # -----------------------------------------------------
        # GRU
        # -----------------------------------------------------

        gru_out, h_n = self.gru(fused)

        # last hidden state
        temporal_feat = h_n[-1]

        # alternatively:
        # temporal_feat = gru_out[:, -1]

        # -----------------------------------------------------
        # Predict future trajectory
        # -----------------------------------------------------

        pred = self.head(temporal_feat)

        pred = pred.reshape(
            B,
            self.pred_horizon,
            self.action_dim
        )

        return pred
    
    def forward_temporal_DCT_raw(self, pc_seq, pose_seq):

        """
        pc_seq   : [B,T,512,3]
        pose_seq : [B,T,pose_dim]
        """

        

        B, T, N, _ = pc_seq.shape

        # -----------------------------------------------------
        # Flatten batch and time
        # -----------------------------------------------------

        pc = pc_seq.reshape(B * T, N, 3)
        pose = pose_seq.reshape(B * T, -1)

        # -----------------------------------------------------
        # PointNet
        # -----------------------------------------------------
        print(pc.shape)
        f_pc = self.mlp(pc)

        # max_feat = torch.max(f_pc, dim=1)[0]
        # mean_feat = torch.mean(f_pc, dim=1)

        # max_feat = self.after_max(max_feat)
        # mean_feat = self.after_mean(mean_feat)

        # feat_pc = torch.cat(
        #     [max_feat, mean_feat],
        #     dim=-1
        # )


        f_pc_dct = self.dct.encode(f_pc.view(f_pc.shape[0], -1))        

        # -----------------------------------------------------
        # Pose encoder
        # -----------------------------------------------------

        f_pose = self.pose_mlp(pose)

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        # fused = torch.cat(
        #     [f_pc_dct, f_pose],
        #     dim=-1
        # )

        # fused = self.fusion(fused)

        # -----------------------------------------------------
        # Restore time dimension
        # -----------------------------------------------------

        f_pose = f_pose.reshape(
            B,
            T,
            -1
        )
        f_pc_dct = f_pc_dct.reshape(
            B,
            T,
            -1
        )

        # -----------------------------------------------------
        # GRU
        # -----------------------------------------------------

        gru_out1, h_n1 = self.gru_1(f_pose)

        # last hidden state
        temporal_feat1 = h_n1[-1]

        gru_out2, h_n2 = self.gru_2(f_pose)

        # last hidden state
        temporal_feat2 = h_n2[-1]

        # alternatively:
        # temporal_feat = gru_out[:, -1]

        # -----------------------------------------------------
        # Predict future trajectory
        # -----------------------------------------------------

        temporal_feat = torch.cat((temporal_feat1, temporal_feat2), dim = -1)


        pred = self.head(temporal_feat)

        pred = pred.reshape(
            B,
            self.pred_horizon,
            self.action_dim
        )

        return pred
    

    def forward_temporal_DCT(self, pc_seq, pose_seq):

        """
        pc_seq   : [B,T,128]
        pose_seq : [B,T,pose_dim]
        """

        B, T, _ = pc_seq.shape

        # -----------------------------------------------------
        # Flatten batch and time
        # -----------------------------------------------------

        pc = pc_seq.reshape(B * T, -1)
        pose = pose_seq.reshape(B * T, -1)

        # -----------------------------------------------------
        # PointNet ENCODER
        # -----------------------------------------------------
        f_pc_dct = self.dct.encode(pc.view(pc.shape[0], -1))        

        # -----------------------------------------------------
        # Pose encoder
        # -----------------------------------------------------
        f_pose = self.pose_mlp(pose)


        # -----------------------------------------------------
        # Concatenation
        # -----------------------------------------------------
        # fused = torch.stack((f_pc_dct, f_pose), dim=-1).reshape(B,T,-1)
        f_pc_dct = f_pc_dct.reshape(B,T,-1)
        f_pose = f_pose.reshape(B,T,-1)


        # -----------------------------------------------------
        # GRU
        # -----------------------------------------------------
        gru_out, h_n1 = self.gru_1(f_pc_dct)
        gru_out, h_n2 = self.gru_2(f_pose)

        # last hidden state
        temporal_feat1 = h_n1[-1]
        temporal_feat2 = h_n2[-1]

        temporal_feat = torch.cat((temporal_feat1, temporal_feat2), dim = -1)

        # -----------------------------------------------------
        # Predict future trajectory
        # -----------------------------------------------------

        pred = self.head(temporal_feat)

        return pred
    




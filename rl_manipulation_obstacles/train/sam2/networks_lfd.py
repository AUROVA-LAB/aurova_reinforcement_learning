import torch
from torch import nn
# from einops import rearrange
import copy

class AttentionPool(nn.Module):
    def __init__(self, F):
        super().__init__()
        self.score = nn.Linear(F,1)

    def forward(self,x):
        # x: [B,T,N,F]

        w = self.score(x)              # [B,T,N,1]
        w = torch.softmax(w, dim=2)

        pooled = (x*w).sum(dim=2)

        return pooled


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        ff_dim=1024,
        dropout=0.1,
        max_len=100
    ):
        super().__init__()

        # Project features to transformer dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable temporal positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x: [B,T,F]
        """

        B, T, _ = x.shape

        x = self.input_proj(x)

        # Add temporal position
        x = x + self.pos_embedding[:, :T]

        x = self.transformer(x)

        return x



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

        return x_rec


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
                    # nn.Linear(3,128), # falta poner el tamaño
                    # nn.LayerNorm(128), # falta poner el tamaño
                    # nn.ReLU(),
                    nn.Linear(3,64), # falta poner el tamaño
                    nn.LayerNorm(64), # falta poner el tamaño
                    nn.ReLU(),
                    nn.Linear(64,128), # falta poner el tamaño
                    nn.LayerNorm(128), # falta poner el tamaño
                    nn.ReLU(),
                    nn.Linear(128,256), # falta poner el tamaño
                    nn.LayerNorm(256), # falta poner el tamaño
                    nn.ReLU(),
                )

                self.proj_mlp = nn.Sequential(
                    nn.Linear(256,64),
                    nn.LayerNorm(64)
                )
                # self.after_mean = nn.Sequential(
                #     nn.Linear(512,hidden_dim), # falta poner el tamaño
                #     nn.LayerNorm(hidden_dim), # falta poner el tamaño)
                # )
                # self.after_max = nn.Sequential(
                #     nn.Linear(512,hidden_dim), # falta poner el tamaño
                #     nn.LayerNorm(hidden_dim), # falta poner el tamaño)
                # )

                # self.forward = self.forward_pc

        else:
            self.cnn1 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn2 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            self.cnn3 = SimpleCNN(in_channels=in_channels, 
                                out_dim=hidden_dim)
            
            self.forward = self.forward_cnn


        # self.dct = FastDCTFeatureReducer(input_dim=1024, output_dim=hidden_dim) # 128


        self.att_pool = AttentionPool(F = 128)

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.bert_mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 2*hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2*hidden_dim),

        )


        # 🔥 Proper fusion layer (fixed)
        # self.fusion = nn.Sequential(
        #     nn.Linear(inc * hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim)
        # )

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

        self.cls_token = nn.Parameter(torch.randn(1, 1, 2*hidden_dim))

        self.temporal_transformer = TemporalTransformer(
            input_dim=hidden_dim*2,
            d_model=128,
            nhead=8,
            num_layers=4
        )

        # # Optional novelty: gating
        # self.gate = nn.Linear(inc * hidden_dim, hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            # nn.Dropout(0.15),
            nn.Linear(hidden_dim, action_dim),
        )

        self.head2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            # nn.Dropout(0.15),
            nn.Linear(hidden_dim, 6),
        )

        # self.forward = self.forward_temporal_DCT
        # self.forward = self.forward_temporal_DCT_raw
        self.forward = self.forward_temporal_DCT_BERT




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
    
    def forward_temporal_DCT(self, pc_seq, pose_seq):

        """
        pc_seq   : [B,T,512,3]
        pose_seq : [B,T,pose_dim]
        """

        B, T, N, _ = pc_seq.shape

        # -----------------------------------------------------
        # Pose encoding
        # -----------------------------------------------------
        pose = pose_seq.reshape(B * T, -1)
        f_pose = self.pose_mlp(pose)
        f_pose = f_pose.reshape(B, T, -1)

        # -----------------------------------------------------
        # Spatial pooling (Point cloud)
        # -----------------------------------------------------
        f_scene = self.att_pool(pc_seq)  # [B,T,128]

        # -----------------------------------------------------
        # Fusion per timestep
        # -----------------------------------------------------
        x = torch.cat([f_scene, f_pose], dim=-1)  # [B,T,F]

        # -----------------------------------------------------
        # CLS token insertion
        # -----------------------------------------------------
        cls = self.cls_token.expand(B, 1, -1)     # [B,1,F]
        x = torch.cat([cls, x], dim=1)            # [B,T+1,F]

        # -----------------------------------------------------
        # Temporal transformer
        # -----------------------------------------------------
        x = self.temporal_transformer(x)

        # -----------------------------------------------------
        # Use CLS output for prediction
        # -----------------------------------------------------
        cls_out = x[:, 0]   # [B, d_model]
        pred = self.head(cls_out)  # [B, action_dim]
        
        # pred = pred.reshape(
        #     B,
        #     self.pred_horizon,
        #     self.action_dim
        # )

        return pred
    
    def forward_temporal_DCT_BERT(self, pc_seq):

        """
        pc_seq   : [B,T,F]
        pose_seq : [B,T,pose_dim]
        """

        B, T, F = pc_seq.shape

        # -----------------------------------------------------
        # Pose encoding
        # -----------------------------------------------------
        # pose = pose_seq.reshape(B * T, -1)
        # f_pose = self.pose_mlp(pose)
        # f_pose = f_pose.reshape(B, T, -1)

        # -----------------------------------------------------
        # BERT encoding
        # -----------------------------------------------------
        pc_seq = pc_seq.reshape(B*T, -1)
        f_scene = self.bert_mlp(pc_seq)
        f_scene = f_scene.reshape(B, T, -1)

        # -----------------------------------------------------
        # Fusion per timestep
        # -----------------------------------------------------
        # x = torch.cat([f_scene, f_pose], dim=-1)  # [B,T,F]
        x = f_scene

        # -----------------------------------------------------
        # CLS token insertion
        # -----------------------------------------------------
        cls = self.cls_token.expand(B, 1, -1)     # [B,1,F]
        x = torch.cat([cls, x], dim=1)            # [B,T+1,F]

        # -----------------------------------------------------
        # Temporal transformer
        # -----------------------------------------------------
        x = self.temporal_transformer(x)

        # -----------------------------------------------------
        # Use CLS output for prediction
        # -----------------------------------------------------
        cls_out = x[:, 0]   # [B, d_model]
        pred = self.head(cls_out)  # [B, action_dim]
        pred_mag = self.head2(cls_out)  # [B, action_dim]
        
        # pred = pred.reshape(
        #     B,
        #     self.pred_horizon,
        #     self.action_dim
        # )

        return pred, pred_mag
    

    def forward_temporal_DCT_raw(self, pc_seq, pose_seq):

        """
        pc_seq   : [B,T,128]
        pose_seq : [B,T,pose_dim]
        """

        B, T, N, _ = pc_seq.shape

        # -----------------------------------------------------
        # Flatten batch and time
        # -----------------------------------------------------

        pc = pc_seq.reshape(B * T, N, -1)
        pose = pose_seq.reshape(B * T, -1)

        # -----------------------------------------------------
        # PointNet ENCODER
        # -----------------------------------------------------
        f_pc_dct = self.mlp(pc)
        f_pc_dct = torch.max(f_pc_dct, dim=1)[0]
        f_pc_dct = self.proj_mlp(f_pc_dct)


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
    




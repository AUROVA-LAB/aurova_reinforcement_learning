import torch
import numpy as np
from easydict import EasyDict

from Point_BERT.models.Point_BERT import PointTransformer


from data import *
import open3d as o3d
from train_utils import *

# ---------------------------------------------------
# Build model
# ---------------------------------------------------

config = EasyDict({
    'trans_dim':384,
    'depth':12,
    'drop_path_rate':0.1,
    'cls_dim':40,
    'num_heads':6,
    'group_size':32,
    'num_group':128,
    'encoder_dims':256
})

model = PointTransformer(config)

# load pretrained checkpoint
# ckpt = torch.load(
#     "Point-BERT.pth",
#     map_location="cpu",
#     weights_only=False
# )

# # depending on checkpoint structure:
# model.load_state_dict(
#     ckpt['base_model'],
#     strict=False
# )

model.load_model_from_ckpt(
    bert_ckpt_path="Point-BERT.pth",)

model.eval()
model.cuda()



# ---------------------------------------------------
# Load your point cloud
# Shape: [N,3]
# ---------------------------------------------------
dataset = HDF5LfDDataset(
        os.path.join(os.getcwd(), "../../dataset")
    )

for i in range(len(dataset)):
    # Preprocess PCs
    pc = dataset[i]["pc"].astype(np.float32)
    pc_ext = dataset[i]["pc_ext"].astype(np.float32)
    pc_front = dataset[i]["pc_front"].astype(np.float32)

    pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

    # ============================================================
    # 2. VOXEL DOWNSAMPLE
    # ============================================================

    voxel_size = 0.025

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc_all[:, 3:])

    pcd = pcd.voxel_down_sample(voxel_size)

    pc_all = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    pc_all = np.concatenate([pc_all, colors], axis=1)


    # ============================================================
    # 3. FILTERING (your original logic cleaned)
    # ============================================================

    pc_all = pc_all[pc_all[:, 2] > 0.025]
    pc_all = pc_all[pc_all[:, 0] > -0.8]
    pc_all = pc_all[pc_all[:, 0] < 0.1]
    pc_all = pc_all[pc_all[:, 1] > -0.5]


    pc_xyz = pc_all[:, :3]
    pc_rgb = pc_all[:, 3:]#  if pc_all.shape[1] > 3 else np.zeros_like(pc_xyz)

   
    sampled_pts = farthest_point_sampling_BERT(
            pc_xyz,
            1024
        )  



    points = torch.tensor(
        pc_xyz,
        dtype=torch.float32
    ).unsqueeze(0).cuda()

    # [1,1024,3]

    with torch.no_grad():

        # forward_features exists in PointTransformer
        __, features, __  = model(points)
        __, features2, __  = model(points)

    print(features.shape)
    raise
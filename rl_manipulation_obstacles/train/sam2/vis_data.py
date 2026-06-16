import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from data import *
from networks_lfd import *
from train_utils import collate_fn
import os
import cv2 as cv
from ultralytics import YOLO
import os
import cv2 as cv

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision import models, transforms

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import hdbscan
from sklearn.preprocessing import StandardScaler


from Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import *


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
def visualize_o3d(pc, title="Point Cloud"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=title
    )

def colored_pcd(points, color):
    # Convert safely
    points = np.asarray(points)

    # Ensure correct shape
    points = points.reshape(-1, 3)

    # Convert dtype
    points = points.astype(np.float64)

    # Make contiguous
    points = np.ascontiguousarray(points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.paint_uniform_color(color)

    return pcd
     
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv.findContours(m.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    return img



# -------------------------
# CONFIG
# -------------------------
MODE = "pc_net"      

YOLO_MODEL = "yolov8n.pt"

SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"


def vis():

    dataset = HDF5LfDDataset(
        os.path.join(os.getcwd(), "../../dataset")
    )

    # -------------------------
    # Load selected model
    # -------------------------
    if MODE == "yolo":
        model = YOLO(YOLO_MODEL)

    elif MODE == "sam":

        sam = sam_model_registry[SAM_TYPE](
            checkpoint=SAM_CHECKPOINT
        )

        sam.to("cuda")

        model = SamAutomaticMaskGenerator(sam)

    elif MODE == "sam2_l":
        
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2 = build_sam2(model_cfg, checkpoint)
        mask_generator = SAM2AutomaticMaskGenerator(
                            model=sam2,
                            )

    elif MODE == "sam_features" or MODE == "sam_pca":

        sam = sam_model_registry[SAM_TYPE](
            checkpoint=SAM_CHECKPOINT
        )

        sam.to("cuda")
        sam.eval()

        model = sam

    elif MODE == "resnet":
        # Load pretrained DeepLabV3 + ResNet50
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        model.eval()

        # Preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    elif MODE == "pc":
        pass

    # -------------------------
    # Main loop
    # -------------------------
    for i in range(len(dataset)):

        if not ("pc" in MODE):
            img = dataset[i]["cam_front_D"]

            # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))

            # float -> uint8
            # if img.dtype != np.uint8:
            #     img = (img * 255).astype(np.uint8)

            img_bgr = img.copy()

            

            if MODE == "yolo":

                results = model(img_bgr, verbose=False)[0]

                annotated = results.plot()

            elif MODE == "sam":

                # img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

                masks = model.generate(img_bgr)
                
                img_tensor = torch.tensor(img_bgr).float().permute(2,0,1).unsqueeze(0).to("cuda")
                
                print(sam.image_encoder(img_tensor))

                annotated = img_bgr.copy()

                for mask_data in masks:

                    mask = mask_data["segmentation"]

                    color = np.random.randint(
                        0, 255, size=(3,), dtype=np.uint8
                    )

                    annotated[mask] = (
                        0.5 * annotated[mask] +
                        0.5 * color
                    ).astype(np.uint8)

            elif MODE == "sam2_l":
                
                annotated = mask_generator.generate(img_bgr.astype(np.float32))
                annotated = show_anns(anns=annotated)[:, :, 1:]

            elif MODE == "sam_features":

                # ---------------------------------
                # preprocess image for SAM
                # ---------------------------------
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

                resized = cv.resize(img_rgb, (1024, 1024))

                tensor = torch.from_numpy(resized).float().cuda()

                # HWC -> BCHW
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
                
                # normalize to SAM expected range
                tensor = tensor / 255.0

                # SAM normalization
                pixel_mean = torch.tensor(
                    [123.675, 116.28, 103.53],
                    device=tensor.device
                ).view(1, 3, 1, 1)

                pixel_std = torch.tensor(
                    [58.395, 57.12, 57.375],
                    device=tensor.device
                ).view(1, 3, 1, 1)

                tensor = tensor * 255.0
                tensor = (tensor - pixel_mean) / pixel_std

                # ---------------------------------
                # image encoder forward
                # ---------------------------------
                with torch.no_grad():
                    embedding = model.image_encoder(tensor)





                    print(embedding.shape)
                    print(embedding.mean(dim=1).mean(-1).shape)






                # embedding shape:
                # [1, 256, 64, 64]

                # ---------------------------------
                # visualize one activation channel
                # ---------------------------------
                feat = embedding[0, 0].detach().cpu().numpy()

                feat = cv.normalize(
                    feat,
                    None,
                    0,
                    255,
                    cv.NORM_MINMAX
                ).astype(np.uint8)

                feat = cv.resize(
                    feat,
                    (img_bgr.shape[1], img_bgr.shape[0])
                )

                feat = cv.applyColorMap(
                    feat,
                    cv.COLORMAP_JET
                )

                annotated = feat

            elif MODE == "sam_pca":

                # ---------------------------------
                # preprocess image
                # ---------------------------------
                img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

                resized = cv.resize(img_rgb, (1024, 1024))

                tensor = torch.from_numpy(resized).float().cuda()

                # HWC -> BCHW
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)

                # normalize
                pixel_mean = torch.tensor(
                    [123.675, 116.28, 103.53],
                    device=tensor.device
                ).view(1, 3, 1, 1)

                pixel_std = torch.tensor(
                    [58.395, 57.12, 57.375],
                    device=tensor.device
                ).view(1, 3, 1, 1)

                tensor = (tensor - pixel_mean) / pixel_std

                # ---------------------------------
                # SAM encoder forward
                # ---------------------------------
                with torch.no_grad():
                    embedding = model.image_encoder(tensor)

                # embedding:
                # [1, 256, 64, 64]

                feat = embedding[0].detach().cpu().numpy()

                # ---------------------------------
                # PCA to RGB
                # ---------------------------------

                C, H, W = feat.shape

                # -> [H*W, C]
                feat_flat = feat.reshape(C, -1).T

                # center
                feat_mean = feat_flat.mean(axis=0)
                feat_flat = feat_flat - feat_mean

                # PCA via SVD
                U, S, Vt = np.linalg.svd(
                    feat_flat,
                    full_matrices=False
                )

                # first 3 principal components
                pca_rgb = feat_flat @ Vt[:3].T
                

                # reshape back
                pca_rgb = pca_rgb.reshape(H, W, 3)

                # normalize each channel
                pca_rgb_min = pca_rgb.min(axis=(0, 1), keepdims=True)
                pca_rgb_max = pca_rgb.max(axis=(0, 1), keepdims=True)

                pca_rgb = (
                    (pca_rgb - pca_rgb_min)
                    / (pca_rgb_max - pca_rgb_min + 1e-8)
                )

                pca_rgb = (255 * pca_rgb).astype(np.uint8)

                # resize to original image size
                pca_rgb = cv.resize(
                    pca_rgb,
                    (img_bgr.shape[1], img_bgr.shape[0]),
                    interpolation=cv.INTER_NEAREST
                )

                annotated = pca_rgb
                
            elif MODE == "resnet":
                # Transform for model
                input_tensor = transform(img_bgr).unsqueeze(0).float()

                with torch.no_grad():
                    output = model(input_tensor)["out"][0]*255

                
                # Get segmentation mask
                annotated = output.argmax(0).unsqueeze(0).repeat(3,1,1).permute(1,2,0).cpu().numpy()*255
                # Optional: scale classes for visualization
                annotated = annotated.astype(np.uint8)

                

            else:
                annotated = img_bgr

            # -------------------------
            # Show
            # -------------------------
            cv.imshow("Visualization", annotated)

            key = cv.waitKey(1)

            if key == 27:  # ESC
                break

        else:

            pc = dataset[i]["pc"].astype(np.float32)
            pc_ext = dataset[i]["pc_ext"].astype(np.float32)
            pc_front = dataset[i]["pc_front"].astype(np.float32)

            pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pc_all[:, 3:])
            pcd = pcd.voxel_down_sample(0.025)

            pc_all = np.asarray(pcd.points)

            # remove ground
            ground_threshold = 0.025  # adjust for your dataset
            pc_all = pc_all[pc_all[:, 2] > ground_threshold]
            # pc_ext = pc_ext[pc_ext[:, 2] > ground_threshold]
            # pc_front = pc_front[pc_front[:, 2] > ground_threshold]

            x_threshold = -0.8  # adjust for your dataset
            pc_all = pc_all[pc_all[:, 0] > x_threshold]
            # pc_ext = pc_ext[pc_ext[:, 0] > x_threshold]
            # pc_front = pc_front[pc_front[:, 0] > x_threshold]

            x_threshold = 0.1  # adjust for your dataset
            pc_all = pc_all[pc_all[:, 0] < x_threshold]
            # pc_ext = pc_ext[pc_ext[:, 0] < x_threshold]
            # pc_front = pc_front[pc_front[:, 0] < x_threshold]

            y_threshold = -0.5  # adjust for your dataset
            pc_all = pc_all[pc_all[:, 1] > y_threshold]
            # pc_ext = pc_ext[pc_ext[:, 1] > y_threshold]
            # pc_front = pc_front[pc_front[:, 1] > y_threshold]




            # ============================================================
            # PREPROCESS
            # ============================================================

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_all)

            # downsample
            pcd = pcd.voxel_down_sample(voxel_size=0.02)

            # remove isolated noise
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )

            points = np.asarray(pcd.points)





            if MODE == "pc_geo":
                

                # ============================================================
                # NORMALS
                # ============================================================

                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.08,
                        max_nn=30
                    )
                )

                normals = np.asarray(pcd.normals)

                # ============================================================
                # LOCAL DENSITY FEATURE
                # ============================================================

                tree = o3d.geometry.KDTreeFlann(pcd)

                density = np.zeros((len(points), 1))

                for i in range(len(points)):
                    _, idx, _ = tree.search_radius_vector_3d(
                        points[i],
                        0.08
                    )
                    density[i] = len(idx)

                density = density / density.max()

                # ============================================================
                # FEATURE VECTOR
                # ============================================================

                features = np.concatenate(
                    [
                        points,      # xyz
                        normals,     # nx ny nz
                        density      # local density
                    ],
                    axis=1
                )

                features = StandardScaler().fit_transform(features)

                # ============================================================
                # HDBSCAN
                # ============================================================

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=80,
                    min_samples=20
                )

                labels = clusterer.fit_predict(features)

                print("Clusters:", len(np.unique(labels[labels >= 0])))

                # ============================================================
                # VISUALIZATION
                # ============================================================

                max_label = labels.max()

                if max_label >= 0:
                    colors = plt.get_cmap("tab20")(
                        labels / max(max_label, 1)
                    )[:, :3]
                else:
                    colors = np.zeros((len(labels), 3))

                # noise -> black
                colors[labels < 0] = [0, 0, 0]

                pcd.colors = o3d.utility.Vector3dVector(colors)

                o3d.visualization.draw_geometries([pcd])


            if MODE == "pc_net":
                pc = dataset[i]["pc"].astype(np.float32)
                pc_ext = dataset[i]["pc_ext"].astype(np.float32)
                pc_front = dataset[i]["pc_front"].astype(np.float32)

                pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

                print("Initial:", pc_all.shape)


                # ============================================================
                # 2. VOXEL DOWNSAMPLE
                # ============================================================

                voxel_size = 0.025

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])

                pcd = pcd.voxel_down_sample(voxel_size)

                pc_all = np.asarray(pcd.points)

                # keep RGB if exists
                if pc.shape[1] > 3:
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pc_all)
                else:
                    colors = np.zeros_like(pc_all)

                pc_all = np.concatenate([pc_all, colors], axis=1)


                # ============================================================
                # 3. FILTERING (your original logic cleaned)
                # ============================================================

                pc_all = pc_all[pc_all[:, 2] > 0.025]
                pc_all = pc_all[pc_all[:, 0] > -0.8]
                pc_all = pc_all[pc_all[:, 0] < 0.1]
                pc_all = pc_all[pc_all[:, 1] > -0.5]


                print("After filtering:", pc_all.shape)


                # ============================================================
                # 4. OPEN3D POINT CLOUD + NORMALS
                # ============================================================

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc_all[:, :3])

                cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.08,
                        max_nn=30
                    )
                )

                normals = np.asarray(cloud.normals)


                # ============================================================
                # 5. POINTNET FEATURES (conv1 output)
                # ============================================================

                pc_xyz = pc_all[:, :3]
                pc_rgb = pc_all[:, 3:] if pc_all.shape[1] > 3 else np.zeros_like(pc_xyz)

                # normalize
                pc_xyz = pc_xyz - np.mean(pc_xyz, axis=0)
                pc_xyz = pc_xyz / (np.max(np.linalg.norm(pc_xyz, axis=1)) + 1e-8)


                # sample for PointNet
                NUM_POINTS = 4096
                idx = np.random.choice(len(pc_xyz), NUM_POINTS, replace=len(pc_xyz) < NUM_POINTS)

                xyz_sample = pc_xyz[idx]
                rgb_sample = pc_rgb[idx]
                normals_sample = normals[idx]


                # ============================================================
                # 6. BUILD POINTNET INPUT
                # ============================================================

                xyz_norm = xyz_sample / (np.max(np.abs(xyz_sample), axis=0, keepdims=True) + 1e-8)

                features_in = np.concatenate(
                    [
                        xyz_sample,      # xyz
                        xyz_norm,        # normalized xyz
                        rgb_sample       # optional color
                    ],
                    axis=1
                )

                points = torch.tensor(features_in.T, dtype=torch.float32).unsqueeze(0).cuda()


                # ============================================================
                # 7. LOAD MODEL
                # ============================================================

                num_classes = 13
                model = get_model(num_classes=num_classes).cuda()

                checkpoint = torch.load(
                    "best_model_sem.pth",
                    map_location="cuda:0",
                    weights_only=False
                )

                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()


                # ============================================================
                # 8. EXTRACT FEATURES (IMPORTANT PART)
                # ============================================================

                with torch.no_grad():
                    seg_pred, _, point_features = model(points)
                    seg_pred = seg_pred.squeeze(0).cpu().numpy()  # (N, num_classes)
                    labels = np.argmax(seg_pred, axis=1)


                colors = plt.cm.tab20(labels / (num_classes - 1))[:, :3]

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(xyz_sample[:, :3])
                cloud.colors = o3d.utility.Vector3dVector(colors)

                o3d.visualization.draw_geometries([cloud])



            elif MODE == "pc_net_geo":
                pc = dataset[i]["pc"].astype(np.float32)
                pc_ext = dataset[i]["pc_ext"].astype(np.float32)
                pc_front = dataset[i]["pc_front"].astype(np.float32)

                pc_all = np.concatenate([pc, pc_ext, pc_front], axis=0)

                print("Initial:", pc_all.shape)


                # ============================================================
                # 2. VOXEL DOWNSAMPLE
                # ============================================================

                voxel_size = 0.025

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc_all[:, :3])

                pcd = pcd.voxel_down_sample(voxel_size)

                pc_all = np.asarray(pcd.points)

                # keep RGB if exists
                if pc.shape[1] > 3:
                    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pc_all)
                else:
                    colors = np.zeros_like(pc_all)

                pc_all = np.concatenate([pc_all, colors], axis=1)


                # ============================================================
                # 3. FILTERING (your original logic cleaned)
                # ============================================================

                pc_all = pc_all[pc_all[:, 2] > 0.025]
                pc_all = pc_all[pc_all[:, 0] > -0.8]
                pc_all = pc_all[pc_all[:, 0] < 0.1]
                pc_all = pc_all[pc_all[:, 1] > -0.5]


                print("After filtering:", pc_all.shape)


                # ============================================================
                # 4. OPEN3D POINT CLOUD + NORMALS
                # ============================================================

                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(pc_all[:, :3])

                cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.08,
                        max_nn=30
                    )
                )

                normals = np.asarray(cloud.normals)


                # ============================================================
                # 5. POINTNET FEATURES (conv1 output)
                # ============================================================

                pc_xyz = pc_all[:, :3]
                pc_rgb = pc_all[:, 3:] if pc_all.shape[1] > 3 else np.zeros_like(pc_xyz)

                # normalize
                pc_xyz = pc_xyz - np.mean(pc_xyz, axis=0)
                pc_xyz = pc_xyz / (np.max(np.linalg.norm(pc_xyz, axis=1)) + 1e-8)


                # sample for PointNet
                NUM_POINTS = 4096
                idx = np.random.choice(len(pc_xyz), NUM_POINTS, replace=len(pc_xyz) < NUM_POINTS)

                xyz_sample = pc_xyz[idx]
                rgb_sample = pc_rgb[idx]
                normals_sample = normals[idx]


                # ============================================================
                # 6. BUILD POINTNET INPUT
                # ============================================================

                xyz_norm = xyz_sample / (np.max(np.abs(xyz_sample), axis=0, keepdims=True) + 1e-8)

                features_in = np.concatenate(
                    [
                        xyz_sample,      # xyz
                        xyz_norm,        # normalized xyz
                        rgb_sample       # optional color
                    ],
                    axis=1
                )

                points = torch.tensor(features_in.T, dtype=torch.float32).unsqueeze(0).cuda()


                # ============================================================
                # 7. LOAD MODEL
                # ============================================================

                num_classes = 13
                model = get_model(num_classes=num_classes).cuda()

                checkpoint = torch.load(
                    "best_model_sem.pth",
                    map_location="cuda:0",
                    weights_only=False
                )

                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()


                # ============================================================
                # 8. EXTRACT FEATURES (IMPORTANT PART)
                # ============================================================

                with torch.no_grad():
                    seg_pred, _, point_features = model(points)

                # (1, 128, N)
                point_features = point_features.squeeze(0).cpu().numpy().T
                xyz = xyz_sample


                # ============================================================
                # 9. BUILD FINAL CLUSTER FEATURES
                # ============================================================

                features_cluster = np.concatenate(
                    [
                        point_features,   # PointNet learned embedding
                        xyz,              # geometry
                        normals_sample    # surface structure
                    ],
                    axis=1
                )

                features_cluster = StandardScaler().fit_transform(features_cluster)


                # ============================================================
                # 10. HDBSCAN CLUSTERING
                # ============================================================

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=80,
                    min_samples=10
                )

                labels = clusterer.fit_predict(features_cluster)

                print("clusters:", len(np.unique(labels[labels >= 0])))


                # ============================================================
                # 11. VISUALIZATION
                # ============================================================

                colors = plt.get_cmap("tab20")(
                    labels / (labels.max() + 1 if labels.max() >= 0 else 1)
                )[:, :3]

                colors[labels < 0] = [0, 0, 0]

                cloud_vis = o3d.geometry.PointCloud()
                cloud_vis.points = o3d.utility.Vector3dVector(xyz)
                cloud_vis.colors = o3d.utility.Vector3dVector(colors)

                o3d.visualization.draw_geometries([cloud_vis])







            # # pcd1 = colored_pcd(pc, [1, 0, 0])       # red
            # # pcd2 = colored_pcd(pc_ext, [0, 1, 0])   # green
            # # pcd3 = colored_pcd(pc_front, [0, 0, 1]) # blue
            # pc_all = colored_pcd(pc_all, [0, 0, 1]) # blue

            # o3d.visualization.draw_geometries([pc_all])



if __name__ == "__main__":
    vis()
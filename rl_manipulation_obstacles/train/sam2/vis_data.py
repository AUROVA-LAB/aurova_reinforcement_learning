import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from rl_manipulation_obstacles.train.sam2.data import *
from networks_lfd import *
from train_utils import collate_fn
import os
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
import cv2 as cv
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision import models, transforms

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator




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
MODE = "sam2_l"      # "yolo" or "sam"

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
        
        checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
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
        

    # -------------------------
    # Main loop
    # -------------------------
    for i in range(len(dataset)):

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

    cv.destroyAllWindows()


if __name__ == "__main__":
    vis()
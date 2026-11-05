import torch



def quat_to_rotmat(q):
    """
    q: (B,4) in (w,x,y,z) order
    """

    # Normalize just in case
    q = q / q.norm(dim=1, keepdim=True)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)

    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)

    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    return R

def transform_points(points, translation, quaternion):
    """
    points:      (B, N, C), first 3 channels are XYZ
    translation: (B, 3)
    quaternion:  (B, 4)  [w, x, y, z]
    """

    # (B, 3, 3)
    R = quat_to_rotmat(quaternion)

    # Rotate
    # points[..., :3] -> (B, N, 3)
    # R.transpose(1, 2) because points are row vectors
    points_world = torch.bmm(points[..., :3], R.transpose(1, 2))

    # Translate
    points_world = points_world + translation[:, None, :]

    return points_world

def depth_to_pointcloud(rgbd, fx, fy, cx, cy):
    """
    depth: (H, W)
    returns: (N, 3) points in camera frame
    """

    depth = rgbd[:, -1]
    rgb = rgbd[:, :-1]

    device = depth.device

    print(depth.shape)
    B, H, W = depth.shape

    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)

    uu, vv = torch.meshgrid(u, v, indexing='xy')

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = torch.stack((x, y, z), dim=-1)

    # Flatten
    points = points.reshape(B, -1, 3)
    colors = rgb.reshape(B, -1, 3)

    # Optional: normalize RGB to [0,1]
    colors = colors.float() / 255.0

    points_rgb = torch.cat([points, colors], dim=-1)

    return points_rgb


def farthest_point_sampling_BERT(xyz, npoint):
    device = xyz.device
    xyz = torch.tensor(xyz).float()#.unsqueeze(0)
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long).to(device)

    batch_indices = torch.arange(B, device=xyz.device).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest

        centroid = xyz[batch_indices, farthest].view(B,1,3)

        dist = ((xyz - centroid)**2).sum(-1)

        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = distance.max(-1)[1]

    return torch.gather(xyz, dim=1, index=centroids.unsqueeze(-1).expand(-1, -1, C))

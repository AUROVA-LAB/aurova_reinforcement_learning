import torch



def depth_to_pointcloud(rgbd, fx, fy, cx, cy):
    """
    depth: (H, W)
    returns: (N, 3) points in camera frame
    """

    depth = rgbd[-1]
    rgb = rgbd[:-1]

    device = depth.device

    H, W = depth.shape

    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)

    uu, vv = torch.meshgrid(u, v, indexing='xy')

    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    points = torch.stack((x, y, z), dim=-1)

    # Flatten
    points = points.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    # Optional: normalize RGB to [0,1]
    colors = colors.float() / 255.0

    points_rgb = torch.cat([points, colors], dim=-1)

    return points_rgb


def quat_to_rotmat(q):
    """
    q = (w, x, y, z)
    """
    w, x, y, z = q

    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], device=q.device)

    return R


def transform_points(points, translation, quaternion):
    """
    points: (N, 3)
    translation: (3,)
    quaternion: (4,)  [w,x,y,z]
    """

    # R = matrix_from_quat(quaternion)



    R = quat_to_rotmat(quaternion)

    points_world = (R @ points[:,:3].T).T + translation
    # points_world = (R.T @ (points + translation).T).T
    # points_world = (points - translation)@R.T  

    points_world = torch.cat((points_world, points[:, 3:]), dim = -1)

    return points_world



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


def preprocess_pointcloud(pc):
    """
    pc: (N, 3) [x,y,z]
    """

    pc = pc[pc[:, 2] > 0.0125]
    pc = pc[pc[:, 0] > -0.8]

    # pc = pc[pc[:, 0] < 0.1]
    # pc = pc[pc[:, 1] > -0.5]

    # Farthest point sampling
    pc = farthest_point_sampling_BERT(pc.unsqueeze(0), 1024).squeeze(0)

    return pc
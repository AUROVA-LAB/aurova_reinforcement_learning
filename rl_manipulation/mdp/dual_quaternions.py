import torch
import math


class DualQuaternion():
    def __init__(self, pose: torch.Tensor, device:str):
        
        assert pose.shape[-1] == 7

        self.__pose = pose
        self.__qr = pose[:, 3:]
        self.__qt = torch.cat((0, pose[:, :3]), dim = -1)

        self.__device = device

        self.__dqr = self.__qr
        self.__dqt = 0.5 * self.__q_mul(q1 = self.__qt, q2 = self.__qr)


    def __q_mul(self, q1: torch.Tensor, q2: torch.Tensor):
        """
        Multiply quaternion q1 with q2.
        Expects two equally-sized tensors of shape [*, 4], where * denotes any number of dimensions.
        Returns q1*q2 as a tensor of shape [*, 4].
        """
        assert q1.shape[-1] == 4
        assert q2.shape[-1] == 4
        original_shape = q1.shape

        # Compute outer product
        terms = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]

        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    def __q_angle(self, q: torch.Tensor):
        """
        Determine the rotation angle of given quaternion tensors of shape [*, 4].
        Return as tensor of shape [*, 1]
        """
        assert q.shape[-1] == 4

        q = self.__q_normalize(q)
        q_re, q_im = torch.split(q, [1, 3], dim=-1)
        norm = torch.linalg.norm(q_im, dim=-1).unsqueeze(dim=-1)
        angle = 2.0 * torch.atan2(norm, q_re)

        return self.__wrap_angle(angle)

    def __q_normalize(self, q: torch.tensor):
        """
        Normalize the coefficients of a given quaternion tensor of shape [*, 4].
        """
        assert q.shape[-1] == 4

        norm = torch.sqrt(torch.sum(torch.square(q), dim=-1))  # ||q|| = sqrt(w²+x²+y²+z²)
        assert not torch.any(torch.isclose(norm, torch.zeros_like(norm, device=q.device)))  # check for singularities
        return  torch.div(q, norm[:, None])  # q_norm = q / ||q||

    def __q_conjugate(self, q: torch.Tensor):
        """
        Returns the complex conjugate of the input quaternion tensor of shape [*, 4].
        """
        assert q.shape[-1] == 4

        conj = torch.tensor([1, -1, -1, -1], device=q.device)  # multiplication coefficients per element
        return q * conj.expand_as(q)

    def __wrap_angle(self, theta: torch.Tensor):
        """
        Helper method: Wrap the angles of the input tensor to lie between -pi and pi.
        Odd multiples of pi are wrapped to +pi (as opposed to -pi).
        """
        pi_tensor = torch.ones_like(theta, device=theta.device) * math.pi
        result = ((theta + pi_tensor) % (2 * pi_tensor)) - pi_tensor
        result[result.eq(-pi_tensor)] = math.pi

        return result
    

    def dq_mul(dq1, dq2):
        """
        Multiply dual quaternion dq1 with dq2.
        Expects two equally-sized tensors of shape [*, 8], where * denotes any number of dimensions.
        Returns dq1*dq2 as a tensor of shape [*, 8].
        """
        assert dq1.shape[-1] == 8
        assert dq2.shape[-1] == 8

        dq1_r, dq1_d = torch.split(dq1, [4, 4], dim=-1)
        dq2_r, dq2_d = torch.split(dq2, [4, 4], dim=-1)

        dq_prod_r = q_mul(dq1_r, dq2_r)
        dq_prod_d = q_mul(dq1_r, dq2_d) + q_mul(dq1_d, dq2_r)
        dq_prod = torch.cat([dq_prod_r, dq_prod_d], dim=-1)

        return dq_prod
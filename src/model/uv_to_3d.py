import torch
import torch.nn as nn

from src.nnutils.geometry import compute_barycentric_coordinates


class UVto3D(nn.Module):

    """
    Module to calculate 3D points from UV values
    """

    def __init__(self, mean_shape, device='cuda:0'):

        """
        :param mean_shape: is a dictionary containing the following parameters
        - uv_map: A R X R tensor of defining the UV steps. Where R is the resolution of the UV map.
        - verts: A (None, 3) tensor of vertex coordinates of the mean shape
        - faces: A (None, 3) tensor of faces of the mean shape
        - face_inds: A R X R tensor where each value is the index of the face for
        the corresponding UV value in uv_map.
        Eg. if uv_map[25, 75] is (0.2, 0.6) then face_inds[25, 75]
        stores the index of the face which corresponds to UV values (0.2, 0.6)
        - uv_verts: A None, 2 tensor of UV values of the vertices of the mean shape
        """

        super(UVto3D, self).__init__()

        self.face_inds = mean_shape['face_inds'].to(device)
        self.verts_uv = mean_shape['uv_verts'].to(device)
        self.verts_3d = mean_shape['verts'].to(device)
        self.faces = mean_shape['faces'].to(device)

        self.uv_res = mean_shape['uv_map'].shape
        self.uv_map_size = torch.tensor(
            [self.uv_res[1] - 1, self.uv_res[0] - 1], dtype=torch.float32).view(1, 2).to(device)

    def forward(self, uv):

        """
        For each UV value
        - Find the closest UV value as per the UV resolution in 'uv_map'
        - Find the face for the rounded off UV value
        - Find the UV values of the vertices for that face
        - Find the barycentric coordinates for the point w.r.t the face
        - Use barycentric coordinates to find the 3D coordinate of the given UV value

        :param uv: [B, None, 2] tensor with UV values (0-1) for which the corresponding 3D points should be calculated
        :return: A [B, None, 3] tensor with the 3D coordinates fot the corresponding UV values
        """

        # Find the closest UV value as per the UV resolution in 'uv_map'
        uv_inds = (self.uv_map_size * uv).round().long().detach()

        # U is along the columns, V is along the rows TODO: Fix this
        # Find the face for the rounded off UV value
        uv_faces = self.faces[self.face_inds[uv_inds[:, 1], uv_inds[:, 0]], :]

        # Find the UV values of the vertices for that face
        face_uv_verts = torch.stack(
            [self.verts_uv[uv_faces[:, 0]],
             self.verts_uv[uv_faces[:, 1]],
             self.verts_uv[uv_faces[:, 2]]], dim=1)

        # Find the barycentric coordinates for the point w.r.t the face
        bary_cord = compute_barycentric_coordinates(face_uv_verts, uv)

        face_verts = torch.stack(
            [self.verts_3d[uv_faces[:, 0]],
             self.verts_3d[uv_faces[:, 1]],
             self.verts_3d[uv_faces[:, 2]]], dim=1)

        # Use barycentric coordinates to find the 3D coordinate of the given UV value
        points3d = face_verts * bary_cord[:, :, None]
        points3d = points3d.sum(1)

        return points3d


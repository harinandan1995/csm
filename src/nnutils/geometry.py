import numpy as np
import scipy.io as sio
import torch


def convert_uv_to_3d(uv):

    """
    Finds the 3D points on a sphere for the corresponding UV values

    :param uv: [..., 2] array of UV values for which 3D points on the sphere should be returned
    :return: [..., 3] 3D points for the given UV values
    """

    phi = 2 * np.pi * (uv[..., 0] - 0.5)
    theta = np.pi * (uv[..., 1] - 0.5)

    if type(uv) == torch.Tensor:
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        points3d = torch.stack([x, y, z], dim=-1)
    else:
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        points3d = np.stack([x, y, z], axis=-1)

    return points3d


def compute_barycentric_coordinates(uv_vertices, uv_points):

    """
    Calculates the barycentric (https://mathworld.wolfram.com/BarycentricCoordinates.html)
    coordinates for the given points w.r.t the vertices

    :param uv_vertices: [None, 3, 2] tensor with the UV values of the vertices
    of the face corresponding the point
    :param uv_points: [None, 2] tensor with UV values of the points for
    which the barycentric coordinates are to be calculated
    :return: [None, 3] tensor with the barycentric coordinates for the given points
    """

    vertices = convert_uv_to_3d(uv_vertices)
    points = convert_uv_to_3d(uv_points)

    vertA = vertices[:, 0, :]
    vertB = vertices[:, 1, :]
    vertC = vertices[:, 2, :]

    AB = vertB - vertA
    AC = vertC - vertA
    BC = vertC - vertB

    AP = points - vertA
    BP = points - vertB
    CP = points - vertC

    areaBAC = torch.norm(torch.cross(AB, AC, dim=1), dim=1)
    areaBAP = torch.norm(torch.cross(AB, AP, dim=1), dim=1)
    areaCAP = torch.norm(torch.cross(AC, AP, dim=1), dim=1)
    areaCBP = torch.norm(torch.cross(BC, BP, dim=1), dim=1)

    w = areaBAP / areaBAC
    v = areaCAP / areaBAC
    u = areaCBP / areaBAC

    barycentric_coordinates = torch.stack([u, v, w], dim=1)
    barycentric_coordinates = torch.nn.functional.normalize(barycentric_coordinates, p=1)

    return barycentric_coordinates


def get_gt_positions_grid(img_size):

    """
    Generates a positions grid W X H X 2 which contains the indices in the grid

    :param img_size: A tuple (W, H)
    :return: The ground truth position grid
    """

    x = torch.linspace(-1, 1, img_size[1]).view(1, -1).repeat(img_size[0], 1)
    y = torch.linspace(-1, 1, img_size[0]).view(-1, 1).repeat(1, img_size[1])
    grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
    grid.unsqueeze(0)

    return grid


def load_mean_shape(mean_shape_path):

    if type(mean_shape_path) == str:
        mean_shape = sio.loadmat(mean_shape_path)
    else:
        mean_shape = mean_shape_path

    # mean_shape['bary_cord'] = torch.from_numpy(mean_shape['bary_cord']).float()
    mean_shape['uv_map'] = torch.from_numpy(mean_shape['uv_map']).float()
    mean_shape['uv_verts'] = torch.from_numpy(mean_shape['uv_verts']).float()
    mean_shape['verts'] = torch.from_numpy(mean_shape['verts']).float()
    mean_shape['sphere_verts'] = torch.from_numpy(mean_shape['sphere_verts']).float()
    mean_shape['face_inds'] = torch.from_numpy(mean_shape['face_inds']).long()
    mean_shape['faces'] = torch.from_numpy(mean_shape['faces']).long()

    return mean_shape

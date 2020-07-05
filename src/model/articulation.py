import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exponential_map


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.children = []


# TODO: test axis prediction if angle prediction is done.
class ArticulationPredictor(nn.Module):
    """
    Articulation predictor.
    It predicts angle (in axis-angle representation) and translation for articulation.
    """

    def __init__(self, num_parts, num_feats=512, num_rots=2, num_trans=3, device='cuda', axis_move=False):
        """
        :param num_parts: the number of part of object
        :param num_feats: the number of feats extracted from images
        :param num_rots: the number of parameters used for angle prediction
        :param num_trans: the number of parameters used for translation prediction
        :param device:
        :param axis_move: flag indicates whether predict translation.
        """
        super(ArticulationPredictor, self).__init__()
        self._num_parts = num_parts
        self._num_feats = num_feats
        self._num_rots = num_rots
        self._num_trans = num_trans
        self.fc = nn.Sequential(
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(),
            nn.Linear(num_feats, num_feats),
            nn.LeakyReLU(),
            nn.Linear(num_feats, num_parts * (num_rots + num_trans))
        )
        self.axis = nn.Parameter(torch.FloatTensor(
            [1, 0, 0]).unsqueeze(0).repeat(num_parts, 1).to(device))
        if not axis_move:
            self.axis.requires_grad = False

    def forward(self, x):
        """
        :param x: input features
        :return: A tuple (R, t)
            - R: [N x K x 3 x 3] Rotation matrix for part transformation.
            - t: [N x K x 3 x 1] Translation for part transformation.
            N = batch size, K = number of part
        """

        batch_size = len(x)
        vec = self.fc.forward(x)

        # convert NXC tensor to a NXPX3 vector
        vec = vec.view(-1, self._num_parts, (self._num_rots + self._num_trans))
        vec_rot = vec[..., 0:(self._num_rots+1)]
        vec_tran = vec[..., self._num_rots:].unsqueeze(-1)
        vec_rot = F.normalize(vec_rot, dim=-1)
        angle = torch.atan2(
            vec_rot[..., 1], vec_rot[..., 0]).unsqueeze(-1).repeat(1, 1, 3)
        self.axis.data = F.normalize(self.axis, dim=-1).data
        axis = self.axis.unsqueeze(0).repeat(batch_size, 1, 1)

        axis = axis.view(-1, 3)
        angle = angle.view(-1, 3)

        R = so3_exponential_map(angle * axis)
        R = R.view(batch_size, self._num_parts, 3, 3)
        return R, vec_tran


class Articulation(nn.Module):
    """
    Articulation predictor for part transforamtion prediction.
    It predicts part transforamtion in the form of vertcies coordinate based on an image.
    """

    def __init__(self,  template_mesh: Meshes, parts: list, num_parts: int, rotation_center: dict,
                 parent: dict, alpha=None, num_feats=512, num_rots=2, num_trans=3,  encoder=None, device='cuda'):
        """
        :param mesh: The base mesh for the certain category.
        :param parts: K element list. Each element in the list represent the part of each vertice in mesh (range:[0, num_parts - 1])
        :param num_parts: The number of parts in mesh.
        :param rotation_center: The rotation center of each part. e.g. rotation_center[index0] -> XYZ coordinate of index0
        :param parent: A dictionary describe the child-parent relationship between parts. e.g. parent[index0] -> index_for_parent_of_index0
        :param alpha: [K x P] The weight of each vertices for each parts
        :param num_feats:The number of extracted features from the encoder.
        :param num_rots: The number of parameter used in rotation prediction.
        :param num_trans: The number of parameter used in translation prediction.
        :param device:
            K = the number of mesh vertice, P = the number of parts
        """

        super(Articulation, self).__init__()
        self.device = device

        self._num_feats = num_feats
        self._num_rots = num_rots
        self._num_parts = num_parts
        self._num_trans = num_trans
        self._predictor = ArticulationPredictor(
            num_parts, num_feats, num_rots, num_trans, device)

        self._mesh = template_mesh
        self._v_to_p = parts  # list: vertice -> part
        # dict: part -> [vertices],  # dict: part -> num
        self._p_to_v, self._p_n = get_p_to_v(parts, num_parts)
        self._r_c = get_r_c(rotation_center, num_parts).to(device)
        self._r_c.requires_grad = False

        self._parent = parent
        # p: [p, parent, grand_parent], no use now
        self._relationship = get_relationship(parent)
        # tree, [p_1, p_2, ...] if x < y, p_x could not be parent of p_y
        self._tree, self._order_list, self._order_list_id = get_tree_order(parent)

        self._alpha = None
        if alpha is not None:
            self._alpha = torch.FloatTensor(alpha).to(self.device)

    def forward(self, x):
        """
        Predicts the articulation to an input features.
        :param x: [N x C x H x W]  The input images, for which the articulation should be predicted.
            N - batch size
            C - the number of channel
            [H, W] - height and weight for images
            K - the number of mesh vertice
            P - the  number of parts
        :return: A tuple (verts, loss)
            - verts:[N x K x 3] The corrdinate of vertices for articulation prediction.
            - t_net:[N x P x 3] The corresponding loss for translation.
        """

        #x = self._encoder(x)
        x = x.view(-1, self._num_feats)
        batch_size = len(x)

        R, t = self._predictor(x)

        # TODO: Add translation for future
        if True:
            t = torch.zeros(
                [batch_size, self._num_parts, 3, 1]).to(self.device)

        t_net = t

        rotation_center = self._r_c.unsqueeze(
            0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)

        t_part = []
        t_global = []
        R_global = []
        for k in self._order_list:
            t_part += [- torch.bmm(R[:, k, ...], rotation_center[:, k, ...]) + rotation_center[:, k, ...] + t[:, k, ...]]

            if self._parent[k] != -1:
                R_global += [torch.bmm(R_global[self._order_list_id[self._parent[k]]],R[:,k,...])]
                t_global += [torch.bmm(R_global[self._order_list_id[self._parent[k]]],t_part[self._order_list_id[k]]) + t_global[self._order_list_id[self._parent[k]]]]
            else:
                t_global += [t_part[self._order_list_id[k]]]
                R_global += [R[:, k, ...]]

        R_global = torch.stack(R_global, dim=1)
        t_global = torch.stack(t_global, dim=1)
        R_global = R_global[:,self._order_list_id, ...]
        t_global = t_global[:,self._order_list_id, ...]


        verts = self._mesh.verts_list()[0].clone()
        verts = verts.unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1) #[N x K x 3 x 1]

        if self._alpha is None:

            alpha = torch.zeros([len(verts), self._num_parts], requires_grad = False).to(self.device)
            for k in self._order_list:
                ele_k = self._p_to_v[k]
                alpha[ele_k,k] = 1
        else:

            alpha = self._alpha

        alpha = alpha.unsqueeze(
            0).unsqueeze(-1).repeat(batch_size, 1, 1, 3)  # [N x K x P x 3]

        non_soften_verts = []
        for k in self._order_list:
            non_soften_verts += [(torch.matmul(
                R_global[:, [k], ...], verts) + t_global[:, [k], ...]).squeeze(-1)]
        non_soften_verts = torch.stack(non_soften_verts, dim=-2)

        non_soften_verts = non_soften_verts[..., self._order_list_id, :]
        arti_verts = (alpha * non_soften_verts).sum(-2)

        arti_verts = arti_verts.squeeze(-1)
        t_net = t_net.squeeze(-1)
        #loss = t_net.pow(2).sum(-1).view(-1, self._num_parts).sum(-1)

        return arti_verts, t_net


class MultiArticulation(nn.Module):
    """Module for predicting a set of articulation pose."""

    def __init__(self, num_hypotheses=8, **kwargs):
        """
        :param num_hypotheses: number of articulation pose which should be predicted.
        :param kwargs: arguments which are passed through to the single articulation  predictors.
        """
        super(MultiArticulation, self).__init__()
        self.num_hypotheses = num_hypotheses

        self.arti = nn.ModuleList(
            [Articulation(**kwargs) for _ in range(num_hypotheses)])

    def forward(self, x, use_gt_cam=True, use_sampled_cam=True, index_list=[]):
        """Predict a certain number of articulation.
        ::param x: [N x F]  The input images, for which the articulation should be predicted.
            N - batch size
            F - the number of feature
            H - umber of articulation pose which should be predicted.
            K - the number of mesh vertice
        :param index_list: [N] list of the index indicates which articulation is used
        :return: A tuple (verts, loss)
            - pred_verts:[N x 1 x K x 3] or [N x H x K x 3] he corrdinate of vertices for articulation prediction.for i-th camera
            - pred_t:[N x 1 x P x 3]  or [N x H x P x 3]The corresponding loss for translation.for i-th camera

        """

        if use_gt_cam:
            assert self.num_hypotheses == 1
            pred_verts, pred_t = self.arti[0](x)
            pred_verts = pred_verts.unsqueeze(1)
            pred_t = pred_t.unsqueeze(1)

        elif use_sampled_cam:
            pred = [self.arti[index](x[[num], ...])
                    for num, index in enumerate(index_list)]
            pred_verts, pred_t = zip(*pred)
            pred_verts = list(pred_verts)
            pred_t = list(pred_t)

            pred_verts = torch.stack(pred_verts, dim = 0)
            pred_t = torch.stack(pred_t, dim = 0)

            #print(pred_verts.size())
            #print(pred_t.size())

        else:
            pred = [ar(x) for ar in self.arti]
            pred_verts, pred_t = zip(*pred)
            pred_verts = list(pred_verts)
            pred_t = list(pred_t)
            pred_verts = torch.stack(pred_verts, dim=1)
            pred_t = torch.stack(pred_t, dim=1)

            #print(pred_verts.size())
            #print(pred_t.size())
        return pred_verts, pred_t


def get_p_to_v(parts: list, num_parts: int):
    parts_info = {}
    parts_stat = {}
    for i in range(num_parts):
        parts_info[i] = []
        parts_stat = 0

    for j, ele in enumerate(parts):
        parts_info[ele] += [j]
        parts_stat += 1

    return parts_info, parts_stat


def get_relationship(relationship: dict):
    rela = {}
    for i in range(len(relationship)):
        rela[i] = [i]
        j = relationship[i]
        while j != -1:
            rela[i] = [j] + rela[i]
            j = relationship[j]
    return rela


def get_tree_order(relationship: dict):
    tree_list = []
    order = []
    root = None
    p = None
    num_max = 1
    for i in range(len(relationship)):
        if relationship[i] == -1:
            root = TreeNode(i)
            tree_list.append(root)
            order.append(i)

    while len(tree_list) != 0:
        for i in range(len(relationship)):
            if relationship[i] == tree_list[0].val:
                p = TreeNode(i)
                tree_list[0].children.append(p)
                tree_list.append(p)
                order.append(i)
        del tree_list[0]

    order_id = [ 0 for _ in order]
    for loc, ele in enumerate(order):
        order_id[ele] = loc
    return root, order, order_id


def get_r_c(rotation_center, num_parts):
    r_c_list = []
    assert len(rotation_center) == num_parts
    for k in range(num_parts):
        assert k in rotation_center.keys()
        r_c_list += [rotation_center[k]]

    return torch.FloatTensor(r_c_list)


# def axang2quat(angle, axis):

#    cangle = torch.cos(angle/2)
#    sangle = torch.sin(angle/2)
#    qw = cangle
#    qx =  axis[...,None,0]*sangle
#    qy = axis[...,None,1]*sangle
#    qz = axis[...,None,2]*sangle
#    quat = torch.cat([qw, qx, qy, qz], dim=-1)
#    return quat

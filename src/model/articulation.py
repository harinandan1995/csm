from pytorch3d.structures import Meshes
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (so3_exponential_map, euler_angles_to_matrix,
                  matrix_to_euler_angles, matrix_to_quaternion,
                  quaternion_multiply, quaternion_to_matrix)


class TreeNode:
  def __init__(self, x):
    self.val = x
    self.children = []


#TODO: test axis prediction if angle prediction is done.
class ArticulationPredictor(nn.Module):
    """
    Articulation predictor.
    It predicts angle (in axis-angle representation) and translation for articulation.
    """
    def __init__(self, num_parts, num_feats=512, num_rots = 2, num_trans = 3, device='cuda', axis_move = False):
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
        self.axis = nn.Parameter(torch.FloatTensor([1,0,0]).unsqueeze(0).repeat(num_parts, 1).to(device))
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
        vec_rot = vec[... ,0:(self._num_rots+1)]
        vec_tran = vec[... , self._num_rots:].unsqueeze(-1)
        vec_rot = F.normalize(vec_rot, dim = 2)
        angle = torch.atan2(vec_rot[... ,1], vec_rot[... ,0]).unsqueeze(-1).repeat(1,1,3)
        self.axis.data = F.normalize(self.axis.unsqueeze(0), dim = -1).squeeze(0).data
        axis = self.axis.unsqueeze(0).repeat(batch_size, 1, 1)

        axis = axis.view(-1,3)
        angle = angle.view(-1,3)

        R = so3_exponential_map(angle * axis)
        R = R.view(batch_size,self._num_parts,3,3)
        return R, vec_tran

class Articulation(nn.Module):
    """
    Articulation predictor for part transforamtion prediction. 
    It predicts part transforamtion in the form of vertcies coordinate based on an image.
    """

    def __init__(self,  template_mesh: Meshes, parts: list, num_parts: int, rotation_center: dict,
            parent: dict, alpha=None, num_feats=512, num_rots=2, num_trans = 3,  encoder=None, device='cuda'):
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

        if not encoder:
            self._encoder = get_encoder(trainable=False)
        else:
            self._encoder = encoder

        self._num_feats = num_feats
        self._num_rots = num_rots
        self._num_parts = num_parts
        self._num_trans = num_trans
        self._predictor = ArticulationPredictor(num_parts, num_feats, num_rots, num_trans, device)

        self._mesh = template_mesh
        self._v_to_p = parts # list: vertice -> part
        # dict: part -> [vertices],  # dict: part -> num   
        self._p_to_v, self._p_n= get_p_to_v(parts, num_parts)    
        self._r_c = get_r_c(rotation_center, num_parts).to(device)
        self._r_c.requires_grad = False

        self._parent = parent
        self._relationship = get_relationship(parent) # p: [p, parent, grand_parent], no use now
        #tree, [p_1, p_2, ...] if x < y, p_x could not be parent of p_y
        self._tree, self._order_list = get_tree_order(parent)

        self._alpha = None
        if alpha:
            self._alpha = torch.FloatTensor(alpha)

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
            - t_net:[N x ] The corresponding loss for translation.
        """

        x = self._encoder(x)
        x = x.view(-1, self._num_feats)
        batch_size = len(x)

        R, t = self._predictor(x)

        #TODO: Add translation for future
        if True:
            t = torch.zeros([batch_size, self._num_parts, 3, 1]).to(self.device)
        
        t_net = t

        rotation_center = self._r_c.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)
        for k in self._order_list:
            t[:,k,...] += -torch.bmm(R[:,k,...],rotation_center[:,k,...]) + rotation_center[:,k,...]
            if self._parent[k] != -1:
                R[:,k,...] = torch.bmm(R[:,self._parent[k],...],R[:,k,...])
                t[:,k,...] = torch.bmm(R[:,self._parent[k],...],t[:,k,...]) + t[:,self._parent[k],...]


        verts = self._mesh.verts_list()[0]
        verts = verts.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)
        arti_verts = torch.zeros_like(verts, requires_grad = True)

        if self._alpha:
            alpha = self._alpha.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,3) #[N x K x P x 3]
            alpha = alpha.transpose(2,3)                                            #[N x K x 3 x P]
            non_soften_verts = torch.zeros_like(verts, requires_grad = True)
            non_soften_verts = non_soften_verts.repeat(1,1,1,1,self._num_parts) #[N x K x 3 x P]
            for k in self._order_list:
                non_soften_verts[..., k] =  (torch.matmul(R[:,[k],...], verts) + t[:,[k],...]).squeeze(-1)

            arti_verts = (alpha * non_soften_verts).sum(-1)

        else:
            for k in self._order_list:
                ele_k = self._p_to_v[k]
                arti_verts[:,ele_k,...] =  torch.matmul(R[:,[k],...], verts[:,ele_k,...]) + t[:,[k],...]

        arti_verts = arti_verts.squeeze(-1)
        t_net = t_net.squeeze(-1)
        #loss = t_net.pow(2).sum(-1).view(-1, self._num_parts).sum(-1)

        return arti_verts, t_net


class MultiArticulation(nn.Module):
    """Module for predicting a set of articulation pose."""

    def __init__(self, num_hypotheses=8, encoder = None, **kwargs):
        """
        :param num_hypotheses: number of articulation pose which should be predicted.
        :param encoder: a list of encoder used in prediction
        :param kwargs: arguments which are passed through to the single articulation  predictors.
        """
        super(MultiArticulation, self).__init__()
        self.num_hypotheses = num_hypotheses

        if encoder:
            self.arti = nn.ModuleList(
                [Articulation(**kwargs) for _ in range(num_hypotheses)])
        else:
            assert len(encoder) == num_hypotheses
            self.arti = nn.ModuleList(
                [Articulation(encoder[i], **kwargs) for i in range(num_hypotheses)])

    def foward(self, x, index):
        """Predict a certain number of articulation. 
        ::param x: [N x C x H x W]  The input images, for which the articulation should be predicted.
            N - batch size
            C - the number of channel
            [H, W] - height and weight for images
            K - the number of mesh vertice
        :param id: the index indicates which articulation is used
        :return: A tuple (verts, loss)
            - verts:[N x K x 3] The corrdinate of vertices for articulation prediction.for i-th camera
            - t_net:[N x ] The corresponding loss for translation.for i-th camera

        """

        #pred =  [ arti_forward(x) for arti_forward in self.arti ]
        #pred_verts, pred_loss = zip(*pred)
        #pred_verts = list(pred_verts)
        #pred_loss = list(pred_loss)
        #pred_verts = torch.stack(pred_verts, dim = -1)
        #pred_loss = torch.stack(pred_loss, dim = -1)
        #return pred_verts, pred_loss

        return self.arti[index](x)


def get_p_to_v( parts : list, num_parts : int):
    parts_info = {}
    parts_stat = {}
    for i in range(num_parts):
        parts_info[i] = []
        parts_stat = 0

    for j, ele in enumerate(parts):
        parts_info[ele] += [j]
        parts_stat += 1

    return parts_info, parts_stat

def get_relationship( relationship : dict ):
    rela = {}
    for i in range(len(relationship)):
      rela[i] = [i]
      j = relationship[i]
      while j != -1:
        rela[i] = [j] + rela[i]
        j = relationship[j]
    return rela
     

def get_tree_order( relationship : dict):
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
    return root, order

def get_r_c(rotation_center, num_parts):
    r_c_list = []
    assert len(rotation_center) == num_parts
    for k in range(num_parts):
        assert k in rotation_center.keys()
        r_c_list += [rotation_center[k]]

    return torch.FloatTensor(r_c_list)

def get_encoder(trainable=False):
    """
    Loads resnet18 and extracts the pre-trained convolutional layers for feature extraction.
    Pre-trained layers are frozen.
    :param trainable: bool. whether to train the resnet layers  
    :return: Feature extractor from resnet18
    """

    resnet = torch.hub.load(
        'pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    encoder = nn.Sequential(*([*resnet.children()][:-1]))
    if not trainable:
        for param in encoder.parameters():
            param.requires_grad = True
    return encoder


#def axang2quat(angle, axis):
#    cangle = torch.cos(angle/2)
#    sangle = torch.sin(angle/2)
#    qw = cangle
#    qx =  axis[...,None,0]*sangle
#    qy = axis[...,None,1]*sangle
#    qz = axis[...,None,2]*sangle
#    quat = torch.cat([qw, qx, qy, qz], dim=-1)
#    return quat
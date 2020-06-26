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
class Predictor(nn.Module):
    def __init__(self, num_parts, num_feats=512, num_rots=2, num_trans = 3, device='cuda', axis_move = False):
        super(QuatPredictorSingleAxis, self).__init__()
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
        
        batch_size = len(x)
        vec = self.fc.forward(x)

        # convert NXC tensor to a NXPX3 vector
        vec = vec.view(-1, self._num_parts, (self._num_rots + self._num_trans))
        vec_rot = vec[... ,0:(self._num_rots+1)]
        vec_tran = vec[... , (self._num_rots):].unsqueeze(-1)
        vec = F.normalize(vec, dim = 2)
        angle = atan2(vec[... ,1], vec[... ,0]).unsqueeze(-1).repeat(1,1,3)
        self.axis.data = F.normalize(self.axis.unsqueeze(0),dim = -1).squeeze(0).data
        axis = self.axis.unsqueeze(0).repeat(batch_size, 1, 1)

        axis = axis.view(-1,3)
        angle = angle.view(-1,3)

        R = so3_exponential_map(angle * axis)
        R = R.view(batch_size,num_parts,3,3)
        return R, vec_tran

class Articulation(nn.Module):
    """
    Articulation predictor for part transforamtion prediction. 
    It predicts part transforamtion in the form of vertcies coordinate based on an image.
    """

    def __init__(self, mesh : Meshes, parts: list, num_parts: int, rotation_center: dict, 
            parent: dict, alpha: None, num_feats=512, num_rots=2, num_trans = 3, device='cuda'):
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
            K = number of mesh vertice, P = number of parts
        """

        super(Articulation, self).__init__()
        self.device = device
        self._num_feats = num_feats
        self._num_rots = num_rots
        self._num_parts = num_parts
        self._num_trans = num_trans
        self._predictor = Predictor(num_parts, num_feats, num_rots, num_trans, device)


        self._mesh = mesh       
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

    def forward(self, feats):
        """
        Predicts the articulation to an input features.
        :param feats: [N x M] The input features, for which the articulation should be predicted.
            N - batch size
            M - number of feature 
            K - number of mesh vertice
        :return: A tuple (verts, loss)
            - verts:[N x K x 3] The corrdinate of vertices for articulation prediction.
            - loss:[N] The corresponding loss for translation.
        """

        feats = feats.view(-1, self._num_feats)
        batch_size = len(x)

        R, t = self._predictor(x)

        #TODO: Add translation for future
        if True:
            t = tensor.zeros([batch_size, self._num_parts, 3, 1]).to(device)       
        
        t_old = t

        rotation_center = self._r_c.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1,1,1)
        for k in _order_list:
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
        t_old = t_old.squeeze(-1)
        loss = t_old.pow(2).sum(-1).view(-1, self._num_parts).sum(-1)

        return arti_verts, loss

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
        
    def foward(self, feats):
        """Predict a certain number of articulation. 
        ::param feats: [N x M] The input features, for which the articulation should be predicted.
            N - batch size
            M - number of feature 
            K - number of mesh vertice
        :return: A tuple (verts, loss)
            - verts:[N x K x 3 x 8] All possible corrdinate of vertices for articulation prediction.
            - loss:[N x 8] The corresponding loss for translation.

        """

        pred = [cpp(x) for cpp in self.arti ]
        pred_verts, pred_loss = zip(*pred)
        pred_verts = list(pred_verts)
        pred_loss = list(pred_loss)
        pred_verts = torch.stack(pred_verts, dim = -1)
        pred_loss = torch.stack(pred_loss, dim = -1)

        return pred_verts, pred_loss



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

    return torch.Tensor(r_c_list)



#def axang2quat(angle, axis):
#    cangle = torch.cos(angle/2)
#    sangle = torch.sin(angle/2)
#    qw = cangle
#    qx =  axis[...,None,0]*sangle
#    qy = axis[...,None,1]*sangle
#    qz = axis[...,None,2]*sangle
#    quat = torch.cat([qw, qx, qy, qz], dim=-1)
#    return quat
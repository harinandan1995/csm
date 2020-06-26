import trimesh
import xml.etree.ElementTree as ET
import os.path as osp
import numpy as np
from scipy.sparse import csr_matrix, csgraph
import matplotlib.pyplot as plt

def Part(fids, vids, name):
    '''

    :param fids: list, index of faces of this part
    :param vids: list, index of vertices of this part
    :param name: string, name of this part
    :return:
    pm: a Part dict
    '''
    pm = {}
    pm['fids'] = fids
    pm['vids'] = vids
    pm['name'] = name
    return pm

def PartModel(id, fids, vids, children, is_main=False, name='NA'):
    '''

    :param id: int
    :param fids: list, index of faces of this part
    :param vids: list, index of vertices of this part
    :param name: string, name of this part
    :param children: list, children node of this part
    :param is_main: bool, whether is the body
    :return:
    pm: a Part dict
    '''
    pm = {}
    pm['id'] = id
    pm['fids'] = fids
    pm['vids'] = vids
    pm['children'] = children
    pm['is_main'] = is_main
    pm['name'] = name
    return pm

def complete_dpm(dpm):
    '''

    :param dpm: a dict contain info of all nodes
    :return:
    '''
    nodes = dpm['nodes']
    verts = dpm['verts']
    for pnode in dpm['nodes']:
        if pnode['children'] is not None:
            for cid in pnode['children']:
                cnode = nodes[cid]
                cnode['rc'], cnode['boundary'] = compute_rc(verts, pnode, cnode)

    blend_alpha = compute_blend_alpha_m2(dpm['verts'], dpm['nodes'], dpm['mesh_dist'])
    dpm['alpha'] = blend_alpha
    return

def euclidean_distance(v1, v2):
    '''

    :param v1: an array 1*3
    :param v2: an array 1*3
    :return: euclidean_distance between v1 and v2
    '''
    return np.sqrt(np.sum((v1 - v2)**2))


def convert_mesh_to_graph(verts, faces):
    '''

    :param verts: a V * 3 array
    :param faces: a F * 3 array
    :return: mesh_distances, a V * V array, min vertices path distance
    '''
    row = []
    column = []
    distances = []
    for face in faces:
        for vx in range(3):
            row.append(face[vx])
            column.append(face[(vx+1) % 3])
            distances.append(euclidean_distance(verts[face[vx]], verts[face[(vx+1) % 3]]))

    dist_adj_mat = csr_matrix((np.array(distances), (np.array(row), np.array(column))))
    mesh_distances = csgraph.dijkstra(csgraph=dist_adj_mat, directed=False)
    return mesh_distances


def compute_blend_alpha_m2(verts, nodes, mesh_distances):
    '''

    :param verts: array, V * 3
    :param nodes:
    :param mesh_distances:
    :return:
    '''
    nparts = len(nodes)
    blend_alpha = np.zeros((len(verts), nparts))
    for node in nodes:
        node_id = node['id']
        node_vids = node['vids']
        vertex2part_dist = np.min(mesh_distances[node_vids,:], axis=0)
        blend_alpha[:, node_id] = np.exp(-10*vertex2part_dist)

    blend_alpha = blend_alpha/(1E-12 + blend_alpha.sum(1).reshape(-1,1))
    return blend_alpha

def compute_blend_alpha(verts, parent, child):
    p_vids = parent['vids']
    c_vids = child['vids']

    common_vids = set(p_vids).intersection(set(c_vids))
    common_vids = list(common_vids)
    p_vids = np.array(p_vids)
    c_vids = np.array(c_vids)
    common_vids = np.array(common_vids)

    alpha = np.zeros([len(verts), 2])
    boundary_verts = verts[common_vids]

    alpha[np.concatenate([p_vids, c_vids]),1] = 1
    alpha[np.concatenate([p_vids, c_vids]),0] = 1

    for pvid in p_vids:
        vert = verts[pvid]
        dist = (boundary_verts - vert.reshape(1,-1))**2
        dist = np.sqrt(dist.sum(1)).min()
        alpha[pvid,1] = alpha[pvid,1]*np.exp(-(1*dist))

    for cvid in c_vids:
        vert = verts[cvid]
        dist = (boundary_verts - vert.reshape(1,-1))**2
        dist = np.sqrt(dist.sum(1)).min()
        alpha[cvid,0] = alpha[cvid,0]*np.exp(-(1*dist))
    alpha = alpha/(1E-12 + alpha.sum(1).reshape(-1,1))
    return alpha


def compute_rc(verts, parent, child):
    p_vids = parent['vids']
    c_vids = child['vids']
    common_vids = set(p_vids).intersection(set(c_vids))
    common_verts = verts[np.array(list(common_vids))]
    return common_verts.mean(axis=0), common_verts


def get_parts_from_components(mesh_one, components, names):
    '''

    :param mesh_one: original mesh with V * 3 vertices and F * 3 faces
    :param components: list of parts mesh
    :param names: list of parts name
    :return: a dict contains face_index, vertex_index, name of each part
    '''
    def create_componenet(mesh_verts, mesh_faces, component, name):
        '''

        :param mesh_verts: a V * 3 array
        :param mesh_faces: a F * 3 array
        :param component: part mesh
        :param name: part name
        :return: a dict contains face_index, vertex_index, name of one part
        '''
        comp_verts = component.vertices
        comp_faces = component.faces
        dist = (comp_verts[:,None,:] - mesh_verts[None,:,:])**2
        dist = dist.sum(2)
        orig_vert_ids = dist.argmin(1)

        new_faces = []
        for face in comp_faces:
            new_faces.append([orig_vert_ids[face[0]], orig_vert_ids[face[1]], orig_vert_ids[face[2]]])
        new_faces = np.array(new_faces)

        dist= (new_faces[:,None,:] - mesh_faces[None,:,:])**2
        dist = dist.sum(2)
        orig_face_ids = dist.argmin(1)

        part = Part(orig_face_ids, orig_vert_ids, name)
        return part

    mesh_verts = mesh_one.vertices
    mesh_faces = mesh_one.faces
    parts = []
    for cx, comp in enumerate(components):
        pm = create_componenet(mesh_verts, mesh_faces, comp, names[cx])
        parts.append(pm)

    name2part = {}
    for p in parts:
        name2part[p['name']] = p

    return name2part


def create_dpm(allNodes, xmlNode, parts, node_count):
    tagName = xmlNode.tag
    part  = parts[tagName]

    id = node_count
    node_count+=1
    childIds = []
    for child in xmlNode.getchildren():
        childId, node_count = create_dpm(allNodes, child, parts, node_count)
        childIds.append(childId)

    node = PartModel(id, fids=part['fids'], vids=part['vids'], children=childIds, is_main=(id==0), name=tagName)
    allNodes[id] = node
    return id, node_count



def save_obj_with_vertex_color(name, results_path, mean_shape):
    verts  = np.round(mean_shape['verts'],5)
    faces = mean_shape['faces']
    vert_colors = np.round(mean_shape['vert_colors'],3)
    obj_file = osp.join(results_path, '{}.obj'.format(name))
    with open(obj_file,'w') as f:
        for vx, v in enumerate(verts):
            if vert_colors is not None:
                f.write("v {} {} {} {} {} {}\n".format(v[0], v[1], v[2], vert_colors[vx][0], vert_colors[vx][1], vert_colors[vx][2]))
            else:
                f.write("v {} {} {} \n".format(v[0], v[1], v[2]))

        faces = faces + 1
        for fv in faces:
            f.write('f {}/{}/ {}/{}/ {}/{}/ \n'.format(fv[0], fv[0], fv[1], fv[1], fv[2], fv[2]))
    return obj_file

def dump_blend_color_maps(model_dir, nodes, blend_alpha, verts, faces, tone='Reds'):
    cm = plt.get_cmap(tone)
    mean_shape = {}
    mean_shape['verts'] = verts
    mean_shape['faces'] = faces
    for nx in range(blend_alpha.shape[1]):
        name = nodes[nx]['name']
        vert_colors = cm(blend_alpha[:,nx])
        mean_shape['vert_colors'] = vert_colors
        save_obj_with_vertex_color('mem_{}_{}'.format(name, tone), model_dir,
                                       mean_shape)

def mesh_info(obj_class = 'horse',model_dir = 'datasets/cachedir/imnet/models/horse'):
    mesh_parts = trimesh.load(osp.join(model_dir, '{}_parts.obj'.format(obj_class)), process=False)
    mesh_one = trimesh.load_mesh(osp.join(model_dir, '{}.obj'.format(obj_class)), process=False)
    hierarchy_xml = osp.join(model_dir, 'hierarchy.xml')
    components = mesh_parts.split(only_watertight=False)
    with open(osp.join(model_dir, 'part_names.txt')) as f:
        part_names = [l.strip() for l in f.readlines()]
    parts = get_parts_from_components(mesh_one, components, part_names)

    hierarchy = ET.parse(hierarchy_xml)
    main = hierarchy.getroot()
    allNodes = [None for _ in range(len(parts))]
    create_dpm(allNodes, main, parts, 0)

    verts = mesh_one.vertices
    faces = mesh_one.faces
    blend_alpha = np.zeros([len(verts), len(allNodes)])
    mesh_distances = convert_mesh_to_graph(verts,faces)
    dpm = {'verts': verts, 'faces': faces, 'nodes' : allNodes, 'alpha': blend_alpha, 'mesh_dist': mesh_distances}
    complete_dpm(dpm)

    ass = dpm['alpha']
    dump_blend_color_maps(model_dir, dpm['nodes'], dpm['alpha'], verts, faces, tone='Reds')
    part_verts = [-1] * len(verts)
    part_parent = {0:-1}
    part_rotation_center = {}
    for i in dpm['nodes']:
        for j in i['vids']:
            part_verts[j] = i['id']
        for j in i['children']:
            part_parent[j] = i['id']
        if 'boundary' in i:
            part_rotation_center[i['id']] = i['rc']
    part_rotation_center[0] = np.mean(verts[[350,346]],axis=0)
    for key in part_rotation_center.keys():
        part_rotation_center[key] = list(part_rotation_center[key])
    arti_info_mesh = {'parts':part_verts, 'num_parts': len(part_rotation_center),
                      'rotation_center': part_rotation_center,'parent': part_parent, 'alpha':ass}
    return arti_info_mesh



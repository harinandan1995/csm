import scipy.io as sio
import numpy as np
import trimesh

data = sio.loadmat('datasets/horse/shape.mat')
verts = data['verts']
faces = data['faces']
mesh = trimesh.Trimesh(verts,faces)
mesh.show()




# # # tail
# # points = [[-0.02,0.7264,0.3301],[-0.001727,0.7924,0.05096]]
#
#
def normal_face(points):
    nfaces = points
    V = nfaces[1] - nfaces[0]
    W = nfaces[2] - nfaces[0]
    nx = V[1]*W[2] - V[2]*W[1]
    ny = V[2]*W[0] - V[0]*W[2]
    nz = V[0]*W[1] - V[1]*W[0]
    n = np.array([nx,ny,nz])
    return n

# matlab
# #head
head_f = [[0.0724,-0.5706,0.6167],[0.01242,-0.5253,0.6793],[0.0003205,-0.5928,0.4328]]

# # neck
neck_f = [[0.1372,-0.4344,0.1971],[-0.03215,-0.2264,0.501],[-0.1066,-0.2787,0.4178]]
head_f = np.array(head_f)
neck_f = np.array(neck_f)

nh = normal_face(head_f)
nn = normal_face(neck_f)
head = []
for i in range(len(verts)):
    if (head_f[0] - verts[i]).dot(nh) <= 0:
        head.append(i+1)
print(head)
neck = []
for i in range(len(verts)):
    if (head_f[0] - verts[i]).dot(nh) > 0 and (neck_f[0] - verts[i]).dot(nn) <= 0:
        neck.append(i+1)
print(neck)
#
# # left front leg
# points = np.array([[0.1147,-0.3069,-0.1071],[0.09215,-0.1581,-0.153],[0.06169,-0.2364,-0.1573]])
# lfleg = []
# nrfl = normal_face(points)
# for i in range(len(verts)):
#     if (points[0] - verts[i]).dot(nrfl) >= 0 and verts[i][2] <= -0.1109 \
#         and verts[i][1]<= -0.1461 and verts[i][0]>=0.06169 :
#         lfleg.append(i+1)
# print(lfleg)
#
# right front leg
rfleg_f = np.array([[0.1823,-0.2255,0.1141],[-0.1441,-0.3066,-0.1038],[-0.131,-0.1607,-0.1537]])
rfleg = []
nrfl = normal_face(rfleg_f)
for i in range(len(verts)):
    if (rfleg_f[0] - verts[i]).dot(nrfl) < 0 and verts[i][2] <= -0.1109 \
        and verts[i][1]<= -0.1425 and verts[i][0]<=-0.06385:
        rfleg.append(i+1)
print(rfleg)
#
# # right hind leg
# rhleg_f = np.array([[0.1961,0.6162,-0.02581],[-0.1788,0.452,-0.01429],[-0.1128,0.8237,0.02853]])
# rhleg = []
# nrhl = normal_face(rhleg_f)
# for i in range(len(verts)):
#     if (rhleg_f[0] - verts[i]).dot(nrhl) < 0 and verts[i][2] <= -0.0441 \
#         and verts[i][1]>=0.5047 and verts[i][0]>=0.05086 :
#         rhleg.append(i+1)
# print(rhleg)
#
# # left hind leg
# points = np.array([[0.1389,0.4415,-0.01026],[0.1277,0.8164,0.02843],[-0.0831,0.4979,-0.04016]])
# lhleg = []
# nrhl = normal_face(points)
# for i in range(len(verts)):
#     if (points[0] - verts[i]).dot(nrhl) >= 0 and verts[i][2] <= -0.0441 \
#         and verts[i][1]>=0.5047 and verts[i][0]<=-0.05093 :
#         lhleg.append(i+1)
# print(lhleg)
#
# others = set(head + neck + rfleg + lfleg + rhleg + lhleg)
# labels = ['body']*643
# for i in head:
#     labels[i] = 'head'
# for i in neck:
#     labels[i] = 'neck'
# for i in lfleg:
#     labels[i] = 'left-front-leg'
# for i in rfleg:
#     labels[i] = 'right-front-leg'
# for i in lhleg:
#     labels[i] = 'left-front-leg'
# for i in rhleg:
#     labels[i] = 'right-front-leg'
# print(labels[1:])



# use distance to get label of vertices
# rfl = np.array([[-0.1428,-0.2171,-0.3319],[-0.07812,-0.1806,-0.3453]])
# rflc = rfl.mean(axis=0)
# mrfl = np.array([-0.1507,-0.2996,-0.7647])
# md = np.linalg.norm(rflc-mrfl)
# rfleg = []
# for i in range(len(verts)):
#     if np.linalg.norm(verts[i]-rflc) <= md and verts[i][2] <= -0.1109 \
#         and verts[i][1]<= -0.1607 and verts[i][0]<=-0.06914:
#         rfleg.append(i+1)
# print(rfleg)

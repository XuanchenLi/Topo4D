import copy
import json
import imageio
import torch
import os
import open3d as o3d
import math
from PIL import Image
import pywavefront
import nvdiffrast.torch as dr
from torchvision.utils import save_image
from skimage import io
import numpy as np
from tqdm import tqdm
import trimesh
from external import build_rotation
from face3d import mesh
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def setup_camera(cam, w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[3, :3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=False
    )
    
    return cam


def params2rendervar(params):
    rendervar = {
            'means3D': params['means3D'],
            'colors_precomp': params['rgb_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
        }
    return rendervar

def params2rendervar_dense(params, variables):
    rendervar = {
            'means3D': params["dense_means3D"],
            'colors_precomp': params['dense_rgb_colors'],
            'rotations': torch.nn.functional.normalize(params['dense_unnorm_rotations']),
            'opacities': torch.sigmoid(params['dense_logit_opacities']),
            'scales': torch.exp(params['dense_log_scales']),
            'means2D': torch.zeros_like(torch.from_numpy(params["dense_means3D"].clone().detach().cpu().numpy()), requires_grad=True, device="cuda") + 0
        }
    
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def l2_loss(x, y):
    return torch.sqrt(((x - y) ** 2) + 1e-20).mean()

def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def quaternion_similarity(q1, q2):
    # θ = arccos[ 2*⟨q1,q2⟩^2 − 1 ]
    return np.rad2deg(math.acos(2 * (np.clip((np.dot(q1, q2)), -1, +1) ** 2) - 1))

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if not k.startswith("dense")}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, args):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]

    os.makedirs(os.path.join(args.output_dir, args.exp, args.seq), exist_ok=True)
    np.savez(os.path.join(args.output_dir, args.exp, args.seq, "params"), **to_save)


def compute_vertex_colors(scene):
    _, mesh = list(scene.meshes.items())[0]
    vertices = scene.vertices
    uvs = scene.parser.tex_coords
    #print(mesh.materials[0].texture._path)
    texture_image = Image.open(mesh.materials[0].texture._path)
    vertex_colors = {}
    for fid, face in enumerate(mesh.faces):
        #print(face, fid)
        for vid, vertex_index in enumerate(face):
            # 获取顶点和 UV 坐标
            vertex = vertices[vertex_index]
            pos = fid*24+vid*8  # T2F_N3F_V3F
            uv = mesh.materials[0].vertices[pos:pos+2]
            color = get_color_from_texture(texture_image, uv)

            if vertex_index in vertex_colors.keys():
                vertex_colors[vertex_index].append(color)
            else:
                vertex_colors[vertex_index] = [color]  

    for vertex, colors in vertex_colors.items():
        avg_color = np.mean(colors, axis=0)
        vertex_colors[vertex] = tuple(avg_color.astype(int))
    #print(sorted(vertex_colors.items()))
    vertex_color = [np.array(value) for key, value in sorted(vertex_colors.items())]
    #print(np.array(vertex_color).shape, np.array(vertices).shape)
    return np.array(vertex_color)



def get_vertex_uvs(scene):
    _, mesh = list(scene.meshes.items())[0]
    vertex_uvs = {}
    for fid, face in enumerate(mesh.faces):
        #print(face, fid)

        for vid, vertex_index in enumerate(face):
            # 获取顶点和 UV 坐标
            pos = fid*24+vid*8  # T2F_N3F_V3F
            uv = mesh.materials[0].vertices[pos:pos+2]

            #print(vertex_index)

            if vertex_index in vertex_uvs.keys():
                vertex_uvs[vertex_index].append(uv)
                #print(vertex_uvs[vertex_index])
                #print(len(vertex_uvs[vertex_index]))
            else:
                vertex_uvs[vertex_index] = [uv]  

    vertex_uvs = [list(set(tuple(item) for item in value)) for key, value in sorted(vertex_uvs.items())]
    return vertex_uvs
    


def compute_vertex_attribute_by_weight_2(variables, attribute):
    
    vertex_father = variables['dense_vertex_father']
    weight = variables['dense_vertex_weight']
    quad_faces = variables["dense_quad_faces"]
    
    dense_attribute = np.zeros((variables["dense_vertex"].shape[0], attribute.shape[1]))
    dense_attribute[:attribute.shape[0]] = attribute
    
    #print(quad_faces[vertex_father].shape)
    selected_attrs = attribute[quad_faces[vertex_father].squeeze(1)]
    weight = weight[..., None]
    #print(selected_attrs.shape, )
    dense_attribute[attribute.shape[0]:] = np.sum(selected_attrs * weight, axis=1)
    #print(dense_attribute)
    
    return dense_attribute



    
def write_obj_with_uv(file_path, vertices, faces, uvs, uv_faces):

    with open(file_path, 'w') as file:

        for vertex in vertices:
            file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')     

        for uv in uvs:
            file.write(f'vt {uv[0]} {uv[1]}\n')

        for face, uv_face in zip(faces, uv_faces):
            face_str = 'f'
            for v_idx, uv_idx in zip(face, uv_face):
                
                face_str += f' {v_idx + 1}/{uv_idx + 1}'
            file.write(face_str + '\n')

def write_obj_del_vertex(file_path, variables, vertices, faces, uvs, uv_faces, del_list):
    neighbor_indices = variables["neighbor_indices"].clone().detach().cpu().numpy()
    del_list = [idx for idx in del_list if all([nei in del_list for nei in neighbor_indices[idx]])]
    res_idx = np.array([idx for idx in range(vertices.shape[0]) if idx not in del_list])
    res_map = {oid:nid for nid, oid in enumerate(res_idx)}
    
    vertices = vertices[res_idx]
    res_face_idx = []
    for idx, face in enumerate(faces):
        flag = 0
        for vid in face:
            if vid in del_list:
               flag = 1
               break
        if flag == 0:
            res_face_idx.append(idx)

    faces_ = [faces[i] for i in res_face_idx]
    for fid, face in enumerate(faces_):
        for idx, vid in enumerate(face):
            faces_[fid][idx] = res_map[vid]
            
    uv_faces_ = [uv_faces[i] for i in res_face_idx]
    write_obj_with_uv(file_path, vertices, faces_, uvs, uv_faces_)

def get_color_from_texture(image, uv_coord):
    width, height = image.size
    u, v = uv_coord
    u = u % 1
    v = v % 1

    x = u * width
    y = (1 - v) * height


    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)



    Q11 = image.getpixel((x1, y1))
    Q21 = image.getpixel((x2, y1))
    Q12 = image.getpixel((x1, y2))
    Q22 = image.getpixel((x2, y2))

    # 计算插值
    R1 = ((x2 - x) * Q11[0] + (x - x1) * Q21[0],
          (x2 - x) * Q11[1] + (x - x1) * Q21[1],
          (x2 - x) * Q11[2] + (x - x1) * Q21[2])

    R2 = ((x2 - x) * Q12[0] + (x - x1) * Q22[0],
          (x2 - x) * Q12[1] + (x - x1) * Q22[1],
          (x2 - x) * Q12[2] + (x - x1) * Q22[2])

    P = ((y2 - y) * R1[0] + (y - y1) * R2[0],
         (y2 - y) * R1[1] + (y - y1) * R2[1],
         (y2 - y) * R1[2] + (y - y1) * R2[2])

    return tuple([int(val) for val in P])


def load_faces_vertices(file_path):
    faces = []
    uv_faces = []
    normal_faces = []
    with open(file_path, 'r') as file:
        for line in file:
  
            if line.startswith('f '):

                parts = line.strip().split(' ')
                parts = [part for part in parts if not len(part) == 0]
                #print(parts)
                face = [int(part.split('/')[0]) - 1 for part in parts[1:]]  
                uv_face = [int(part.split('/')[1]) - 1 for part in parts[1:]]
                normal_faces = [int(part.split('/')[2]) - 1 for part in parts[1:] if len(part.split('/')) == 3]

                faces.append(face)
                uv_faces.append(uv_face)
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    _, mesh = list(scene.meshes.items())[0]
    vertices = np.array(scene.vertices)
    
    return faces, uv_faces, vertices, normal_faces


def get_face_faces(faces, face_idx, face_mask):
    face_list = []
    no_face_list = []
    #print(face_mask)
    for idx, face in enumerate(faces):
        flag = False
        for vid in face:
            if vid in face_mask:
                #print(idx)
                face_list.append(idx)
                #face_idx_list.append(idx)
                flag = True
                break
        if not flag:
            no_face_list.append(idx)
    #print(face_list)
    return np.array(faces[face_list]), np.array(face_idx[face_list], dtype=np.int32),\
        np.array(faces[no_face_list]), np.array(face_idx[no_face_list], dtype=np.int32)


def bilinear_interpolate(vertices, face, num_points):

    num_pts = num_points ** 2
    inner_points = np.zeros((num_pts, 3))
    p1, p2, p3, p4 = vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[3]]
    weights = np.zeros((num_pts, 4))

    for i in range(1, num_points + 1):
        for j in range(1, num_points + 1):

            t = i / (num_points + 1)
            u = j / (num_points + 1)

            point = (1 - t) * (1 - u) * p1 + t * (1 - u) * p2 + t * u * p3 + (1 - t) * u * p4
            inner_points[(i - 1) * num_points + (j - 1)] = point
            weights[(i - 1) * num_points + (j - 1)] = np.array([(1 - t) * (1 - u), t * (1 - u),
                                                                t * u, (1 - t) * u
                                                                ])

    return inner_points, weights



def build_dense_vertices(vertices, faces, density):
    dense_vertices = np.zeros((vertices.shape[0] + density * density * faces.shape[0], 3))
    dense_vertices[:vertices.shape[0]] = vertices
    new_vertex_father = np.zeros((density * density * faces.shape[0], 1), dtype=np.int32)
    new_vertex_weight = np.zeros((density * density * faces.shape[0], 4))
    for idx, face in enumerate(faces):
        new_pts, weight = bilinear_interpolate(vertices, face, density)
        #print(new_vertex_weight[idx * density * density : (idx + 1) * density * density].shape)
        st = vertices.shape[0] + idx * density * density
        dense_vertices[st:st + density * density] = new_pts
        new_vertex_weight[idx * density * density: (idx + 1) * density * density] = weight
        new_vertex_father[idx * density * density: (idx + 1) * density * density] = idx

        
    return dense_vertices, new_vertex_father, new_vertex_weight


def bilinear_interpolate_2(vertices, face, uv_face, num_points, offset, uv_offset, edge_dict, pt_uvs):

    num_pts = (num_points + 2) ** 2 - 4
    num_faces = (num_points + 1) ** 2
    inner_points = np.zeros((num_pts, 3))
    p1, p2, p3, p4 = vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[3]]
    weights = np.zeros((num_pts, 4))
    
    new_faces = np.zeros((num_faces, 4))
    new_uv_faces = np.zeros((num_faces, 4))
    pts_idx = np.zeros((num_points + 2, num_points + 2))
    pts_idx_uv = np.zeros((num_points + 2, num_points + 2))
    cnt = 0
    cnt_f = 0
    
    if len(pt_uvs[face[0]]) == 1 or len(pt_uvs[face[1]]) == 1:
        if tuple(sorted([face[0], face[1]])) in edge_dict:
            p1p2_f = 1
        else:
            edge_dict[tuple(sorted([face[0], face[1]]))] = np.zeros((num_points, 2))
            p1p2_f = 0
    else:
        p1p2_f = 0
    if len(pt_uvs[face[1]]) == 1 or len(pt_uvs[face[2]]) == 1:
        if tuple(sorted([face[1], face[2]])) in edge_dict:
            p2p3_f = 1
        else:
            edge_dict[tuple(sorted([face[1], face[2]]))] = np.zeros((num_points, 2))
            p2p3_f = 0
    else:
        p2p3_f = 0
    if len(pt_uvs[face[2]]) == 1 or len(pt_uvs[face[3]]) == 1:
        if tuple(sorted([face[2], face[3]])) in edge_dict:
            p3p4_f = 1
        else:
            edge_dict[tuple(sorted([face[2], face[3]]))] = np.zeros((num_points, 2))
            p3p4_f = 0
    else:
        p3p4_f = 0
    if len(pt_uvs[face[0]]) == 1 or len(pt_uvs[face[3]]) == 1:
        if tuple(sorted([face[0], face[3]])) in edge_dict:
            p1p4_f = 1
        else:
            edge_dict[tuple(sorted([face[0], face[3]]))] = np.zeros((num_points, 2))
            p1p4_f = 0
    else:
        p1p4_f = 0

    
    for i in range(0, num_points + 2):
        for j in range(0, num_points + 2):
            if (i == 0 and j == 0):
                pts_idx[i, j] = face[0]
                pts_idx_uv[i, j] = uv_face[0]
                continue
            if (i == 0 and j == num_points + 1):
                pts_idx[i, j] = face[3]
                pts_idx_uv[i, j] = uv_face[3]
                continue
            if (i == num_points + 1 and j == 0):
                pts_idx[i, j] = face[1]
                pts_idx_uv[i, j] = uv_face[1]
                continue
            if (i == num_points + 1 and j == num_points + 1):
                pts_idx[i, j] = face[2]
                pts_idx_uv[i, j] = uv_face[2]
                new_faces[cnt_f, 0] = pts_idx[i-1, j-1]
                new_faces[cnt_f, 1] = pts_idx[i, j-1]
                new_faces[cnt_f, 2] = pts_idx[i, j]
                new_faces[cnt_f, 3] = pts_idx[i-1, j]
                
                new_uv_faces[cnt_f, 0] = pts_idx_uv[i-1, j-1]
                new_uv_faces[cnt_f, 1] = pts_idx_uv[i, j-1]
                new_uv_faces[cnt_f, 2] = pts_idx_uv[i, j]
                new_uv_faces[cnt_f, 3] = pts_idx_uv[i-1, j]
                cnt_f += 1
                continue
            
            
            if i == 0 and p1p4_f == 1: # p1p4
                if face[0] > face[3]:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[0], face[3]]))][num_points - j, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[0], face[3]]))][num_points - j, 1]
                else:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[0], face[3]]))][j - 1, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[0], face[3]]))][j - 1, 1]
                
            elif j == 0 and p1p2_f == 1: # p1p2
                if face[0] > face[1]:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[0], face[1]]))][num_points - i, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[0], face[1]]))][num_points - i, 1]
                else:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[0], face[1]]))][i - 1, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[0], face[1]]))][i - 1, 1]
            elif i == num_points + 1 and p2p3_f == 1: #p2p3
                if face[1] > face[2]:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[1], face[2]]))][num_points - j, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[1], face[2]]))][num_points - j, 1]
                else:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[1], face[2]]))][j - 1, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[1], face[2]]))][j - 1, 1]
            elif j == num_points + 1 and p3p4_f == 1: #p3p4
                if face[2] > face[3]:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[2], face[3]]))][i - 1, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[2], face[3]]))][i- 1, 1]
                else:
                    pts_idx[i, j] = edge_dict[tuple(sorted([face[2], face[3]]))][num_points - i, 0]
                    pts_idx_uv[i, j] = edge_dict[tuple(sorted([face[2], face[3]]))][num_points - i, 1]
            else:
            
                # 计算插值比例
                t = i / (num_points + 1)
                u = j / (num_points + 1)

                # 使用双线性插值公式
                point = (1 - t) * (1 - u) * p1 + t * (1 - u) * p2 + t * u * p3 + (1 - t) * u * p4
                inner_points[cnt] = point
                weights[cnt] = np.array([(1 - t) * (1 - u), t * (1 - u),
                                        t * u, (1 - t) * u
                                        ])
                pts_idx[i, j] = offset + cnt
                pts_idx_uv[i, j] = uv_offset + cnt
                
                cnt += 1
            
            if i != 0 and j != 0:
                #print(i, j)
                new_faces[cnt_f, 0] = pts_idx[i-1, j-1]
                new_faces[cnt_f, 1] = pts_idx[i, j-1]
                new_faces[cnt_f, 2] = pts_idx[i, j]
                new_faces[cnt_f, 3] = pts_idx[i-1, j]
                
                new_uv_faces[cnt_f, 0] = pts_idx_uv[i-1, j-1]
                new_uv_faces[cnt_f, 1] = pts_idx_uv[i, j-1]
                new_uv_faces[cnt_f, 2] = pts_idx_uv[i, j]
                new_uv_faces[cnt_f, 3] = pts_idx_uv[i-1, j]
                cnt_f += 1
    
    if p1p4_f == 0 and (len(pt_uvs[face[0]]) == 1 or len(pt_uvs[face[3]]) == 1):
        if face[0] > face[3]:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[0], face[3]]))][i, 0] = pts_idx[0, num_points - i]
                edge_dict[tuple(sorted([face[0], face[3]]))][i, 1] = pts_idx_uv[0, num_points - i]
        else:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[0], face[3]]))][i, 0] = pts_idx[0, i + 1]
                edge_dict[tuple(sorted([face[0], face[3]]))][i, 1] = pts_idx_uv[0, i + 1]
            
    if p1p2_f == 0 and (len(pt_uvs[face[0]]) == 1 or len(pt_uvs[face[1]]) == 1):
        if face[0] > face[1]:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[0], face[1]]))][i, 0] = pts_idx[num_points - i, 0]
                edge_dict[tuple(sorted([face[0], face[1]]))][i, 1] = pts_idx_uv[num_points - i, 0]
        else:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[0], face[1]]))][i, 0] = pts_idx[i + 1, 0]
                edge_dict[tuple(sorted([face[0], face[1]]))][i, 1] = pts_idx_uv[i + 1, 0]
    if p2p3_f == 0 and (len(pt_uvs[face[1]]) == 1 or len(pt_uvs[face[2]]) == 1):
        if face[1] > face[2]:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[1], face[2]]))][i, 0] = pts_idx[num_points + 1, num_points - i]
                edge_dict[tuple(sorted([face[1], face[2]]))][i, 1] = pts_idx_uv[num_points + 1, num_points - i]
        else:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[1], face[2]]))][i, 0] = pts_idx[num_points + 1, i + 1]
                edge_dict[tuple(sorted([face[1], face[2]]))][i, 1] = pts_idx_uv[num_points + 1, i + 1]
    if p3p4_f == 0 and (len(pt_uvs[face[2]]) == 1 or len(pt_uvs[face[3]]) == 1):
        if face[2] > face[3]:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[2], face[3]]))][i, 0] = pts_idx[i + 1, num_points + 1]
                edge_dict[tuple(sorted([face[2], face[3]]))][i, 1] = pts_idx_uv[i + 1, num_points + 1]
        else:
            for i in range(num_points):
                edge_dict[tuple(sorted([face[2], face[3]]))][i, 0] = pts_idx[num_points - i, num_points + 1]
                edge_dict[tuple(sorted([face[2], face[3]]))][i, 1] = pts_idx_uv[num_points - i, num_points + 1]         
    
    #print(cnt_f, num_faces)
    assert cnt_f == num_faces
    return inner_points[:cnt], weights[:cnt], new_faces, new_uv_faces, edge_dict, cnt


def build_dense_vertices_2(variables, vertices, faces, face_idx, density, pt_uvs):
    #print(pt_uvs)
    # for i in pt_uvs:
    #     if len(i) != i:
    #         print(i)
        #print(len(i))
    face_new_pts_num = (density + 2) ** 2 - 4
    new_pts_num = ((density + 2) ** 2 - 4) * faces.shape[0]
    dense_vertices = np.zeros((vertices.shape[0] + new_pts_num, 3))
    dense_vertices[:vertices.shape[0]] = vertices
    new_vertex_father = np.zeros((new_pts_num, 1), dtype=np.int32)
    new_vertex_weight = np.zeros((new_pts_num, 4))
    face_new_face_num = (density + 1) ** 2
    dense_uv_faces = np.zeros((faces.shape[0] * face_new_face_num, 4))
    dense_faces = np.zeros((faces.shape[0] * face_new_face_num, 4))
    dense_uv = np.zeros((variables["uvs_ori"].shape[0] + new_pts_num, 2))
    dense_uv[:variables["uvs_ori"].shape[0]] = variables["uvs_ori"]
    #print(variables["uv_faces_ori"][0])
    uv_ori = variables["uvs_ori"]
    edge_dict = {}
    face_new_pts_num = 0
    st = vertices.shape[0]
    uv_st= variables["uvs_ori"].shape[0]
    for idx, face in enumerate(faces):
        st = st + face_new_pts_num
        uv_st = uv_st + face_new_pts_num
        fid = face_idx[idx]
        uv_face = variables["uv_faces_ori"][fid]
        new_pts, weight, new_faces, new_uv_faces, edge_dict, face_new_pts_num = bilinear_interpolate_2(vertices, face, uv_face, density, st, uv_st, edge_dict, pt_uvs)
        #print(new_vertex_weight[idx * density * density : (idx + 1) * density * density].shape)
        dense_vertices[st:st + face_new_pts_num] = new_pts
        new_vertex_weight[st - vertices.shape[0]: st - vertices.shape[0] + face_new_pts_num] = weight
        new_vertex_father[st - vertices.shape[0]: st - vertices.shape[0] + face_new_pts_num] = idx
        dense_faces[idx*face_new_face_num : (idx+1)*face_new_face_num] = new_faces
        dense_uv_faces[idx*face_new_face_num : (idx+1)*face_new_face_num] = new_uv_faces
        
        #st = variables["uvs_ori"].shape[0] + idx * face_new_pts_num
        #fid = face_idx[idx]
        uvid = variables["uv_faces_ori"][fid]
        uvs = uv_ori[uvid]
        weight = weight[..., None]
        dense_uv[uv_st:uv_st + face_new_pts_num] = np.sum(uvs * weight, axis=1)
        #print(weight.shape, new_uvs.shape)
        #weight = weight[..., None]
        #dense_attribute[attribute.shape[0]:] = np.sum(uvs * weight, axis=1)
        #print(uvs.shape)
    new_pts_cnt = st + face_new_pts_num - vertices.shape[0]
    dense_vertices = dense_vertices[:vertices.shape[0] + new_pts_cnt]
    dense_uv = dense_uv[:variables["uvs_ori"].shape[0] + new_pts_cnt]
    new_vertex_father = new_vertex_father[:new_pts_cnt]
    new_vertex_weight = new_vertex_weight[:new_pts_cnt]
        
    return dense_vertices, dense_faces, dense_uv, dense_uv_faces, new_vertex_father, new_vertex_weight


def triangulate_faces(faces):

    triangulated_faces = []
    for face in faces:
        if len(face) == 4:  
            triangulated_faces.append([face[0], face[1], face[2]])
            triangulated_faces.append([face[0], face[2], face[3]])
        elif len(face) == 3: 
            triangulated_faces.append(face)

    return triangulated_faces


def find_adjacent_vertices(vertices, faces):
    adjacent_vertices = {}

    for i in range(len(vertices)):
        adjacent_vertices[i] = set()

    for quad in faces:
        if len(quad) == 4:
            v1, v2, v3, v4 = quad
            adjacent_vertices[v1].update([v2, v3, v4])
            adjacent_vertices[v2].update([v1, v3, v4])
            adjacent_vertices[v3].update([v1, v2, v4])
            adjacent_vertices[v4].update([v1, v2, v3])
        else:
            v1, v2, v3 = quad
            adjacent_vertices[v1].update([v2, v3])
            adjacent_vertices[v2].update([v1, v3])
            adjacent_vertices[v3].update([v1, v2])

    return {k: list(v) for k, v in adjacent_vertices.items()}

def vertex2face(faces, mask):
    flat_faces = faces
    selected_idx = []
    ex_list = mask
    for fid, f in enumerate(flat_faces):
        flag = 0
        #print(f)
        for idx in f:
            if idx not in ex_list:
                flag = 1
                break
        if flag == 0:
            selected_idx.append(fid)
   
    flat_faces = flat_faces[np.array(selected_idx)]
    return flat_faces

def vertex2face_more(faces, mask):
    flat_faces = faces
    selected_idx = []
    ex_list = mask
    for fid, f in enumerate(flat_faces):
        flag = 0
        #print(f)
        for idx in f:
            if idx in ex_list:
                flag = 1
                break
        if flag == 1:
            selected_idx.append(fid)
   
    flat_faces = flat_faces[np.array(selected_idx)]
    return flat_faces

def label_colormap(n_label=11):
    """Label colormap.
    Parameters
    ----------
    n_labels: int
        Number of labels (default: 11).
    value: float or int
        Value scale or value of label color in HSV space.
    Returns
    -------
    cmap: numpy.ndarray, (N, 3), numpy.uint8
        Label id to colormap.
    """
    if n_label == 11:  # helen, ibugmask
        cmap = np.array(
            [
                (0, 0, 0),
                (255, 255, 0),
                (139, 76, 57),
                (139, 54, 38),
                (0, 205, 0),
                (0, 138, 0),
                (154, 50, 205),
                (72, 118, 255),
                (255, 165, 0),
                (0, 0, 139),
                (255, 0, 0),
            ],
            dtype=np.uint8,
        )
    elif n_label == 19:  # celebamask-hq
        cmap = np.array(
            [
                (0, 0, 0),
                (204, 0, 0),
                (76, 153, 0),
                (204, 204, 0),
                (51, 51, 255),
                (204, 0, 204),
                (0, 255, 255),
                (255, 204, 204),
                (102, 51, 0),
                (255, 0, 0),
                (102, 204, 0),
                (255, 255, 0),
                (0, 0, 153),
                (0, 0, 204),
                (255, 51, 153),
                (0, 204, 204),
                (0, 51, 0),
                (255, 153, 51),
                (0, 204, 0),
            ],
            dtype=np.uint8,
        )
    else:

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap


def update_optimizer(update_list, optimizer):
    for param_group in optimizer.param_groups:
        if param_group["name"] in update_list.keys():
            param_group['lr'] = update_list[param_group["name"]]

cmap = label_colormap(n_label=14)[:, [2, 1, 0]]         
target_colors = [torch.tile(torch.tensor(cmap[label_index]).reshape(3, 1, 1), (1, 512, 375)).cuda() for label_index in range(14)]
target_colors_multiface = [torch.tile(torch.tensor(cmap[label_index]).reshape(3, 1, 1), (1, 512, 333)).cuda() for label_index in range(14)]
target_colors_1K = [torch.tile(torch.tensor(cmap[label_index]).reshape(3, 1, 1), (1, 1024, 750)).cuda() for label_index in range(14)]

def get_mask(target_labels, mask, cmap_index, target_colors):
    filtered_mask = torch.zeros_like(mask)
    filtered_mask *= 0
    mask = mask * 255

    for label in target_labels:
        label_index = cmap_index[label]
        target_color = target_colors[label_index]
        mask_condition = torch.all(torch.abs(mask - target_color) < 1, dim=0)
        mask_condition = torch.tile(mask_condition, (3, 1, 1))
        #print(torch.sum(mask_condition).item())
        filtered_mask[mask_condition] = 1  
    return filtered_mask


def write_loss_json(dir, loss_list, loss_w):
    if os.path.exists(os.path.join(dir, "loss.json")):
        return
    loss_list = {k: (False if v is None else True) for k, v in loss_list.items()}
    dict_list = [loss_list, loss_w]
    # 写入JSON文件
    with open(os.path.join(dir, "loss.json"), 'w') as outfile:
        json.dump(dict_list, outfile, indent=4)
        

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def duplicate_texture_vertex_color_2(variables, colors):
    color_render = []
    uvs = variables["uvs_ori"].copy()
    uvs_texture = variables["uvs_texture_ori"].copy()
    #print(len(uvs), len(uvs_texture))
    uv_dict = {}
    for idx, uvs_ in enumerate(uvs_texture):
        for uv in uvs_:
            uv_dict[tuple(uv)] = idx
    #print(len(uv_dict.keys()))
    color_render = [colors[uv_dict[tuple(uv)]] for uv in uvs]
    return color_render



def process_uv(uv_coords, uv_h = 256, uv_w = 256):
    uv_coords[:,0] = uv_coords[:,0]*(uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1]*(uv_h - 1)
    uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords


def write_texture(path, uvs, colors, faces, res=1024):
    uv_h = uv_w = res
    uv_coords = process_uv(uvs, uv_h, uv_w)
    uv_texture_map = mesh.render.render_colors(uv_coords, faces, colors, uv_h, uv_w, c=3)
    #print(uv_texture_map.shape)
    #print(np.sum(uv_texture_map == np.nan))
    uv_texture_map = (uv_texture_map * 255).astype(np.uint8)
    io.imsave(path, np.squeeze(uv_texture_map))


def save_mesh(out_dir, params, variables, frame, res=1024, gen_texture=True):

    os.makedirs(out_dir, exist_ok=True)
    if frame != 1:
        vertices = params["means3D"].clone().detach()
        mesh_t = trimesh.Trimesh(vertices.cpu().numpy(), faces=variables["faces"])
        normals = torch.from_numpy(mesh_t.vertex_normals.copy()).cuda()
        scales = torch.exp(params['log_scales'])
        rots = build_rotation(params['unnorm_rotations'])
        normals_rot = torch.linalg.inv(rots.float()) @ normals.unsqueeze(-1).float()
        cast_scales = torch.sqrt(1.0 / (torch.sum((normals_rot.squeeze(2) ** 2) / (scales ** 2), dim=1)))
        cast_scales = torch.clamp(cast_scales, 0.0, 0.001)

        vertices = vertices + cast_scales.unsqueeze(-1) * normals
        vertices = vertices.clone().detach().cpu().numpy()
    else:
        vertices = params["means3D"].clone().detach().cpu().numpy()

    vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=-1)  # turn to homogeneous coords
    trans_g = np.linalg.inv(variables["trans_g"])
    vertices = vertices @ trans_g[:3, :3].T
    vertices = vertices + trans_g[:3, 3]
    
    write_obj_with_uv(os.path.join(out_dir, "face.obj"),
                vertices, copy.deepcopy(variables['faces_ori']),
                copy.deepcopy(variables['uvs_ori']),
                copy.deepcopy(variables['uv_faces_ori']),
                    )

    if gen_texture:
        dense_colors = params["dense_rgb_colors"].clone().detach().clamp(0.0, 1.0).cpu().numpy()
        # In our topology, vertices around the seam may correspond to multiple UV coordinates
        # Special treatment is needed for these points, before uv mapping
        colors = np.array(duplicate_texture_vertex_color_2(variables, dense_colors[:params["means3D"].shape[0]]))
        colors = np.concatenate((colors, dense_colors[params["means3D"].shape[0]:]), axis=0)

        write_texture(os.path.join(out_dir, "face.png"), np.array(variables['dense_uvs'].copy()), colors, np.array(variables['dense_uv_faces'].copy()), res=res)
    
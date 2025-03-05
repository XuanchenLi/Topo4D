import argparse
import time
import cv2
import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
import trimesh
from loss_util import *
import pywavefront
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from camera import load_camera, rotate_image, rotate_image_cam
from helpers import *
from external import *
import torchvision.transforms as transforms
import pickle


#rotation of the input views: -1 for clockwise; 1 for anticlockwise
rotate_mask = {
    "J87351627":-1, "K19210959":-1, "K98707288":1, "K98707289":1, "K98707290":-1,
    "K98707291":1, "K98707292":-1, "K98707293":-1, "K98707294":-1, "K98707295":-1,
    "K98707296":1, "K98707297":-1, "K99216880":-1, "K99216881":-1, "K99216882":1,
    "K99216883":1, "K99216885":1, "K99216886":-1, "K99216887":1, "K99216888":1,
    "K99216890":-1, "K99216891":-1, "K99216892":1, "K99216893":1,
    
}
#predefined facial regions
face_region = [
    "Caruncle", "Chin", "Ear", "EarNeckBack", "EarSocket", "EyeLidBottom",
    "EyeLidInnerBottom", "EyeLidInnerTop", "EyeLidOuterTop", "EyeLidOuterBottom",
    "EyeLidTop", "EyeSocket", "Face", "HeadBack", "LipBottom", "LipInnerBottom",
    "LipInnerTop", "LipOuterBottom", "LipOuterTop", "LipTop", "MouthSocket", "MouthSocketBottom", "MouthSocketTop",
    "NeckBack", "NeckFront", "Nostril"
]
#views that needs to be excluded
blacklist = {
    # "J87351627", ""K19210959"
}

#cmap index for face parsing mask
cmap_index = {
    "background": 0, "skin": 1, "l_eyebrow": 2, "r_eyebrow": 3,
    "l_eye": 4, "r_eye":5, "nose": 6, "upper_lip": 7,
    "inner_mouth": 8, "lower_lip": 9, "hair": 10, "l_ear": 11,
    "r_ear": 12, "glasses":13
}


def get_cameras(data_dir, seq, resize_factor=8):
    calib_fname = os.path.join(data_dir, seq, "cameras.xml")
    img_fname = sorted(glob(os.path.join(data_dir, seq, "000001", "*.jpg"))) + sorted(glob(os.path.join(data_dir, seq, "000001", "*.png")))
    cams = {}
    cams_ori = {}

    for fname in img_fname:
        cam, trans_g = load_camera(calib_fname, fname.split("/")[-1].split(".")[0], resize_factor=resize_factor, rt=rotate_mask[fname.split('/')[-1].split('.')[0]])
        cam_ori, trans_g = load_camera(calib_fname, fname.split("/")[-1].split(".")[0], resize_factor=1, rt=rotate_mask[fname.split('/')[-1].split('.')[0]])
        
        cams[fname.split("/")[-1]] = cam
        cams_ori[fname.split("/")[-1]] = cam_ori

    return cams, cams_ori, trans_g

def get_dataset(data_dir, seq, frame, cameras, use_mask=False, blacklist=[]):
    dataset = []
    
    img_fnames = sorted(glob(os.path.join(data_dir, seq, "%06d" % frame, "*.jpg"))) + sorted(glob(os.path.join(data_dir, seq, "%06d" % frame, "*.png")))
    img_fnames = [item for item in img_fnames if not any(item.split("/")[-1].startswith(black) for black in blacklist)]
    for idx, img_f in enumerate(img_fnames):
        im = np.array(copy.deepcopy(Image.open(img_f))) / 255.0
        ori_h, ori_w = im.shape[0:2]
        cam = cameras[img_f.split("/")[-1]]
        im = rotate_image(im, cam, angle=rotate_mask[img_f.split('/')[-1].split('.')[0]]*90)
        
        mask = None
        if use_mask:
            mask_fname = os.path.join("/",*(img_f.split('/')[:-2]), "mask", *(img_f.split('/')[-2:]))
            mask_fname = mask_fname.split(".")
            mask_fname[-1] = "png"
            mask_fname = '.'.join(mask_fname)
            mask = np.array(copy.deepcopy(Image.open(mask_fname)))[:ori_h, :ori_w] / 255.0
            mask = rotate_image(mask, cam, angle=rotate_mask[img_f.split('/')[-1].split('.')[0]]*90)
            mask = torch.tensor(mask).float().cuda().permute(2, 0, 1)
        
        w, h, k, w2c = cam["image_size"][1], cam["image_size"][0], cam["intrinsics"], cam["extrinsics"]

        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])])

        cam = setup_camera(cam, w, h, k, w2c, near=0.01, far=100)
        im = torch.tensor(im).float().cuda().permute(2, 0, 1)

        
        dataset.append({'cam': cam, 'im': im , 'id': idx, 'mask': mask, 'cam_name': img_f.split("/")[-1].split('.')[0]})
    return dataset

def get_batch(todo_dataset, dataset, idx=None):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    if idx is None:
        curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    else:
        curr_data = todo_dataset[idx]
    return curr_data


def initialize_params(args, trans_g):
    rt_dir = os.path.join(args.input_dir, args.seq)

    #========================initialize gaussian mesh attributes and topological priors===============================
    scene = pywavefront.Wavefront(os.path.join(rt_dir, "face_v5.obj"), collect_faces=True)
    _, mesh = list(scene.meshes.items())[0]
    vertices = np.asarray(scene.vertices)
    uvs_ori = np.array(scene.parser.tex_coords)
    uvs_texture_ori = get_vertex_uvs(scene)

    trans_g = np.linalg.inv(trans_g)  # global transform
    vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=-1)  # turn to homogeneous coords
    vertices = trans_g.dot(vertices.T).T  # 
    vertices = vertices[:, :3] / vertices[:, 3][:, np.newaxis]  # turn to non-homogeneous coords

    colors = compute_vertex_colors(scene)

    max_cams = 24
    sq_dist, _ = o3d_knn(vertices, 1)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    mesh_t = trimesh.Trimesh(vertices, faces=np.array(mesh.faces))
    q_normals = build_quaterion(torch.from_numpy(mesh_t.vertex_normals.copy()).float()).numpy()
    # initialize gaussian attributes
    params = {
        'means3D': vertices,
        'rgb_colors': colors[:, :3] / 255.0,
        'unnorm_rotations': q_normals,
        'logit_opacities': np.ones((vertices.shape[0], 1)) * 1000,
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist)/ 2)[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    faces, uv_faces_ori, _, _= load_faces_vertices(os.path.join(rt_dir, "face_v5.obj"))

    #read predefined facial regions
    with open('./assets/facial_regions.pkl', 'rb') as f:
        facial_regions = pickle.load(f)


    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'init_scale': torch.from_numpy(np.sqrt(mean3_sq_dist)).cuda(),
                 'facial_regions': facial_regions,
            
                 'faces': np.array(mesh.faces),
                 'trans_g': trans_g,
                 'faces_ori': faces,
                 'uvs_ori': np.array(uvs_ori),
                 'uv_faces_ori': uv_faces_ori,
                 'uvs_texture_ori': uvs_texture_ori
                 }
    
    # initialize one-ring neighbor priors
    mesh = trimesh.Trimesh(params['means3D'].clone().detach().cpu().numpy(), faces=variables["faces"])
    neighbor_indices = find_adjacent_vertices(vertices, faces)
    neighbor_indices = [lst for _, lst in sorted(neighbor_indices.items())]
    neighbor_indices_ori = copy.deepcopy(neighbor_indices)
    max_ns = max(len(lst) for lst in neighbor_indices)
    for i, lst in enumerate(neighbor_indices):
        if len(lst) < max_ns:
            lst.extend([i] * (max_ns - len(lst)))
    neighbor_sq_dist = []
    neighbor_wh_dist = []
    for vertex_index, neighbors in enumerate(neighbor_indices):
        distances = []
        wh_distances = []
        for idx, neighbor_index in enumerate(neighbors):
            if neighbor_index in variables["facial_regions"]["eye_del_masks"] and vertex_index not in variables["facial_regions"]["eye_del_masks"]:
                distance = np.sum((mesh.vertices[vertex_index] - mesh.vertices[neighbor_index])**2)
                distances.append(distance)
                wh_distances.append(np.sum(((mesh.vertices[vertex_index] - mesh.vertices[neighbor_index])*1000)**2))
            else:
                distance = np.sum((mesh.vertices[vertex_index] - mesh.vertices[neighbor_index])**2)
                distances.append(distance)
                wh_distances.append(distance)

        neighbor_sq_dist.append(np.array(distances))
        neighbor_wh_dist.append(np.array(wh_distances))
    neighbor_sq_dist = np.array(neighbor_sq_dist)
    neighbor_sq_wh_dist = np.array(neighbor_wh_dist)
    neighbor_weight = np.exp(-2000 * neighbor_sq_wh_dist)
    neighbor_weight[neighbor_weight==1] = 0.0
    
    neighbor_indices = np.array(neighbor_indices)
    neighbor_dist = np.sqrt(neighbor_sq_dist)


    variables["neighbor_indices_ori"] = neighbor_indices_ori
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()
    

    #========================initialize dense gaussian mesh attributes and topological priors===============================
    # preprocess mesh for uv-space densification
    # Our topology is mostly composed of quadrilateral faces, but there are also a few triangular faces present, we only densify the quadrilateral faces for convenience
    # For efficiency, we only optimize the texture of the frontal face area, so here we only process the frontal face
    dense_num = args.density if args.gen_tex else 1
    vertices = params['means3D'].clone().detach().cpu().numpy()
    face_vertex_mask = facial_regions["face_masks"]
    # find the triangular faces and quadrilateral faces
    quad_faces = np.array([face for face in variables['faces_ori'] if len(face) == 4])
    quad_faces_idx = np.array([idx for idx, face in enumerate(variables['faces_ori']) if len(face) == 4])
    tri_faces = np.array([face for face in variables['faces_ori'] if len(face) == 3])
    tri_uv_faces = np.array([face for face in variables['uv_faces_ori'] if len(face) == 3])
    # only process the frontal face area
    quad_faces, quad_faces_idx, no_face_quad_faces, no_face_quad_faces_idx = get_face_faces(np.array(quad_faces), 
                                                                                            np.array(quad_faces_idx)
                                                                                            , face_vertex_mask)
    no_face_quad_uv_faces = np.array([variables['uv_faces_ori'][i] for i in no_face_quad_faces_idx]) 
    # uv-space densification, only needs to be computed once at the begining.
    # This step may take a few minutes, depending on the densification density.
    # This step is actually only for obtaining the topology and interpolation weights of the dense Gaussian Mesh,
    # which is fixed for the same topology and density. You can save the calculation result, and directly load it for acceleration.
    vertices, dense_faces, dense_uv, dense_uv_faces, new_vertex_father, new_vertex_weight = build_dense_vertices_2(variables, vertices, quad_faces, quad_faces_idx, dense_num, variables['uvs_texture_ori'])
    # The topology of the final dense Gaussian Mesh includes a few triangular faces, densified quadrilateral faces, and original non-frontal area faces 
    dense_uv_faces = tri_uv_faces.tolist() + dense_uv_faces.tolist() + no_face_quad_uv_faces.tolist()
    dense_faces = tri_faces.tolist() + dense_faces.tolist() + no_face_quad_faces.tolist()
    # Triangulate all faces for convenience
    dense_uv_faces = triangulate_faces(dense_uv_faces)
    dense_faces = triangulate_faces(dense_faces)
    
    variables['dense_quad_faces'] = quad_faces
    variables['dense_vertex_father'] = new_vertex_father
    variables['dense_vertex_weight'] = new_vertex_weight
    variables['dense_faces'] = dense_faces
    variables['dense_uv_faces'] = dense_uv_faces
    variables['dense_vertex'] = np.array(vertices)

    sq_dist, _ = o3d_knn(np.array(vertices), 4)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    aux_ = params['rgb_colors'].clone().detach()
    with torch.no_grad():
        # Set the color of the non-frontal area to black
        aux_[variables["facial_regions"]["static_masks"]] = 0.0
        aux_[variables["facial_regions"]["dynamic_masks"]] = 0.0
        aux_[variables["facial_regions"]["mouth_inner_masks"]] = 0.0
    aux_ = aux_.cpu().numpy()
    colors = compute_vertex_attribute_by_weight_2(variables, aux_)
    # Initialize dense gaussian mesh learnable attributes
    params['dense_rgb_colors'] = torch.nn.Parameter(torch.tensor(colors).cuda().float().contiguous().requires_grad_(True))
    params['dense_logit_opacities'] = torch.nn.Parameter(inverse_sigmoid(0.9999 * torch.ones((vertices.shape[0], 1))).cuda().float().contiguous().requires_grad_(True))
    variables["dense_init_colors"] = params['dense_rgb_colors'].clone().detach()
    params["dense_means3D"] = torch.from_numpy(
            compute_vertex_attribute_by_weight_2(variables, params["means3D"].clone().detach().cpu().numpy())
            ).cuda().float().requires_grad_(False)
    params['dense_log_scales'] = torch.nn.Parameter(torch.tensor(np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))).cuda().float().contiguous().requires_grad_(True))
    params['dense_unnorm_rotations'] = torch.nn.Parameter(torch.tensor(np.tile([1, 0, 0, 0], (vertices.shape[0], 1))).cuda().float().contiguous().requires_grad_(True))
    
    variables['dense_uvs'] = dense_uv
    variables['dense_max_2D_radius'] = torch.zeros(vertices.shape[0]).cuda().float()


    return params, variables


def initialize_optimizer(params):
    
    lrs = {
        'means3D': 0.0,
        'rgb_colors': 0.0025,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.0,
        'log_scales': 0.001,
        
        'dense_means3D': 0.0,
        'dense_unnorm_rotations': 0.001,
        'dense_logit_opacities': 0.0,
        'dense_log_scales': 0.0,
        'dense_rgb_colors': 0.0025,
        

        'cam_m': 1e-4,
        'cam_c': 1e-4,

    }
    param_groups = [
        {'params': [v], 'name': k, 'lr': lrs[k]}
        for k, v in params.items()
        ]

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep, use_mask=False, losses_list={}, losses_weights={}):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    if losses_weights["im"] != 0:
        im, radius, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    
        curr_id = curr_data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    
    if not use_mask:
        if losses_weights["im"] != 0:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    else:
        if is_initial_timestep:
            losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))       
        else:
            # mask inner mouth to avoid lip vertices learning the wrong color
            if losses_weights["im"] != 0:
                target_labels = ["inner_mouth"]
                filtered_mask = get_mask(target_labels, curr_data["mask"], cmap_index, variables["target_colors_low"])
                no_masked_index = filtered_mask == 1
                masked_gt = curr_data['im'].clone()
                masked_gt[no_masked_index] *= 0.1
                losses['im'] = 0.8 * l1_loss_v1(im, masked_gt) + 0.2 * (1.0 - calc_ssim(im, masked_gt))
                
    
    if not is_initial_timestep:
        fg_pts = rendervar['means3D']
        fg_rot = rendervar['rotations']
        
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["rig_w"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["rot_w"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["iso_w"])
        
        
        losses['flat'] = losses_list['flat'](rendervar['means3D']) if losses_list['flat'] is not None else torch.tensor(0.0)
        losses['flat_lip_bottom'] = losses_list['flat_lip_bottom'](rendervar['means3D']) if losses_list['flat_lip_bottom'] is not None else torch.tensor(0.0)
        losses['flat_lip_socket'] = losses_list['flat_lip_socket'](rendervar['means3D']) if losses_list['flat_lip_socket'] is not None else torch.tensor(0.0)
        losses['flat_eye'] = losses_list['flat_eye'](rendervar['means3D']) if losses_list['flat_eye'] is not None else torch.tensor(0.0)
        losses['flat_face_bottom'] = losses_list['flat_face_bottom'](rendervar['means3D']) if losses_list['flat_face_bottom'] is not None else torch.tensor(0.0)
        losses['flat_lid_top'], _ = losses_list['flat_lid_top'](rendervar['means3D'], variables["cos_init_lid_top"]) if losses_list['flat_lid_top'] is not None else torch.tensor(0.0)
        losses['flat_lid_bottom'], _ = losses_list['flat_lid_bottom'](rendervar['means3D'], variables["cos_init_lid_bottom"]) if losses_list['flat_lid_bottom'] is not None else torch.tensor(0.0)
        losses['flat_lip'], _ = losses_list['flat_lip'](rendervar['means3D'], variables["cos_init_lip"]) if losses_list['flat_lip'] is not None else torch.tensor(0.0)
        losses['flat_mouth'], _ = losses_list['flat_mouth'](rendervar['means3D'], variables["cos_init_mouth"]) if losses_list['flat_mouth'] is not None else torch.tensor(0.0)

    else:
        min_scale = torch.min(rendervar["scales"], dim=1).values
        losses['scale'] = torch.sum(min_scale)
        max_scale = torch.max(rendervar["scales"], dim=1).values
        losses['scale_max'] = torch.sum(F.relu(max_scale - variables["init_scale"] * 1.5))
        # At the first frame, cache the topological priors for computing losses in following frames
        losses['flat_lid_top'], variables["cos_init_lid_top"] = losses_list['flat_lid_top'](rendervar['means3D']) if losses_list['flat_lid_top'] is not None else torch.tensor(0.0)
        losses['flat_lid_bottom'], variables["cos_init_lid_bottom"] = losses_list['flat_lid_bottom'](rendervar['means3D']) if losses_list['flat_lid_bottom'] is not None else torch.tensor(0.0)
        losses['flat_lip'], variables["cos_init_lip"] = losses_list['flat_lip'](rendervar['means3D']) if losses_list['flat_lip'] is not None else torch.tensor(0.0)
        losses['flat_mouth'], variables["cos_init_mouth"] = losses_list['flat_mouth'](rendervar['means3D']) if losses_list['flat_mouth'] is not None else torch.tensor(0.0)


    loss_detail = {k:losses_weights[k] * v for k, v in losses.items() }
    loss = sum([losses_weights[k] * v for k, v in losses.items()])
    if losses_weights["im"] != 0:
        seen = radius > 0
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen
    return loss, variables, loss_detail



def get_loss_dense(params, curr_data, variables, timestep, iteration, optimizer, 
                   use_mask=False, losses_list={}, losses_weights={}):
    losses = {}
    
    rendervar = params2rendervar_dense(params, variables)
    rendervar['means2D'].retain_grad()

    im, radius, _, _= Renderer(raster_settings=curr_data['cam'])(**rendervar)

    variables['dense_means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    if not use_mask:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    else:
        # facial regions for optimization
        target_labels = ["skin", "l_eyebrow", "r_eyebrow", "nose", 
                         "upper_lip", "lower_lip", "l_ear", "r_ear",
                          "hair"]
        filtered_mask = get_mask(target_labels, curr_data["mask"], cmap_index, variables["target_colors_dense"])
        masked_index = filtered_mask == 1
        masked_im = torch.zeros_like(im).cuda()
        masked_im[masked_index] = im[masked_index]
        masked_gt = torch.zeros_like(im).cuda()
        masked_gt[masked_index] = curr_data['im'][masked_index]
        losses['im'] = ((masked_im - masked_gt).abs().sum() / masked_index.sum())
    
    losses['soft_color'] = l1_loss_v2(params['dense_rgb_colors'], variables["dense_init_colors"])
    
    loss_detail = {k:losses_weights[k] * v for k, v in losses.items() }
    loss = sum([losses_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['dense_scene_radius'] = radius
    variables['dense_max_2D_radius'][seen] = torch.max(radius[seen], variables['dense_max_2D_radius'][seen])
    variables['dense_seen'] = seen
    
    
    return loss, variables, loss_detail


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])

    new_pts = pts
    new_rot = torch.nn.functional.normalize(rot)

    prev_inv_rot_fg = rot
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.clone().detach()
    variables['prev_offset'] = prev_offset.clone().detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)
    
    
    return params, variables


def initialize_post_first_timestep(params, variables, optimizer):
    variables["first_frame_attributes"] = {}
    with torch.no_grad():
        variables["first_frame_attributes"]["dynamic_eye_colors"] = params['rgb_colors'][variables["facial_regions"]["dynamic_eye_masks"]].clone().detach()
  
        variables["first_frame_attributes"]["inner_colors"] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["eye_del_masks"]])
        variables["first_frame_attributes"]["eye_around_colors"] = params['rgb_colors'][variables["facial_regions"]["eye_around_masks"]].clone().detach()
        variables["first_frame_attributes"]["eye_bottom_colors"] = params['rgb_colors'][variables["facial_regions"]["region_masks"]["EyeLidBottom"]].clone().detach()
        variables["first_frame_attributes"]["mouth_around_colors"] = params['rgb_colors'][variables["facial_regions"]["mouth_around_masks"]].clone().detach()
        variables["first_frame_attributes"]["face_bottom_colors"] = params['rgb_colors'][variables["facial_regions"]["face_bottom_masks"]].clone().detach()
    return variables


def report_progress(params, dataset, t, i, progress_bar, every_i=500, idx=[], path=None):
    if i % every_i == 0:
        for id in idx:
            for d in dataset:
                if d["cam_name"] == id:
                    data = d
                    break
            # data = dataset[id]
            # print(dataset)
            im, _, _, _= Renderer(raster_settings=data['cam'])(**params2rendervar(params))
            curr_id = data['id']
            im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
            psnr = calc_psnr(im, data['im']).mean()
            if path is None:
                save_image(im, "./output/test/vis%s_%d.png" % (id, i))
            else:
                os.makedirs(os.path.join(path, "%06d"%t), exist_ok=True)
                save_image(im, os.path.join(path, "%06d"%t, "vis%s_%d.png" % (id, i)))
        #save_image(data['im'], "./output/test/vis_gt%d.png" % i)
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)
        
        
def report_progress_dense(variables, params, dataset, t, i, progress_bar, every_i=500, idx=[], path=None):
    if i % every_i == 0:
        for id in idx:
            for d in dataset:
                if d["cam_name"] == id:
                    data = d
                    break
            im, _, _, _= Renderer(raster_settings=data['cam'])(**params2rendervar_dense(params, variables))
            curr_id = data['id']
            #im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
            psnr = calc_psnr(im, data['im']).mean()
            if path is None:
                save_image(im, "./output/test/dense_%s_%d.png" % (id, i))
            else:
                os.makedirs(os.path.join(path, "%06d"%t), exist_ok=True)
                save_image(im, os.path.join(path, "%06d"%t, "dense_%s_%d.png" % (id, i)))
            #save_image(data['im'], "./output/test/vis_gt%d.png" % i)
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def update_dense_states(params, variables, is_init):

    if is_init:
        pass
    else:
        variables["dense_init_colors"] = params['dense_rgb_colors'].clone().detach()
        params["dense_means3D"] = torch.from_numpy(
            compute_vertex_attribute_by_weight_2(variables, params["means3D"].clone().detach().cpu().numpy())
            ).cuda().float()

    return params, variables


def initialize_losses(variables):

    losses = {
        'flat': FlattenLoss(torch.tensor(variables["facial_regions"]["flat_faces"])),
        'flat_lip_bottom': FlattenLoss(torch.tensor(variables["facial_regions"]["lip_bottom_flat_faces"])),
        'flat_eye': FlattenLoss_v2(variables, mask_list=["EyeLidOuterTop", "EyeLidTop", "EyeLidBottom"]),
        'flat_lip_socket': FlattenLoss_v2(variables, mask_list=[],
                                          pre_mask=variables["facial_regions"]["lip_socket_flat_masks"].tolist()),

        'flat_face_bottom': FlattenLoss_v2(variables, 
                                           mask_list=["LipOuterTop", "LipOuterBottom", "Chin", "NeckFront",
                                                      "LipBottom", "LipTop", "LipInnerBottom", "LipInnerTop",
                                                      "EyeLidOuterBottom", "EyeLidBottom",
                                                       "MouthSocket", "EyeSocket"],
                                           pre_mask=variables["facial_regions"]["face_flat_masks"].tolist(),
                                           ex_mask= variables["facial_regions"]["lip_flat_edge_masks"].tolist()
                                           ),
        'flat_lip': SoftFlattenLoss(torch.tensor(variables["facial_regions"]["lip_flat_faces"])),
        'flat_mouth': SoftFlattenLoss(torch.tensor(variables["facial_regions"]["mouth_flat_faces"])),

        'flat_lid_top': SoftFlattenLoss(torch.tensor(variables["facial_regions"]["lid_top_flat_faces"])),
        'flat_lid_bottom': SoftFlattenLoss(torch.tensor(variables["facial_regions"]["lid_bottom_flat_faces"]))
    }

    losses_weights = {'im': 1.0, 'rigid': 3.5, 'rot': 20.0, 'iso': 20.0,
                'flat': 2e-4, 'flat_lip_bottom': 2e-4,
                'flat_lid_top': 2e-4, 'flat_lid_bottom':1e-2, 'flat_lip':1e-4, 'flat_mouth':1e-3,
                'flat_eye': 1e4, 'flat_face_bottom': 1e3, 'flat_lip_socket': 1e3,
                'scale': 10.0, 'scale_max':10.0
                }
    losses_weights_dense = {
        'im': 1.0, 'soft_color': 0.02
    }

    # Adjust different loss weights based on predefined facial regions
    with torch.no_grad():
        iso_w = variables["neighbor_weight"].detach().clone()
        if losses_weights["iso"] != 0:
            iso_w[variables["facial_regions"]["eye_lid_up_masks"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["EyeLidOuterTop"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["EyeLidTop"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["mouth_inner_masks"], :] *= 5.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["Chin"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["LipOuterTop"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["LipOuterBottom"], :] *= 1.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["EyeSocket"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["MouthSocket"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["region_masks"]["NeckFront"], :] *= 0.0 / losses_weights["iso"]
            iso_w[variables["facial_regions"]["face_flat_masks"], :] *= 0.0 / losses_weights["iso"]


        rig_w = variables["neighbor_weight"].detach().clone()
        if losses_weights["rigid"] != 0:
            rig_w[variables["facial_regions"]["eye_lid_up_masks"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["EyeLidOuterTop"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["EyeLidTop"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["mouth_inner_masks"], :] *= 0.5 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["Chin"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["LipOuterTop"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["LipOuterBottom"], :] *= 0.1 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["MouthSocket"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["EyeSocket"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["region_masks"]["NeckFront"], :] *= 0.0 / losses_weights["rigid"]
            rig_w[variables["facial_regions"]["face_flat_masks"], :] *= 0.0 / losses_weights["rigid"]
        rot_w = variables["neighbor_weight"].detach().clone()
        if losses_weights["rot"] != 0:
            rot_w[variables["facial_regions"]["region_masks"]["EyeLidOuterTop"], :] *= 50.0 / losses_weights["rot"]
            rot_w[variables["facial_regions"]["region_masks"]["EyeLidTop"], :] *= 50.0 / losses_weights["rot"]
            rot_w[variables["facial_regions"]["region_masks"]["EyeLidBottom"], :] *= 100.0 / losses_weights["rot"]
            rot_w[variables["facial_regions"]["region_masks"]["EyeSocket"], :] *= 100.0 / losses_weights["rot"]
            rot_w[variables["facial_regions"]["eye_inner_masks"], :] *= 100.0 / losses_weights["rot"]

    variables["iso_w"] = iso_w
    variables["rig_w"] = rig_w
    variables["rot_w"] = rot_w

    return variables, losses, losses_weights, losses_weights_dense


def train(args):
    if os.path.exists(os.path.join(args.output_dir, args.exp, args.seq)):
        print(f"Experiment '{args.exp}' for sequence '{args.seq}' already exists. Exiting.")
        return

    cameras, _, trans_g = get_cameras(args.input_dir, args.seq, resize_factor=args.down_ratio)
    cameras_dense, _, trans_g = get_cameras(args.input_dir, args.seq, resize_factor=1)
    params, variables = initialize_params(args, trans_g)
    low_img_size = list(cameras.values())[0]["image_size"]
    ori_img_size = list(cameras_dense.values())[0]["image_size"]
    optimizer = initialize_optimizer(params)
    variables, losses, loss_weights, loss_weights_dense = initialize_losses(variables)
    
    output_params = []

    # learning weights for subsquent frames
    new_lr = {
        'logit_opacities': 0.0,
        'log_scales': 0.0,
        'unnorm_rotations': 0.001,
        'rgb_colors':0.0,
        'means3D': 0.000016,

        'dense_log_scales': 0.0,
        'cam_m': 0.0,
        'cam_c':0.0,
    }

    # cache some predefined values, which should be fixed during optimization
    with torch.no_grad():
        static_verts = params['means3D'][variables["facial_regions"]["static_masks"]].clone().detach()
        static_face_colors = params['rgb_colors'][variables["facial_regions"]["face_masks"]].clone().detach()
        params["rgb_colors"][variables["facial_regions"]["dynamic_mouth_masks"]] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["dynamic_mouth_masks"]])
        params["rgb_colors"][variables["facial_regions"]["dynamic_eye_masks"]] = torch.ones_like(params["rgb_colors"][variables["facial_regions"]["dynamic_eye_masks"]])
        dynamic_mouth_opacity = inverse_sigmoid(0.99999 * torch.ones((params["means3D"][variables["facial_regions"]["dynamic_mouth_masks"]].shape[0], 1))).requires_grad_(False).cuda()
        dynamic_mouth_scales = torch.log(torch.ones_like(params["log_scales"][variables["facial_regions"]["dynamic_mouth_masks"]]) * 0.01).cuda()
        eye_inner_opacity = inverse_sigmoid(0.000001 * torch.ones((params["means3D"][variables["facial_regions"]["eye_inner_masks"]].shape[0], 1))).requires_grad_(False).cuda()
        mouth_inner_scales = torch.log(torch.ones_like(params["log_scales"][variables["facial_regions"]["mouth_inner_masks"]]) * 0.002).cuda()
        dynamic_eye_scales = torch.log(torch.ones_like(params["log_scales"][variables["facial_regions"]["dynamic_eye_masks"]]) * 0.0025).cuda()
        dynamic_eye_opacity = inverse_sigmoid(0.99999 * torch.ones((params["means3D"][variables["facial_regions"]["dynamic_eye_masks"]].shape[0], 1))).requires_grad_(False).cuda()

    use_mask = True
    use_mask_dense = False
    cmap = label_colormap(n_label=14)[:, [2, 1, 0]]    
    if use_mask:
        variables['target_colors_low'] = [torch.tile(torch.tensor(cmap[label_index]).reshape(3, 1, 1), (1, low_img_size[0], low_img_size[1])).cuda() for label_index in range(14)]
    if use_mask_dense:
        variables["target_colors_dense"] = [torch.tile(torch.tensor(cmap[label_index]).reshape(3, 1, 1), (1, ori_img_size[0], ori_img_size[1])).cuda() for label_index in range(14)]
    
    # gen_tex = False
    for t in range(args.frame_num):
        is_initial_timestep = (t == 0)
        num_iter_per_timestep = args.init_opt_num if is_initial_timestep else args.opt_num
        
        if not is_initial_timestep:
            # Initialization before frame by frame optimization
            params, variables = initialize_per_timestep(params, variables, optimizer)
            use_mask = True
            new_lr["rgb_colors"] = 0.0
            new_lr["means3D"] = 0.000016
            update_optimizer(new_lr, optimizer)

        # Read the current frame data
        dataset = get_dataset(args.input_dir, args.seq, t + 1, cameras, use_mask=use_mask, blacklist=blacklist)
        if len(dataset) == 0:
            break
        todo_dataset = []

        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep geometry {t}")
        loss_w = loss_weights
        # Current frame optimization
        for i in range(num_iter_per_timestep):
                
            curr_data = get_batch(todo_dataset, dataset, idx=None)
            loss, variables, loss_detail = get_loss(params, curr_data, variables, is_initial_timestep, 
                                       use_mask=use_mask, losses_list=losses, losses_weights=loss_w)
            
            loss.backward()
            
            # freeze values that should be fixed during optimization
            with torch.no_grad():
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # The values of some regions should be fixed to avoid tracking wrong facial areas
                params['means3D'][variables["facial_regions"]["static_masks"]] = static_verts 
                params["logit_opacities"][variables["facial_regions"]["eye_inner_masks"]] = eye_inner_opacity
                params["rgb_colors"][variables["facial_regions"]["dynamic_mouth_masks"]] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["dynamic_mouth_masks"]])
                params["logit_opacities"][variables["facial_regions"]["dynamic_mouth_masks"]] = dynamic_mouth_opacity
                params["log_scales"][variables["facial_regions"]["dynamic_mouth_masks"]] = dynamic_mouth_scales
                params["log_scales"][variables["facial_regions"]["mouth_inner_masks"]] = mouth_inner_scales
                if is_initial_timestep:
                    if i < int(num_iter_per_timestep * 0.7):
                        # The attributes of the eye socket area should be optimized later to avoid learning wrong colors
                        params["log_scales"][variables["facial_regions"]["dynamic_eye_masks"]] = dynamic_eye_scales
                        params["logit_opacities"][variables["facial_regions"]["dynamic_eye_masks"]] = dynamic_eye_opacity

                    params['rgb_colors'][variables["facial_regions"]["face_masks"]] = static_face_colors
                    params['rgb_colors'][variables["facial_regions"]["mouth_inner_masks"]] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["mouth_inner_masks"]])
                    
                else:
                    # The values of some regions should be fixed to avoid tracking wrong facial areas
                    params['rgb_colors'][variables["facial_regions"]["dynamic_eye_masks"]] = variables["first_frame_attributes"]["dynamic_eye_colors"]
                    params['rgb_colors'][variables["facial_regions"]["dynamic_mouth_masks"]] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["dynamic_mouth_masks"]])
                    params['rgb_colors'][variables["facial_regions"]["eye_del_masks"]] =  variables["first_frame_attributes"]["inner_colors"]
                    params['rgb_colors'][variables["facial_regions"]["eye_around_masks"]] = variables["first_frame_attributes"]["eye_around_colors"]
                    params['rgb_colors'][variables["facial_regions"]["region_masks"]["EyeLidBottom"]] = variables["first_frame_attributes"]["eye_bottom_colors"]
                    params['rgb_colors'][variables["facial_regions"]["mouth_around_masks"]] = variables["first_frame_attributes"]["mouth_around_colors"]
                    params['rgb_colors'][variables["facial_regions"]["face_bottom_masks"]] = variables["first_frame_attributes"]["face_bottom_colors"]
                    params['rgb_colors'][variables["facial_regions"]["mouth_inner_masks"]] = torch.zeros_like(params["rgb_colors"][variables["facial_regions"]["mouth_inner_masks"]])
                    
                report_progress(params, dataset, t+1, i, progress_bar, every_i=args.log_freq, idx=args.log_views, path=os.path.join(args.output_dir, args.exp, args.seq))

            # updata color for a few iter
            if not is_initial_timestep and i >= args.opt_num - 100:
                n_lr = copy.deepcopy(new_lr)
                n_lr["rgb_colors"] = 0.00025
                n_lr["means3D"] = 0.0

                update_optimizer(n_lr, optimizer)
                loss_w = loss_weights
        progress_bar.close()
        #  =================================================Texture Optimization======================================================
        sav_tex = True
        if args.gen_tex:
            sav_tex = True
            # num_iter_per_timestep = args.init_dense_opt_num if is_initial_timestep else args.dense_opt_num
            num_iter_per_timestep = args.dense_opt_num
            with torch.no_grad(): 
                params, variables = update_dense_states(params, variables, is_initial_timestep)

            dataset = get_dataset(args.dense_input_dir, args.seq, t + 1, cameras_dense, use_mask=use_mask_dense, blacklist=blacklist)
            if len(dataset) == 0:
                num_iter_per_timestep = 0
                sav_tex = False
            todo_dataset = []
            progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep texture {t}")

            for i in range(num_iter_per_timestep):
                curr_data = get_batch(todo_dataset, dataset, idx=None)
                with torch.no_grad():
                    params["dense_rgb_colors"][variables["facial_regions"]["static_masks"]] = 0.0
                    params["dense_rgb_colors"][variables["facial_regions"]["dynamic_masks"]] = 0.0
                    params["dense_rgb_colors"][variables["facial_regions"]["mouth_inner_masks"]] = 0.0
                loss, variables, loss_detail = get_loss_dense(params, curr_data, variables, t, i, optimizer,
                                        use_mask=use_mask_dense, 
                                        losses_weights=loss_weights_dense)
                loss.backward()
                with torch.no_grad():
                    optimizer.step()
                    optimizer.zero_grad()
                    report_progress_dense(variables, params, dataset, t+1, i, progress_bar, every_i=args.dense_log_freq, idx=args.log_views, path=os.path.join(args.output_dir, args.exp, args.seq))
            progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        
        
        if is_initial_timestep:
            # cache some values that shoud be fixed during optimizaing the following frames after first frame optimization
            variables = initialize_post_first_timestep(params, variables, optimizer)

        if t % args.ckp_freq == 0 and t != 0:
            save_params(output_params, args)
            write_loss_json(os.path.join(args.output_dir, args.exp, args.seq), losses, loss_weights)
        
        save_mesh(os.path.join(args.output_dir, args.exp, args.seq, "%06d"%(t + 1)), params, variables, t + 1, res=args.tex_res, gen_texture=args.gen_tex and sav_tex)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp', type=str, default=f'exp_op1', help="Experiment name.")
    parser.add_argument('-s', '--seq', type=str, default="seq_01", help="Input sequence name.")
    parser.add_argument('-id', '--input_dir', type=str, default=f'/data/Topo4D/videos_low', help="Root of inputs, the input sequence should be '$input_dir/$seq'")
    parser.add_argument('-od', '--output_dir', type=str, default=f'/data/Topo4D/Topo4D_results', help="Root of outputs, results will be saved in '$output_dir/$exp/$seq'")
    parser.add_argument('-did', '--dense_input_dir', type=str, default=f'/data/Topo4D/videos', help="Root of high resolution inputs, the input sequence should be '$dense_input_dir/$seq'")
    parser.add_argument('-fn', '--frame_num', type=int, default=800, help="Frame number.")
    parser.add_argument('-t', '--gen_tex', action='store_true', help="Whether generate texture.")
    parser.add_argument('-tr', '--tex_res', type=int, default=8192, help="Texture resolution.")
    parser.add_argument('-dn', '--density', type=int, default=30, help="Density for uv-space densification.")
    parser.add_argument('-dr', '--down_ratio', type=int, default=8, help="Downsample ratio of geometry optimization inputs compared with raw captures.")
    parser.add_argument('-ddr', '--dense_down_ratio', type=int, default=1, help="Downsample ratio of texture optimization inputs compared with raw captures.")

    parser.add_argument('-ion', '--init_opt_num', type=int, default=7000, help="Iteration number for optimizing the first frame.")
    parser.add_argument('-on', '--opt_num', type=int, default=1100, help="Iteration number for geometry generation.")
    parser.add_argument('-don', '--dense_opt_num', type=int, default=301, help="Iteration number for texture generation.")
    parser.add_argument('-lf', '--log_freq', type=int, default=500, help="Frequence of saving gaussian rendering results per frame.")
    parser.add_argument('-dlf', '--dense_log_freq', type=int, default=300, help="Frequence of saving dense gaussian rendering results per frame.")
    parser.add_argument('-lv', '--log_views', type=list, default=["K98707293"], help="Views of the saved renderings.")
    parser.add_argument('-cf', '--ckp_freq', type=int, default=5, help="Frequence of saving gaussian attributes.")


    args = parser.parse_args()

    train(args)
    torch.cuda.empty_cache()
from bdb import set_trace
from fileinput import filename
import math
import os
import xml.etree.ElementTree as ET
import torch
from pickletools import uint8
import numpy as np
from scipy.linalg import lstsq
from os.path import splitext, basename
from skimage.transform import rescale, resize, rotate
from scipy.spatial.transform import Rotation as R

def convert_distortion_parms(k1, k2, fl, fx, fy, width, height):
    # OpenCV wants radial distortion parameters that are applied to image plane coordinates
    # prior to being scaled by fx and fy (so not pixel coordinates). In contrast, k1 and k2
    # are defined via Tsai camera calibration http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/DIAS1/
    K1 = k1 * (fl ** 2.0)
    K2 = k2 * (fl ** 4.0)
    # Also, K1 and K2 are actually undistortion coefficients. They go from distorted to undistorted image
    # plane coordinates. OpenCV wants coefficients that go the other way.
    r_values = .01 * np.array(range(1, 101)) * (((width / fx) ** 2.0 + (height / fy) ** 2.0) ** 0.5)
    undistorted_r_values = r_values * (1 + K1 * (r_values ** 2.0) + K2 * (r_values ** 4.0))
    distortion_factors = r_values / undistorted_r_values
    # Given the undistorted and distorted distances, we solve for the new distortion factors via linear regression
    k1, k2 = lstsq(np.matrix([undistorted_r_values ** 2.0, undistorted_r_values ** 4.0]).T, np.matrix(distortion_factors - 1.0).T)[0]
    return (k1, k2)



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


def extract_agisoft_intrinsic(rt, sensor_idx, resize_factor=1, rot=0):
    f = None
    K = None
    K2 = None
    pw = None
    py = None
    cx = None
    cy = None
    img_w = None
    img_h = None
    #print(sensor_idx)
    for t in rt.findall("sensor"):
        if int(t.get("id")) == sensor_idx:
            for p in t.findall("property"):
                if p.get("name") == "pixel_width":
                    pw = float(p.get("value"))
                if p.get("name") == "pixel_height":
                    py = float(p.get("value"))

            res_node = t.find("resolution")
            img_w = int(res_node.get("width"))
            img_h = int(res_node.get("height"))

            c_node = t.find("calibration")
            f = float(c_node.find("f").text)
            if c_node.find("cx") is not None:
                cx = img_w / 2.0 + float(c_node.find("cx").text)
                cy = img_h / 2.0 + float(c_node.find("cy").text)
            else:
                cx = img_w / 2.0 
                cy = img_h / 2.0 
            if c_node.find("k1") is not None:
                K = float(c_node.find("k1").text)
            else:
                K = 0.0
            if c_node.find("k2") is not None:
                K2 = float(c_node.find("k2").text)
            else:
                K2 = 0.0
            break

    focal_length = f
    principal_point = np.array([cx, cy])
    pw = pw if pw is not None else 1.0
    py = py if py is not None else 1.0
    pixel_size = np.array([pw, py])

    if resize_factor != 1:
        img_w, img_h = math.floor(img_w / resize_factor), math.floor(img_h / resize_factor)
        focal_length /= resize_factor
        principal_point /= resize_factor

    # Transformation from undistorted image plane to distorted image coordinates
    #print(pixel_size)
    K1, K2 = convert_distortion_parms(K, K2, focal_length * pixel_size[0], focal_length, focal_length,
                                      img_w, img_h)
    radial_distortion = np.array([K1, K2]).reshape(-1, )
    if rot != 0:
        intrinsics = np.array([
        [focal_length, 0, principal_point[1]],
        [0, focal_length, img_w - principal_point[0]],
        [0, 0, 1.0]])
        img_size = np.array([img_w, img_h])
    else:
        intrinsics = np.array([
        [focal_length, 0, principal_point[0]],
        [0, focal_length, principal_point[1]],
        [0, 0, 1.0]])
        img_size = np.array([img_h, img_w])

    return radial_distortion, intrinsics,  img_size


def extract_agisoft_extrinsic(rt, component, img_name, to_meters=False, rot=0):
    node = None
    sensor_id = None
    component = component.find("component")
    transform_g = None
    if component is not None:
        if component.find("transform") is not None:
            R_G = np.array([float(val) for val in component.find("transform").find("rotation").text.split()]).reshape((3, 3))
            T_G = np.array([float(val) for val in component.find("transform").find("translation").text.split()]).reshape((3,))
            transform_g = np.eye(4)
            transform_g[:3, :3] = R_G
            transform_g[:3, 3] = T_G
        #transform_g[:3, 1:3] *= -1  # camera 2 world; opengl axis
        #transform_g = np.linalg.inv(transform_g)
        
    for t in rt.findall("camera"):
        if t.get("label") == img_name:
            sensor_id = int(t.get("sensor_id"))
            node = t
            break
    #print(img_name)
    transform = np.array([float(val) for val in node.find("transform").text.split()]).reshape((4, 4))

    transform[:3, 1:3] *= -1  # camera 2 world; opengl axis
    theta = -1 * rot * 90 * np.pi / 180  # 90 degrees
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix_z_90 = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1],
    ])
    transform[:3, :3] = transform[:3, :3].dot(rotation_matrix_z_90)
    #print(transform)
    extrinsics_opengl = np.linalg.inv(transform)[:3, :4]
    R_opengl = extrinsics_opengl[:3, :3]
    t_opengl = extrinsics_opengl[:3, 3]
    # Define the coordinate transformation matrix from OpenGL to COLMAP
    # This flips the Y and Z axis
    coord_transform = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])
    # Apply the coordinate transformation
    R_colmap = np.dot(coord_transform, R_opengl)
    t_colmap = np.dot(coord_transform, t_opengl)
    extrinsic_colmap = np.eye(4)
    extrinsic_colmap[:3, :3] = R_colmap
    extrinsic_colmap[:3, 3] = t_colmap
    c2w = get_camera_to_world_transformation(extrinsic_colmap)
    #print(333, c2w[:3, :4])
    center = c2w[:3, 3]
    view_direction = c2w[:3, :3].dot(np.array([0, 0, 1]))

    return extrinsic_colmap[:3, :4], center, view_direction, sensor_id, transform_g


def load_camera(calib_fname, img_name, resize_factor=1, to_meters=False, rt=0):
    tree = ET.parse(calib_fname)
    root = tree.getroot()
    root = root.find("chunk")
    #component = root.find("components")
    extrinsics, center, view_direction, id, trans_g = extract_agisoft_extrinsic(root.find("cameras"), root.find("components"), img_name, to_meters, rot=rt)
    radial_distortion, intrinsics, img_size = extract_agisoft_intrinsic(root.find("sensors"), id, resize_factor, rot=rt)

    camera = {
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'radial_distortion': radial_distortion,
        'camera_center': center,
        'view_direction': view_direction,
        'image_size': img_size,
        'name': img_name
    }
    return camera, trans_g


def get_camera_to_world_transformation(extrinsics):
    '''
    Get transformation matrix (3,4) that transforms points in the local camera coordinate systems
    to the world coordinate system.
    '''

    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    return np.hstack((R.T, -R.T.dot(t)[:,np.newaxis]))

def rotate_image(image, camera=None, angle = 90):
    image = rotate(image, angle, resize=True)
    return image

def rotate_image_cam(image, camera=None, angle = 90):
     image = rotate(image, angle, resize=True)
     if camera is not None:
        Rt = np.array([ 
            [ 0,    1,             0           ],
            [-1,    0, camera['image_size'][1] ],
            [ 0,    0,             1           ]
        ])
        theta = - angle * np.pi / 180  # 90 degrees
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix_z_90 = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1],
        ])
        # rotation_matrix_z_90 = np.array([
        #     [1, 0, 0],
        #     [0, cos_theta, sin_theta],
        #     [-sin_theta, 0, cos_theta],
        # ])
        #camera['extrinsics'][:3, :3] = camera['extrinsics'][:3, :3].dot(rotation_matrix_z_90.T)
        
        #camera['extrinsics'][:3, :3] = camera['extrinsics'][:3, :3].dot(rotation_matrix_z_90)
        #camera['extrinsics'][:3, :3] = rotation_matrix_z_90.T.dot(camera['extrinsics'][:3, :3])
        #camera['extrinsics'][:3, 3] = camera['extrinsics'][:3, 3].dot(rotation_matrix_z_90)
        camera = dict(camera)
        fx, fy = camera['intrinsics'][0, 0], camera['intrinsics'][1, 1]
        
        camera['intrinsics'] = Rt.dot(camera['intrinsics'])
        camera['intrinsics'][0, 0] = fy
        camera['intrinsics'][1, 1] = fx
        camera['intrinsics'][0, 1] = 0.0
        camera['intrinsics'][1, 0] = 0.0
        camera['image_size'] = camera['image_size'][::-1]
        return image, camera
     else:
        return image
     

def scale_image(image, scale_factor, camera=None):
    img = rescale(image, scale_factor, channel_axis=2, anti_aliasing=True)
    if camera is None:
        return img
    else:
        scale_mat = np.eye(3)
        scale_mat[0,0] = scale_mat[1,1] = scale_factor
        camera['intrinsics'] = scale_mat.dot(camera['intrinsics'])
        return img, camera

def perspective_project(points, camera_intrinsics, camera_extrinsics, radial_distortion, eps=1e-7):
    '''
    Projection of 3D points into the image plane using a perspective transformation.
    :param points:      array of 3D points (num_points X 3)
    :return:            array of projected 2D points (num_points X 2)
    '''

    num_points, _ = points.shape
    ones = np.ones((num_points, 1))
    points_homogeneous = np.concatenate((points, ones), axis=-1)

    # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
    points_image = camera_extrinsics.dot(points_homogeneous.T).T

    # Transformation from 3D camera coordinate system to the undistorted image plane 
    z_coords = points_image[:,2]
    z_coords[np.where(np.abs(z_coords) < eps)] = 1.0
    points_image[:,0] = points_image[:,0] / z_coords
    points_image[:,1] = points_image[:,1] / z_coords

    # Transformation from undistorted image plane to distorted image coordinates
    K1, K2 = radial_distortion[0], radial_distortion[1]
    r2 = points_image[:,0]**2 + points_image[:,1]**2
    r4 = r2**2
    radial_distortion_factor = (1 + K1*r2 + K2*r4)
    points_image[:,0] = points_image[:,0]*radial_distortion_factor
    points_image[:,1] = points_image[:,1]*radial_distortion_factor    
    points_image[:,2] = 1.0

    # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
    points_image = camera_intrinsics.dot(points_image.T).T
    return points_image

def batch_perspective_project(points, camera_intrinsics, camera_extrinsics, radial_distortion, eps=1e-7):
    device = points.device
    batch_size, num_points, _ = points.shape
    points = points.transpose(1, 2)

    ones = torch.ones(batch_size, 1, num_points).to(device)
    points_homogeneous = torch.cat((points, ones), axis=-2) # (batch_size, 4, num_points)

    # Transformation from the world coordinate system to the image coordinate system using the camera extrinsic rotation (R) and translation (T)
    points_image = camera_extrinsics.bmm(points_homogeneous) # (batch_size, 3, num_points)

    # Transformation from 3D camera coordinate system to the undistorted image plane 
    mask = (points_image.abs() < eps)
    mask[:,:2,:] = False
    points_image[mask] = 1.0 # Avoid division by zero

    points_image_x = points_image[:,0,:] / points_image[:,2,:]
    points_image_y = points_image[:,1,:] / points_image[:,2,:]

    # Transformation from undistorted image plane to distorted image coordinates
    K1, K2 = radial_distortion[:,0], radial_distortion[:,1]       # (batch_size)
    r2 = points_image_x**2 + points_image_y**2            # (batch_size, num_points)
    r4 = r2**2
    radial_distortion_factor = (1 + K1[:, None]*r2 + K2[:, None]*r4)  # (batch_size, num_points)

    points_image_x = points_image_x*radial_distortion_factor
    points_image_y = points_image_y*radial_distortion_factor
    points_image_z = torch.ones_like(points_image[:,2,:])
    points_image = torch.cat((points_image_x[:, None, :], points_image_y[:, None, :], points_image_z[:, None, :]), dim=1)

    # Transformation from distorted image coordinates to the final image coordinates with the camera intrinsics
    points_image = camera_intrinsics.bmm(points_image)              # (batch_size, 3, num_points)    
    points_image = torch.transpose(points_image, 1, 2)[:,:,:2]      # (batch_size, num_points, 2) 
    return points_image


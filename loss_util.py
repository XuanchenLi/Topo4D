import math
import torch
import torch.nn as nn
import trimesh
import pymesh
import numpy as np
from scipy.sparse import coo_matrix

class LaplacianLoss(nn.Module):
    def __init__(self, verts, faces, mask=None):
        super(LaplacianLoss, self).__init__()

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        laplacian = trimesh.smoothing.laplacian_calculation(mesh, equal_weight=False)

        laplacian_coo = laplacian.tocoo()
        indices = torch.LongTensor(np.vstack((laplacian_coo.row, laplacian_coo.col)))
        values = torch.FloatTensor(laplacian_coo.data)
        shape = laplacian_coo.shape
        
        self.laplacian = torch.sparse.FloatTensor(indices, values, torch.Size(shape)).to_dense().cuda()
        if mask is None:
            self.mask = torch.tensor((list(range(verts.shape[0])))).long().cuda()
        else:
            self.mask = torch.tensor(mask).long().cuda()

        self.register_buffer('delta_V', self.laplacian @ torch.tensor(mesh.vertices, dtype=torch.float32).cuda())

    def forward(self, V_prime):
        delta_V_prime = self.laplacian @ V_prime

        loss = torch.sum((delta_V_prime[self.mask] - self.delta_V[self.mask]) ** 2)

        return loss


class ARAPLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()

        dx_sub = self.laplacian.matmul(dx) # N, Nv, 3
        x_sub = self.laplacian.matmul(x)   # N, Nv, 3

        dx_diff = dx_sub - dx[:, None]
        x_diff = x_sub - x[:, None]

        diffdx += dx_diff.pow(2).sum(dim=-1)
        diffx += x_diff.pow(2).sum(dim=-1)

        diff = (diffx - diffdx).abs()
        diff = diff[:, self.laplacian.bool()].mean(dim=1)

        return diff
    
    
    
class EdgeLoss(nn.Module):
    def __init__(self, faces, average=False, size_factor=1.0):
        super(EdgeLoss, self).__init__()
        edge_set = set()

        for tri in faces:
            tri = tri.numpy()

            edge_set.add((tri[0], tri[1]))
            edge_set.add((tri[1], tri[2]))
            edge_set.add((tri[0], tri[2]))
        self.edges = torch.tensor(np.array(list(edge_set))).long().cuda()
        self.size_factor = size_factor

    def forward(self, x):
        x = x * self.size_factor
        p1 = x[self.edges[:, 0]]
        p2 = x[self.edges[:, 1]]

        distance = nn.functional.pairwise_distance(p1, p2, p=2)

        return torch.std(distance)


class NormLoss(nn.Module):
    def __init__(self, norm):
        super(NormLoss, self).__init__()
        self.norm = norm.cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        cos_theta = 1 - self.cos(x, self.norm).abs()
        loss = torch.mean(cos_theta)

        return loss


class FlattenLoss(nn.Module):
    def __init__(self, faces, threshold=0, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.shape[0]
        self.average = average
        self.threshold = threshold

        faces = faces.detach().cpu().numpy()

        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))
        vert_face = {}
        for k, v in enumerate(faces):
            for vx in v:
                if vx not in vert_face.keys():
                    vert_face[vx] = [k]
                else:
                    vert_face[vx].append(k)

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        idx = 0
        nosin_list = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) > 2:
                continue
            # for face in faces:
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) == 2:
                nosin_list.append(idx)

            for faceid in sorted(list(set(vert_face[v0]) & set(vert_face[v1]))):
                face = faces[faceid]
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
            idx += 1
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        v0s = v0s[nosin_list]
        v1s = v1s[nosin_list]
        v2s = v2s[nosin_list]

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s

        vertices = vertices.unsqueeze(0)
        batch_size = vertices.shape[0]
        #print(self.v0s.shape)
        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        
        threshold = math.cos(self.threshold * math.pi / 180)
        cos = torch.where(cos > threshold, -1, cos)
        
        loss = (cos + 1).pow(2).sum(dims)
        #print((cos + 1).pow(2).shape)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss
        

class FlattenLoss_v2(nn.Module):
    def __init__(self, variables, mask_list=[], pre_mask = [], ex_mask=[]):
        super(FlattenLoss_v2, self).__init__()
        self.variables = variables
        #print(self.variables["neighbor_indices_ori"])
        max_ns = max(len(lst) for lst in self.variables["neighbor_indices_ori"])
        self.neighbor_num = torch.tensor([len(lst) for lst in self.variables["neighbor_indices_ori"]]).cuda()
        # min_ns = min(len(lst) for lst in neighbor_indices)
        # print(max_ns, min_ns)
        mask = []
        for i, lst in enumerate(self.variables["neighbor_indices_ori"]):
            m = [1] * self.neighbor_num[i]
            if len(m) < max_ns:
                m.extend([0] * (max_ns - len(lst)))
            mask.append(m.copy())
        self.mask = torch.tensor(mask).cuda().requires_grad_(False)
        self.mask = self.mask.unsqueeze(-1).repeat([1, 1, 3])
        #print(self.mask.shape)
        #print(self.neighbor_num)
        self.loss = nn.MSELoss()
        self.region_mask = []
        for r in mask_list:
            self.region_mask += self.variables["facial_regions"]["region_masks"][r].tolist()
        self.region_mask += pre_mask
        self.region_mask = list(set(self.region_mask) - set(ex_mask))
        if len(self.region_mask) == 0:
            self.region_mask = [idx for idx in range(self.mask.shape[0])]
        self.region_mask = list(set(self.region_mask))
        self.region_mask = torch.from_numpy(np.array(self.region_mask)).cuda()
        
    def forward(self, vertices):
        #neighbor_num = torch.tensor([len(lst) for lst in self.variables["neighbor_indices_ori"]]).cuda()
        neighbor_pos = vertices[self.variables["neighbor_indices"]] * self.mask
        pos_sum = torch.sum(neighbor_pos, dim=1)
        ave_pos = pos_sum / self.neighbor_num.unsqueeze(1)
        #print(self.loss(ave_pos, vertices))
        #print(self.mask.shape, neighbor_pos.shape, ave_pos.shape)
        return self.loss(ave_pos[self.region_mask], vertices[self.region_mask])

class SoftFlattenLoss(nn.Module):
    def __init__(self, faces, threshold=180, average=False):
        super(SoftFlattenLoss, self).__init__()
        self.nf = faces.shape[0]
        self.average = average
        self.threshold = threshold

        faces = faces.detach().cpu().numpy()

        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))
        vert_face = {}
        for k, v in enumerate(faces):
            for vx in v:
                if vx not in vert_face.keys():
                    vert_face[vx] = [k]
                else:
                    vert_face[vx].append(k)

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        idx = 0
        nosin_list = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) > 2:
                continue
            # for face in faces:
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) == 2:
                nosin_list.append(idx)

            for faceid in sorted(list(set(vert_face[v0]) & set(vert_face[v1]))):
                face = faces[faceid]
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
            idx += 1
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        v0s = v0s[nosin_list]
        v1s = v1s[nosin_list]
        v2s = v2s[nosin_list]

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, cos_init=None, eps=1e-6):
        # make v0s, v1s, v2s, v3s

        vertices = vertices.unsqueeze(0)
        batch_size = vertices.shape[0]
        #print(self.v0s.shape)
        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)
        cos_ori = cos.detach().clone()
        dims = tuple(range(cos.ndimension())[1:])
        #threshold = math.cos(self.threshold * math.pi / 180)
        #cos = torch.where(cos < threshold, -1, cos)
        if cos_init is not None:
            #cos = torch.where(abs(torch.arccos(cos) - torch.arccos(cos_init)) < self.threshold * torch.pi / 180, -1, cos)
            loss = (1 - torch.cos(abs(torch.arccos(cos) - torch.arccos(cos_init)))).sum()
        else:
            loss = (cos + 1).pow(2).sum(dims)
        #loss = (cos + 1).pow(2).sum(dims)
        #print((cos + 1).pow(2).shape)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss, cos_ori
        

class SoftFlattenLoss_v2(nn.Module):
    def __init__(self, faces, threshold=180, average=False):
        super(SoftFlattenLoss_v2, self).__init__()
        self.nf = faces.shape[0]
        self.average = average
        self.threshold = threshold

        faces = faces.detach().cpu().numpy()

        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))
        vert_face = {}
        for k, v in enumerate(faces):
            for vx in v:
                if vx not in vert_face.keys():
                    vert_face[vx] = [k]
                else:
                    vert_face[vx].append(k)

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        idx = 0
        nosin_list = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) > 2:
                continue
            # for face in faces:
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) == 2:
                nosin_list.append(idx)

            for faceid in sorted(list(set(vert_face[v0]) & set(vert_face[v1]))):
                face = faces[faceid]
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
            idx += 1
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        v0s = v0s[nosin_list]
        v1s = v1s[nosin_list]
        v2s = v2s[nosin_list]

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, cos_init=None, eps=1e-6):
        # make v0s, v1s, v2s, v3s

        vertices = vertices.unsqueeze(0)
        batch_size = vertices.shape[0]
        #print(self.v0s.shape)
        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)
        cos_ori = cos.detach().clone()
        dims = tuple(range(cos.ndimension())[1:])
        #threshold = math.cos(self.threshold * math.pi / 180)
        #cos = torch.where(cos < threshold, -1, cos)
        if cos_init is not None:
            #cos = torch.where(abs(torch.arccos(cos) - torch.arccos(cos_init)) < self.threshold * torch.pi / 180, -1, cos)
            loss = (1 - torch.cos(abs(torch.arccos(cos) - torch.arccos(cos_init)))).pow(2).sum(dims)
        else:
            loss = (cos + 1).pow(2).sum(dims)
        #loss = (cos + 1).pow(2).sum(dims)
        #print((cos + 1).pow(2).shape)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss, cos_ori


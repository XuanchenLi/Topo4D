# Topo4D: Topology-Preserving Gaussian Splatting for High-Fidelity 4D Head Capture (ECCV 2024)

<a href='https://arxiv.org/pdf/2406.00440/'><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.00434-red?link=https%3A%2F%2Farxiv.org%2Fabs%2F2310.00434"></a>
<a href='https://xuanchenli.github.io/Topo4D/'><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-blue?logo=github&labelColor=black&link=https%3A%2F%2Fraineggplant.github.io%2FDiffPoseTalk"></a>


Official implementation of [Topo4D](https://xuanchenli.github.io/Topo4D/)

---
## Abstract
![teaser](./figs/teaser.png)

4D head capture aims to generate dynamic topological meshes and corresponding texture maps from videos, which is widely utilized in movies and games for its ability to simulate facial muscle movements and recover dynamic textures in pore-squeezing. The industry often adopts the method involving multi-view stereo and non-rigid alignment. However, this approach is prone to errors and heavily reliant on time-consuming manual processing by artists. To simplify this process, we propose Topo4D, a novel framework for automatic geometry and texture generation, which optimizes densely aligned 4D heads and 8K texture maps directly from calibrated multi-view time-series images. Specifically, we first represent the time-series faces as a set of dynamic 3D Gaussians with fixed topology in which the Gaussian centers are bound to the mesh vertices. Afterward, we perform alternative geometry and texture optimization frame-by-frame for high-quality geometry and texture learning while maintaining temporal topology stability. Finally, we can extract dynamic facial meshes in regular wiring arrangement and high-fidelity textures with pore-level details from the learned Gaussians. Extensive experiments show that our method achieves superior results than the current SOTA face reconstruction methods both in the quality of meshes and textures.


---
## Install
```bash
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# face3d
git clone https://github.com/YadiraF/face3d
cd face3d
cd face3d/mesh/cython
python setup.py build_ext -i 
```

## Prepare Your Data
\[TODO\] We will soon release some example sequence for testing.

## Testing
### Optimize Geometry only
```bash
python train.py --input_dir "input low resolution data root" --output_dir "your output root" --exp "experiment name" --seq "sequence name"
```
### Optimize Texture and Geometry
```bash
python train.py --input_dir "input low resolution data root" --output_dir "your output root" --exp "experiment name" --seq "sequence name" --dense_input_dir "input high resolution data root" --gen_tex --tex_res 8192
```

## Acknowledgement
This work is built on awesome research works and open-source projects, thanks a lot to all the authors.
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
- [diff-gaussian-rasterization modified by Jiaxiang Tang](https://github.com/ashawkey/diff-gaussian-rasterization)
- [Face3D](https://github.com/yfeng95/face3d)
- [Face Parsing](https://github.com/hhj1897/face_parsing)

---
## Citation	
If our work is useful for your research, please consider citing:
```
@inproceedings{li2025topo4d,
  title={Topo4D: Topology-Preserving Gaussian Splatting for High-fidelity 4D Head Capture},
  author={Li, Xuanchen and Cheng, Yuhao and Ren, Xingyu and Jia, Haozhe and Xu, Di and Zhu, Wenhan and Yan, Yichao},
  booktitle={European Conference on Computer Vision},
  pages={128--145},
  year={2025},
  organization={Springer}
}
```

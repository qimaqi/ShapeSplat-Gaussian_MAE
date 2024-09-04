# CAD Model Rendering scripts
In this part we show how to render CAD model to images and corresponding poses for later Gaussian-Splats. We use the format of [NeRF Synthetic Dataset](https://github.com/bmild/nerf). We adapt the code from work [work](https://github.com/bmild/nerf) and the code from [work](https://github.com/Xharlie/ShapenetRender_more_variation) for the rendering.

## Installation
We use blender to render CAD model to images. Following blender install [instruction](https://www.blender.org/download/lts/3-6/), we use blender 3.6.13. You can install by
```sh
cd ./blender_install # or change to your prefer location
wget https://www.blender.org/download/release/Blender3.6/blender-3.6.13-linux-x64.tar.xz

tar -xvf blender-3.6.13-linux-x64.tar.xz
```
After unpacking Blender, we can use it for rendering through the Python interface.

## Environment:
we need package **trimesh** and **PIL** for processing
```
pip install trimesh
pip install Pillow
```


## Datasets
Prepare the CAD Dataset for usage. Get the ShapeNet from [ShapeNet website](https://shapenet.org/). Place the installed ShapeNet to **./ShapeNet**. We support both ShapeNetv1 and ShapeNetv2 rendering.
Similarily, get the ModelNet from [ModelNet website](https://modelnet.cs.princeton.edu/#) and place it in **./ModelNet40**. Also you can download it from [Kaggle](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/data).
Note that unlike other point cloud data processing like [Point-MAE](https://github.com/Pang-Yatian/Point-MAE/blob/main/DATASET.md) and [Point-Bert](https://github.com/Julie-tang00/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), we download the CAD model files instead of processed point cloud.

While using these excellent datasets, please remember to cite the original ShapeNet and ModelNet datasets.


## Rendering
After preparing the blender and dataset, we can start rendering CAD Models to images. We use **start_idx** and **end_idx** to divide the whole dataset which is easier for submit multiple jobs to multiple gpus. We save the first object in ShapeNetv2 and ModelNet for example.

```
# for ShapeNetv2

python3 render_shapenet.py --start_idx=0 --end_idx=1 --model_root_dir=./ShapeNet/ --render_root_dir=./ShapeNet/render/ --blender_location=./blender_install/blender-3.6.13-linux-x64/blender --shapenetversion=v2

# for ModelNet40
python3 render_modelnet.py --start_idx=0 --end_idx=1 --model_root_dir=./ModelNet40 --render_root_dir=./ModelNet40/render/ --blender_location=./blender_install/blender-3.6.13-linux-x64/blender


```

The render output format will be :
```
├──object/
│   ├──image.zip
│   ├──depth.zip
│   ├──normal.zip
│   ├──point_cloud.obj
│   ├──transforms_train.json
│   ├──transforms_test.json
│   ├──transforms_val.json
│   ├──.......
├──.......
```
Note that the train use 72 views and validation and test is seleced for last 4 views from train views. You can adjust the views sampling in by modifying variable *vertical_list* in render_blender.py and the data split.


## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
<div style="max-width: 1200px; overflow-x: auto;">
<pre>
<code>
@misc{ma2024shapesplat,
      title={ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining}, 
      author={Qi Ma and Yue Li and Bin Ren and Nicu Sebe and Ender Konukoglu and Theo Gevers and Luc Van Gool and Danda Pani Paudel},
      year={2024},
      eprint={2408.10906},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10906}, 
}

@article{chang2015shapenet,
  title={Shapenet: An information-rich 3d model repository},
  author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
  journal={arXiv preprint arXiv:1512.03012},
  year={2015}
}

@inproceedings{wu20153d,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}

@article{xu2019disn,
  title={Disn: Deep implicit surface network for high-quality single-view 3d reconstruction},
  author={Xu, Qiangeng and Wang, Weiyue and Ceylan, Duygu and Mech, Radomir and Neumann, Ulrich},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}

@inproceedings{mildenhall2020nerf,
 title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
 author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
 year={2020},
 booktitle={ECCV},
}


</code>
</pre>
</div>

## Acknowledgements


# ShapeSplat-Gaussian-MAE
The Offical implementation of work: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining

#### [Qi Ma](https://scholar.google.com/citations?user=l_5rfO4AAAAJ&hl=en&oi=ao)<sup>1,2</sup>$^\star$, [Yue Li](https://unique1i.github.io/)<sup>3</sup>$^\star$, [Bin Ren](https://amazingren.github.io/)<sup>2,4,5</sup>$^\dagger$, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>5</sup>, [Ender Konukoglu](https://people.ee.ethz.ch/~kender/) <sup>1</sup>, [Theo Gevers](https://scholar.google.com/citations?user=yqsvxQgAAAAJ&hl=en&oi=ao)<sup>3</sup>, [Luc Van Gool ](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)<sup>1,2</sup>, and [Danda Pani Paudel](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en)<sup>1,2</sup> 
$\star$: Equal Contribution, $\dagger$: Corresponding Author <br>

<sup>1</sup> ETH Zürich, Switzerland <br>
<sup>2</sup> INSAIT Sofia University, Bulgaria <br>
<sup>3</sup> University of Amsterdam, Netherlands <br>
<sup>4</sup> University of Pisa, Italy <br>
<sup>5</sup> University of Trento, Italy <br>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.10906)
[![project](https://img.shields.io/badge/project-page-brightgreen)](https://unique1i.github.io/ShapeSplat/)
[![code](https://img.shields.io/badge/code-page-brightgreen)](https://github.com/qimaqi/ShapeSplats-Gaussian-MAE.git)


## News
- [x] `21.08.2024`: The Paper is released on [Arxiv](https://arxiv.org/pdf/2408.10906)
- [x] `20.08.2024`: The [project Page](https://scholar.google.com/citations?user=l_5rfO4AAAAJ&hl=en) is released!
- [ ] Code coming soon
- [ ] Dataset release: We are actively discussing this detail with the ShapeNet team and provide an update as soon as possible. Stay tuned! 


## Method
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
3D Gaussian Splatting (3DGS) has become the de facto method of 3D representation in many vision tasks. This calls for the 3D understanding directly in this representation space. To facilitate the research in this direction, we first build a large-scale dataset of 3DGS using the commonly used ShapeNet and ModelNet datasets. Our dataset ShapeSplat consists of 65K objects from 87 unique categories, whose labels are in accordance with the respective datasets. The creation of this dataset utilized the compute equivalent of 2 GPU years on a TITAN XP GPU.
We utilize our dataset for unsupervised pretraining and supervised finetuning for classification and segmentation tasks. To this end, we introduce Gaussian-MAE, which highlights the unique benefits of representation learning from Gaussian parameters. Through exhaustive experiments, we provide several valuable insights. In particular, we show that (1) the distribution of the optimized GS centroids significantly differs from the uniformly sampled point cloud (used for initialization) counterpart; (2) this change in distribution results in degradation in classification but improvement in segmentation tasks when using only the centroids; (3) to leverage additional Gaussian parameters, we propose Gaussian feature grouping in a normalized feature space, along with splats pooling layer, offering a tailored solution to effectively group and embed similar Gaussians, which leads to notable improvement in finetuning tasks.
</details>

<details>
  <summary>
  <font size="+1">中文摘要</font>
  </summary>
3D高斯光栅化（3DGS）已成为许多视觉任务中3D表示的事实标准。这需要在这种表示空间中直接进行3D理解。为推动该方向的研究，我们首先使用常用的ShapeNet和ModelNet数据集构建了一个大规模的3DGS数据集。我们的数据集ShapeSplat包含来自87个独特类别的65K个对象，其标签与各自的数据集保持一致。创建该数据集使用了相当于2个GPU年（在TITAN XP GPU上）的计算量。
我们利用这个数据集进行无监督预训练和有监督微调，以用于分类和分割任务。为此，我们引入了Gaussian-MAE，突出了从高斯参数进行表示学习的独特优势。通过详尽的实验，我们提供了几个有价值的见解。特别是，我们展示了：（1）优化后的GS中心的分布与均匀采样的点云（用于初始化）相比有显著差异；（2）这种分布变化在仅使用中心时导致分类任务的性能下降，但分割任务的性能提升；（3）为了利用额外的高斯参数，我们提出了在归一化特征空间中进行高斯特征分组，并结合散点池化层，提供了一种有效分组和嵌入相似高斯的定制解决方案，从而在微调任务中显著提升了性能。
</details>


## Datasets


## Installation



## Citation

If you find our work helpful, please consider citing the following paper and/or ⭐ the repo.
```
@misc{ma2024shapesplatlargescaledatasetgaussian,
      title={ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining}, 
      author={Qi Ma and Yue Li and Bin Ren and Nicu Sebe and Ender Konukoglu and Theo Gevers and Luc Van Gool and Danda Pani Paudel},
      year={2024},
      eprint={2408.10906},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10906}, 
}
```

## Acknowledgements

...

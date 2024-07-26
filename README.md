# CellMix
[[`Demo`](https://github.com/sagizty/CellMix/blob/main/CellMix%20Demo.ipynb)] [[`Paper`](https://arxiv.org/abs/2301.11513)] [[`BibTeX`](#Citation)]

## Introduction
Pathological image analysis, enhanced by deep learning, is critical for advancing diagnostic accuracy and improving patient outcomes. These images contain biomedical objects, or "instances," such as cells, tissues, and other structures across multiple scales. Their identities and spatial relationships significantly influence classification performance.

While current success heavily depends on data utilization, obtaining high-quality annotated pathological samples is extremely challenging. To overcome this challenge, data augmentation techniques generate pseudo-samples using mixing-based methods. However, these methods fail to fully consider the unique features of pathology images, such as local specificity, global distribution, and inner/outer-sample instance relationships.

Through mathematical exploration, we highlight the essence of shuffling to explicitly enhance the modeling of these instances. Accordingly, we introduce a novel, plug-and-play online data augmentation tool, CellMix, which explicitly augments instance relationships. Specifically, the input images are divided into patches based on the granularity of pathology instances, and the patches are in-place shuffled within the same batch. Thus, the absolute relationships among instances can be effectively preserved while new relationships can be further introduced. Moreover, to dynamically control task difficulty and explore multiple scales of instances, we incorporate a self-paced curriculum learning engine. This strategy enables the model to adaptively handle distribution-related noise and efficiently explore instances at various scales.

Extensive experiments on 11 pathological datasets, covering 8 diseases and 9 organs across 4 magnification scales, demonstrate state-of-the-art performance. Numerous ablation studies confirm its robust generalizability and scalability, providing novel insights into pathological image analysis and significant potential to enhance diagnostic precision. The proposed online data augmentation module is open-sourced as a plug-and-play tool to foster further research and clinical applications. It brings novel insights that potentially transform pathology image modeling approaches.

<img width="1505" alt="CellMix_Structure" src="https://github.com/user-attachments/assets/7a47c014-0e15-4fb8-94ac-feebbced8a47">
<img width="806" alt="LossDrive_Structure" src="https://github.com/user-attachments/assets/4b51c7e5-7d16-4989-9514-3c626a6b25ab">
![CAM_augmented](https://github.com/user-attachments/assets/9dc10221-ed09-4a01-9349-772f58d061b5)
<img width="1291" alt="CAM_augmented_Appendix" src="https://github.com/user-attachments/assets/3dccb8c3-de2e-4bba-a79b-ae54fc9537e8">

## USAGE
You can import the whole set of online data augmentation methods from [[`CellMix/utils/online_augmentations.py`](https://github.com/sagizty/CellMix/blob/main/utils/online_augmentations.py)]

augmentation = get_online_augmentation(augmentation_name, p=0.5, class_num=2, batch_size=4, edge_size=224, device='cpu')

and apply the augmentation in the training loop:
    augment_images, augment_labels, GT_labels = Augmentation(input_images, input_labels)

To apply the dynamicaly self-pased curriculum learning, you can refer to our training demo in [[`training`](https://github.com/sagizty/CellMix/blob/main/Train.py)]


# Citation
@article{zhang2023cellmix,
  title={CellMix: A General Instance Relationship based Method for Data Augmentation Towards Pathology Image Classification},
  author={Zhang, Tianyi and Yan, Zhiling and Li, Chunhui and Ying, Nan and Lei, Yanli and Feng, Yunlu and Zhao, Yu and Zhang, Guanglei},
  journal={arXiv preprint arXiv:2301.11513},
  year={2023}
}

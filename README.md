# CellMix

Pathology image analysis crucially relies on the availability and quality of annotated pathological samples, which are very difficult to collect and need lots of human effort. To address this issue, beyond traditional preprocess data augmentation methods, mixing-based approaches are effective and practical. However, previous mixing-based data augmentation methods do not thoroughly explore the essential characteristics of pathology images, including the local specificity, global distribution, and inner/outer-sample instance relationship. To further understand the pathology characteristics and make up effective pseudo samples, we propose the CellMix framework with a novel distribution-based in-place shuffle strategy. We split the images into patches with respect to the granularity of pathology instances and do the shuffle process across the same batch. In this way, we generate new samples while keeping the absolute relationship of pathology instances intact. Furthermore, to deal with the perturbations and distribution-based noise, we devise a loss-drive strategy inspired by curriculum learning during the training process, making the model fit the augmented data adaptively. It is worth mentioning that we are the first to explore data augmentation techniques in the pathology image field. Experiments show SOTA results on 7 different datasets. We conclude that this novel instance relationship-based strategy can shed light on general data augmentation for pathology image analysis. 

<img width="1133" alt="Screen Shot 2022-11-22 at 17 21 56" src="https://user-images.githubusercontent.com/50575108/203275968-4fbb99ea-558d-4eb4-beb2-d7019cfb13e2.png">

<img width="307" alt="Screen Shot 2022-11-22 at 17 22 12" src="https://user-images.githubusercontent.com/50575108/203275989-82410b2c-5add-4877-9877-799694f7712e.png">

<img width="644" alt="Screen Shot 2022-11-22 at 17 22 25" src="https://user-images.githubusercontent.com/50575108/203276007-9b3f52d2-ebce-447f-bce4-994d99f8c3f7.png">

<img width="644" alt="Screen Shot 2022-11-22 at 17 22 44" src="https://user-images.githubusercontent.com/50575108/203276029-858b9205-f219-4be7-a854-c05e1834fa66.png">

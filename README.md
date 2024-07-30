# CellMix
[[`pip`](https://pypi.org/project/CellMix/)] [[`Demo`](https://github.com/sagizty/CellMix/blob/main/CellMix%20Demo.ipynb)] [[`Paper`](https://arxiv.org/abs/2301.11513)] [[`BibTeX`](#Citation)]

## Introduction
Pathological image analysis, enhanced by deep learning, is critical for advancing diagnostic accuracy and improving patient outcomes. These images contain biomedical objects, or "instances," such as cells, tissues, and other structures across multiple scales. Their identities and spatial relationships significantly influence classification performance.

While current success heavily depends on data utilization, obtaining high-quality annotated pathological samples is extremely challenging. To overcome this challenge, data augmentation techniques generate pseudo-samples using mixing-based methods. However, these methods fail to fully consider the unique features of pathology images, such as local specificity, global distribution, and inner/outer-sample instance relationships.

Through mathematical exploration, we highlight the essence of shuffling to explicitly enhance the modeling of these instances. Accordingly, we introduce a novel, plug-and-play online data augmentation tool, CellMix, which explicitly augments instance relationships. Specifically, the input images are divided into patches based on the granularity of pathology instances, and the patches are in-place shuffled within the same batch. Thus, the absolute relationships among instances can be effectively preserved while new relationships can be further introduced. Moreover, to dynamically control task difficulty and explore multiple scales of instances, we incorporate a self-paced curriculum learning engine. This strategy enables the model to adaptively handle distribution-related noise and efficiently explore instances at various scales.

Extensive experiments on 11 pathological datasets, covering 8 diseases and 9 organs across 4 magnification scales, demonstrate state-of-the-art performance. Numerous ablation studies confirm its robust generalizability and scalability, providing novel insights into pathological image analysis and significant potential to enhance diagnostic precision. The proposed online data augmentation module is open-sourced as a plug-and-play tool to foster further research and clinical applications. It brings novel insights that potentially transform pathology image modeling approaches.

<img width="1505" alt="CellMix_Structure" src="https://github.com/user-attachments/assets/7a47c014-0e15-4fb8-94ac-feebbced8a47">

## USAGE (plug-and-play)
You can import the whole set from [[`pip`](https://pypi.org/project/CellMix/)]
```Shell
pip install CellMix
```

```Python
from CellMix.online_augmentations import get_online_augmentation
from CellMix.schedulers import ratio_scheduler, patch_scheduler
from SoftCrossEntropyLoss import SoftCrossEntropy
```
or download github repo from [[`plug-in`](https://github.com/sagizty/CellMix/blob/main/utils)]
```Python
from utils.online_augmentations import get_online_augmentation
from utils.schedulers import ratio_scheduler, patch_scheduler
from utils.SoftCrossEntropyLoss import SoftCrossEntropy
```

This is a pseudo-code demo for how to use CellMix online data augmentation

### STEP 1: Set up the Augmentation for triggering online data augmentation in training
```Python
Augmentation = get_online_augmentation(augmentation_name='CellMix',
                                       p=0.5,  # this is the triggering chance of activation
                                       class_num=2,
                                       batch_size=4,
                                       edge_size=224,
                                       device='cpu')
```
augmentation_name: name of data-augmentation method, this repo supports:
- CellMix (and the ablations)
- CutOut
- CutMix
- MixUp
- ResizeMix
- SaliencyMix
- FMix

<img width="1291" alt="CAM_augmented_Appendix" src="https://github.com/user-attachments/assets/3dccb8c3-de2e-4bba-a79b-ae54fc9537e8">

When the Augmentation is called, it will return three tensor: 
- augment_images (Batch, C, H, W), 
- augment_labels (Batch, Class_num) soft-label (expected confidence for each category)
- GT_long_labels (Batch) long-int tensor for classification recording (determind by the highest category in soft-label)

### STEP 2: Set Up the loss for learning, we use SoftCrossEntropy for classification task
```Python
loss_func = SoftCrossEntropy() # this one is CrossEntropy for soft-label
```

### STEP 3: Set Up the dynamic (self-paced curriclum learning) schedulers for Online Data Augmentation During Training
<img width="806" alt="LossDrive_Structure" src="https://github.com/user-attachments/assets/4b51c7e5-7d16-4989-9514-3c626a6b25ab">

#### Patch Strategy (default is 'loop'):
```python
puzzle_patch_size_scheduler = patch_scheduler(
    total_epochs=num_epochs,
    warmup_epochs=warmup_epochs,
    edge_size=224,
    basic_patch=16,
    strategy=patch_strategy,  # 'loop'
    threshold=loss_drive_threshold,
    fix_patch_size=None,  # Specify to fix to 16, 32, 48, 64, 96, 128, 192
    patch_size_jump=None  # Specify to 'odd' or 'even'
)
```
1. **linear**: 
   - Adjusts the patch size from small to large, managing the fix-position ratio plan after the warmup epochs.

2. **reverse**: 
   - Adjusts the patch size from large to small, managing the fix-position ratio plan after the warmup epochs.

3. **random**: 
   - Randomly chooses a specific patch size for each epoch.

4. **loop**: 
   - Tunes the patch size from small to large in a loop (e.g., a loop of 7 epochs through the patch size list), changing the patch size at most once every epoch.

5. **loss-driven** ('loss_hold' or 'loss_back'):
   - Follows the reverse method but fixes the patch size if the loss-driven strategy is activated. This maintains the shuffling with instances at the same scale, guiding the model to learn the same or more fixed patches, reducing complexity by introducing fewer outer-sample instances.


#### Ratio Strategy (default is 'loop'):
```python
fix_position_ratio_scheduler = ratio_scheduler(
    total_epochs=num_epochs,
    warmup_epochs=warmup_epochs,
    basic_ratio=0.5,
    strategy=ratio_strategy,  # 'linear'
    threshold=loss_drive_threshold,
    fix_position_ratio=None  # Specify to fix
)
```
1. **decay** ('decay' or 'ratio-decay'):
   - A basic curriculum plan that reduces the fix-position ratio linearly, managing the fix-position ratio plan after the warmup epochs.

2. **loss-driven** ('loss_hold' or 'loss_back'):
   - Dynamically adjusts the fix-position ratio based on the loss performance after the warmup epochs.
     - If the loss value `l` is less than the threshold `T`, indicating sufficient learning of the current complexity, the shuffling complexity is increased by reducing the fix-position ratio following the `ratio_floor_factor`.
     - If the loss value `l` exceeds the threshold `T`, indicating that the current complexity is too high, two strategies are employed:
       - **loss-hold**: Keeps the fix-position ratio unchanged in the next epoch, continuing with the current curriculum.
       - **loss-back**: Reduces complexity by setting the fix-position ratio 10% higher than the current curriculum plan.

This setup ensures that the augmentation strategies dynamically adapt to the training process, optimizing learning efficiency and performance.


### STEP 4: Apply the augmentations in the training loop:
```Python
if phase == 'train':
    # STEP 4.a. data augmentation
    # cellmix
    if fix_position_ratio_scheduler is not None and puzzle_patch_size_scheduler is not None:
        # epoch, epoch_loss is for the dynamic design in cellmix
        # epoch_loss is the average loss for each sample
        fix_position_ratio = fix_position_ratio_scheduler(epoch, epoch_loss)
        puzzle_patch_size = puzzle_patch_size_scheduler(epoch, epoch_loss)

        # inputs, labels is obtained from dataloader
        augment_images, augment_labels, GT_long_labels = Augmentation(inputs, labels,
                                                                      fix_position_ratio,
                                                                      puzzle_patch_size)
    # Counterpart augmentations
    else:
        augment_images, augment_labels, GT_long_labels = Augmentation(inputs, labels)

    # STEP 4.b. forward
    # track grad if only in train!
    with torch.set_grad_enabled(phase == 'train'):

        outputs = model(augment_images)  # pred outputs of confidence: [B,CLS]
        _, preds = torch.max(outputs, 1)  # idx outputs: [B] each is a idx
        loss = loss_func(outputs, augment_labels)  # cross entrphy of one-hot outputs: [B,CLS] and idx label [B]

        # STEP 4.b. log and backward
        # log criteria: update
        log_running_loss += loss.item()
        running_loss += loss.item() * augment_images.size(0)
        running_corrects += torch.sum(preds.cpu() == GT_long_labels.cpu().data)

        # backward + optimize only if in training phase
        if phase == 'train':
            loss.backward()
            optimizer.step()
```

### STEP 5: update the loss for loss-driven design to adjust the curriculum
```Python
epoch_loss = running_loss / dataset_sizes[phase]  # loss-per-sample at this epoch
```
To apply the dynamicaly self-pased curriculum learning, you can refer to our training demo in [[`training`](https://github.com/sagizty/CellMix/blob/main/Train.py)]

### To force-triggering the data augmentation (such as visulization), you can use act=True
```Python
augment_images, augment_labels, GT_long_labels = Augmentation(inputs, labels, act=True)
```

# Citation
@article{zhang2023cellmix,
  title={CellMix: A General Instance Relationship based Method for Data Augmentation Towards Pathology Image Classification},
  author={Zhang, Tianyi and Yan, Zhiling and Li, Chunhui and Ying, Nan and Lei, Yanli and Feng, Yunlu and Zhao, Yu and Zhang, Guanglei},
  journal={arXiv preprint arXiv:2301.11513},
  year={2023}
}

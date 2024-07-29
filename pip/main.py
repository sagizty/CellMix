# This is a pseudo-code demo for how to use CellMix online data augmentation
from online_augmentations import get_online_augmentation
from schedulers import ratio_scheduler, patch_scheduler
from SoftCrossEntropyLoss import SoftCrossEntropy


# STEP 1: Set up the Augmentation for triggering online data augmentation in training
Augmentation = get_online_augmentation(augmentation_name='CellMix',
                                       p=0.5,  # this is the triggering chance of activation
                                       class_num=2,
                                       batch_size=4,
                                       edge_size=224,
                                       device='cpu')
# when the Augmentation is called, it will return three tensor: 
# augment_images (Batch, C, H, W), 
# augment_labels (Batch, Class_num) soft-label (expected confidence for each category)
# GT_long_labels (Batch) long-int tensor for classification recording (determind by the highest category in soft-label)

# STEP 2: Set Up the loss for learning, we use SoftCrossEntropy for classification task
loss_func = SoftCrossEntropy() # this one is CrossEntropy for soft-label

# STEP 3: Set Up the dynamic (self-paced curriclum learning) schedulers for Online Data Augmentation During Training
puzzle_patch_size_scheduler = patch_scheduler(total_epochs=num_epochs,
                                              warmup_epochs=warmup_epochs,
                                              edge_size=224,
                                              basic_patch=16,
                                              strategy=patch_strategy,  # 'loop'
                                              threshold=loss_drive_threshold,
                                              fix_patch_size=None,  # specify to fix to 16,32,48,64,96,128,192
                                              patch_size_jump=None)  # specify to 'odd' or 'even'
'''
patch_strategy (default is 'loop')
        1.	linear: 
                This strategy is a basic curriculum adjusting the patch size from small to large.
                It manages the fix-position ratio plan in the epochs after the warmup_epochs.
        2.	reverse: 
                This strategy is a basic curriculum adjusting the patch size from large to small.
                It manages the fix-position ratio plan in the epochs after the warmup_epochs.
        3.	random: 
                This strategy involves randomly choosing a specific patch size for each epoch.
        4.	loop: 
                This strategy involves tuning the patch size from small to large in a loop
                (e.g. a loop of 7 epochs to go through the patch size list), changing the patch size
                at most once every epoch.
        5.	loss-driven ('loss_hold' or 'loss_back'):
                This strategy follows the reverse method. However, the patch size is
                fixed to the current value if the loss-driven strategy is activated for changing
                the fix-position ratio. It maintains the shuffling with the instances at the same scale.
                Therefore, if the loss-driven is activated for the schedulers, the model is guided to
                learn the same or more fixed patches. In this case the fix-ratio is fixed or increase,
                which become an easier complexity by introducing less outer-sample instances.
'''

fix_position_ratio_scheduler = ratio_scheduler(total_epochs=num_epochs,
                                               warmup_epochs=warmup_epochs,
                                               basic_ratio=0.5,
                                               strategy=ratio_strategy,  # 'linear'
                                               threshold=loss_drive_threshold,
                                               fix_position_ratio=None  # specify to fix
                                               )
'''
ratio_strategy (default is 'loop')
    1.	decay ('decay' or 'ratio-decay'):
        This strategy is a basic curriculum plan that reduce the fix-position-ratio linearly.
        It manages the fix-position ratio plan in the epochs after the warmup_epochs.
                    
    2.	loss-driven ('loss_hold' or 'loss_back'):
        This strategy dynamically adjusts the fix-position-ratio curriculum based on the
        loss performance on the current epoch after warmup_epochs.

        Firstly, When the loss value l is less than the given threshold T,
        it indicates that the model has sufficiently learned the current complexity.
        Such a case suggests that the current shuffling need to be more complex 
        than the current curriculum plan. Therefore, we increase difficulty by 
        reducing the fix-position ratio following the ratio_floor_factor.

        On the contrary, when the loss value l exceeds the expected threshold T, it suggests that 
        the current complexity is too hard for the model. We employ two strategies 
        ('loss_hold' or 'loss_back') to control the schedule. 
        The first is the loss-hold strategy, which keeps the fix-position ratio unchanged 
        in the next epoch and continues to learn the current curriculum. 
        The second is the loss-back strategy, which reduces the complexity by setting it 10% higher 
        than the current curriculum plan. 
'''

# STEP 4: Apply the augmentations in the training loop:
if phase == 'train':
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

# To force-triggering the data augmentation (such as visulization), you can use act=True
augment_images, augment_labels, GT_long_labels = Augmentation(inputs, labels, act=True)

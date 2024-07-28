"""
Schedulers   Script  ver： Jul 28th 13:00

patch_scheduler is used to regulate patch size for multi-scale learning

ratio_scheduler is used to regulate the complexity of curriculum learning:
    this fix-position ratio controls the percentage of patches being fixed in the shuffling!
        higher ratio -> more patches are fixed -> less patches are shuffled -> lower complexity -> easier to learn
        lower ratio -> less patches are fixed -> more patches are shuffled -> higher complexity -> harder to learn
"""

import math
import random


def factor(num):
    """
    find factor of input num
    """
    factors = []
    for_times = int(math.sqrt(num))
    for i in range(for_times + 1)[1:]:
        if num % i == 0:
            factors.append(i)
            t = int(num / i)
            if not t == i:
                factors.append(t)
    return factors


def defactor(num_list, basic_num):  # check multiples
    array = []
    for i in num_list:
        if i // basic_num * basic_num - i == 0:
            array.append(i)
    array.sort()  # accend
    return array


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup
    epoch，ok with float，to be more flexible，
    like: data_iter_step / len(data_loader) + epoch
    """
    # calculate the lr for this time
    if epoch < args.warmup_epochs:  # for warmup
        lr = args.lr * epoch / args.warmup_epochs  # lr increase from zero to the setted lr

    else:  # after warmup do cosin lr decay
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    # update lr in the optmizer
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class patch_scheduler:
    """
    this is used to drive the patch size by loss and epoch
    the patch list is automatically get
    """

    def __init__(self, total_epochs=200, warmup_epochs=20, edge_size=384, basic_patch=16, strategy=None,
                 threshold=4.0, loss_reducing_factor=0.933, fix_patch_size=None, patch_size_jump=None):
        """

        :param total_epochs:
        :param warmup_epochs:

        :param edge_size: image input size

        :param basic_patch: basic embedding patch for transformer (usually 16 or 14)
                            this helps the shuffling to maintain the embedding for each patch token

        :param strategy: curriculum plans of learning the different patch sizes
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

        :param threshold: a threshold for compare loss and decide how to adjust curriculum,
                          if None, it will be obtained by estimated from the warmup epochs

        :param loss_reducing_factor: a factor for reducing the threshold as the complexity should
                                     be increased with training

        :param fix_patch_size: if specified, fix the patch size to this value

        :param patch_size_jump: if specified, the list of patch size will be selected by 'odd' or 'even'
                                this is designed for ablation on the 'smoother learning' with more patch sizes
        """
        super().__init__()

        self.strategy = strategy

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        # automatically build legal patch list, from small to big size
        self.patch_list = defactor(factor(edge_size), basic_patch)

        self.threshold = threshold  # a threshold for compare loss and decide how to adjust curriculum
        self.loss_reducing_factor = loss_reducing_factor
        self.fix_patch_size = fix_patch_size

        # from small to big patch, No need for patch at all fig level
        if len(self.patch_list) > 1:
            self.patch_list = self.patch_list[:-1]

        # jump_patch_list by selecting the 'odd' or 'even', but both with the smallest patch size
        if patch_size_jump == 'odd':  # 384:[196, 96, 48, 16]
            jump_patch_list = self.patch_list[0::2]
            self.patch_list = jump_patch_list
        elif patch_size_jump == 'even':  # 384:[128, 64, 32, 16]
            jump_patch_list = self.patch_list[1::2]
            # add back the smallest
            temp_list = [self.patch_list[0]]
            temp_list.extend(jump_patch_list)
            self.patch_list = temp_list
        else:
            pass

        if self.strategy in ['reverse', 'loss_back', 'loss_hold']:  # start from big to small
            self.patch_list.sort(reverse=True)

        if self.strategy is None or self.strategy == 'fixed':
            puzzle_patch_size = self.fix_patch_size or self.patch_list[0]
            print('patch_list:', puzzle_patch_size)
        else:
            print('patch_list:', self.patch_list)

        # self.loss_log ?

    def __call__(self, epoch, loss=0.0):
        # Designed for flexable ablations
        if self.strategy == 'linear' or self.strategy == 'reverse':  # reverse from big size to small
            if epoch < self.warmup_epochs:  # warmup
                puzzle_patch_size = 32  # fixed size for warmup
            else:
                puzzle_patch_size = self.patch_list[min(int((epoch - self.warmup_epochs)
                                                            / (self.total_epochs - self.warmup_epochs)
                                                            * len(self.patch_list)), len(self.patch_list) - 1)]

        elif self.strategy == 'loop':
            # looply change the patch size, after [group_size] epoches we change once
            group_size = 3

            if epoch < self.warmup_epochs:
                puzzle_patch_size = 32  # in warm up epoches, fixed patch size at 32 fixme exploring
            else:
                group_idx = (epoch - self.warmup_epochs) % (len(self.patch_list) * group_size)
                puzzle_patch_size = self.patch_list[int(group_idx / group_size)]

        elif self.strategy == 'random':  # random size strategy
            puzzle_patch_size = random.choice(self.patch_list)

        elif self.strategy == 'loss_back':
            if epoch < self.warmup_epochs:  # for warmup
                puzzle_patch_size = 32  # in warm-up we use the fix size
            else:
                if loss == 0.0:
                    puzzle_patch_size = self.patch_list[min(int((epoch - self.warmup_epochs)
                                                                / (self.total_epochs - self.warmup_epochs)
                                                                * len(self.patch_list)), len(self.patch_list) - 1)]

                elif loss < self.threshold:
                    puzzle_patch_size = self.patch_list[min(max(int((epoch - self.warmup_epochs)
                                                                    / (self.total_epochs - self.warmup_epochs)
                                                                    * len(self.patch_list)) + 1, 0),
                                                            len(self.patch_list) - 1)]
                    self.threshold *= self.loss_reducing_factor
                else:
                    puzzle_patch_size = self.patch_list[min(max(int((epoch - self.warmup_epochs)
                                                                    / (self.total_epochs - self.warmup_epochs)
                                                                    * len(self.patch_list)) - 1, 0),
                                                            len(self.patch_list) - 1)]

        elif self.strategy == 'loss_hold':
            if epoch < self.warmup_epochs:  # for warmup
                puzzle_patch_size = 32  # in warm-up we use the fix size
            else:
                if loss == 0.0:
                    puzzle_patch_size = self.patch_list[min(int((epoch - self.warmup_epochs)
                                                                / (self.total_epochs - self.warmup_epochs)
                                                                * len(self.patch_list)), len(self.patch_list) - 1)]

                elif loss < self.threshold:
                    puzzle_patch_size = self.patch_list[min(max(int((epoch - self.warmup_epochs)
                                                                    / (self.total_epochs - self.warmup_epochs)
                                                                    * len(self.patch_list)) + 1, 0),
                                                            len(self.patch_list) - 1)]
                    self.threshold *= self.loss_reducing_factor
                else:
                    puzzle_patch_size = self.patch_list[min(max(int((epoch - self.warmup_epochs)
                                                                    / (self.total_epochs - self.warmup_epochs)
                                                                    * len(self.patch_list)), 0),
                                                            len(self.patch_list) - 1)]

        else:
            # if self.strategy is None or 'fixed' or 'ratio-decay'
            puzzle_patch_size = self.fix_patch_size or self.patch_list[0]  # basic_patch

        return puzzle_patch_size


class ratio_scheduler:
    """
    ratio_scheduler is used to regulate the complexity of curriculum learning:
    this fix-position ratio controls the percentage of patches being fixed in the shuffling!
        higher ratio -> more patches are fixed -> less patches are shuffled -> lower complexity -> easier to learn
        lower ratio -> less patches are fixed -> more patches are shuffled -> higher complexity -> harder to learn

    the scheduler is used to drive the fix position ratio by loss and epoch
        the ratio is control by ratio_floor_factor=0.5, upper_limit=0.9, lower_limit=0.2
    """
    def __init__(self, total_epochs=200, warmup_epochs=20, basic_ratio=0.25, strategy=None, threshold=4.0,
                 fix_position_ratio=None, loss_reducing_factor=0.933, ratio_floor_factor=0.5,
                 upper_limit=0.9, lower_limit=0.2):
        """

        :param total_epochs:
        :param warmup_epochs:

        :param basic_ratio: basic ratio of fixed patches in learning with augmented samples

        :param strategy: loss-driven strategy for dynamic adjusting the complexity of shuffling
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

        :param threshold: a threshold for compare loss and decide how to adjust curriculum,
                          if None, it will be obtained by estimated from the warmup epochs

        :param fix_position_ratio: if specified, set the ratio to this value instead of dynamic adjusting

        :param loss_reducing_factor: a factor for reducing the threshold as the complexity should
                                     be increased with training

        :param ratio_floor_factor: the reducing factor for adjusting the fix_position_ratio curriculum,
                                   as the complexity should be increased with training

        :param upper_limit: the lower_limit value of adjusting the fix_position_ratio
        :param lower_limit: the lower_limit value of adjusting the fix_position_ratio
        """

        # fixme basic_ratio and fix_position_ratio(when stage is fixed) is a bit conflicting, not good enough
        super().__init__()
        self.strategy = strategy

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        self.basic_ratio = basic_ratio

        self.threshold = threshold
        self.loss_reducing_factor = loss_reducing_factor

        self.fix_position_ratio = fix_position_ratio

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.ratio_floor_factor = ratio_floor_factor

    def __call__(self, epoch, loss=0.0):
        if self.strategy == 'ratio-decay' or self.strategy == 'decay':
            if epoch < self.warmup_epochs:  # for warmup
                fix_position_ratio = self.basic_ratio  # fixed
            else:
                max_ratio = min(3 * self.basic_ratio, self.upper_limit)  # upper-limit of 0.9
                min_ratio = max(self.basic_ratio * self.ratio_floor_factor, self.lower_limit)

                fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                              - (epoch - self.warmup_epochs)) /
                                             (self.total_epochs - self.warmup_epochs)
                                             * max_ratio, min_ratio), max_ratio)

        elif self.strategy == 'loss_back':

            if epoch < self.warmup_epochs:  # for warmup
                fix_position_ratio = self.basic_ratio  # in warm-up we use the fix ratio

            else:
                max_ratio = min(3 * self.basic_ratio, self.upper_limit)
                min_ratio = max(self.basic_ratio * self.ratio_floor_factor, self.lower_limit)
                if loss == 0.0:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio, min_ratio), max_ratio)
                elif loss < self.threshold:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio * 0.9, min_ratio), max_ratio)
                    self.threshold *= self.loss_reducing_factor
                else:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio * 1.1, min_ratio), max_ratio)

        elif self.strategy == 'loss_hold':

            if epoch < self.warmup_epochs:  # for warmup
                fix_position_ratio = self.basic_ratio  # in warm-up we use the fix ratio

            else:
                max_ratio = min(3 * self.basic_ratio, self.upper_limit)
                min_ratio = max(self.basic_ratio * self.ratio_floor_factor, self.lower_limit)

                if loss == 0.0:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio, min_ratio), max_ratio)
                elif loss < self.threshold:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio * 0.9, min_ratio), max_ratio)
                    self.threshold *= self.loss_reducing_factor
                else:
                    fix_position_ratio = min(max(((self.total_epochs - self.warmup_epochs)
                                                  - (epoch - self.warmup_epochs)) /
                                                 (self.total_epochs - self.warmup_epochs)
                                                 * max_ratio, min_ratio), max_ratio)

        else:  # basic_ratio
            fix_position_ratio = self.fix_position_ratio or self.basic_ratio

        return fix_position_ratio

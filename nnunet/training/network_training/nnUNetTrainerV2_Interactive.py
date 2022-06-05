#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from torch.cuda.amp import autocast
import numpy as np
import torch
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset_from_list
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import load_dataset_from_list
from torch import nn
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class nnUNetTrainerV2_Interactive(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, args=None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.interact_samples = args.num_samples
        self.slice_ratio = np.clip(args.slice_ratio,a_min=0.,a_max = 1.)
        self.nointeract = args.nointeract
        self.fixiteration = args.fixiteration
        self.sample_dataset = args.sample_dataset
        self.num_training_sample = 50
        self.num_val_sample = int(self.num_training_sample / 4)

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = [join(self.dataset_directory, self.plans['data_identifier'] +
                                                       f"_sample{i}_ratio{self.slice_ratio}_stage{self.stage}") for i in
                                                  range(self.interact_samples)]
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset_from_list(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def load_dataset(self):
        self.dataset = load_dataset_from_list(self.folder_with_preprocessed_data)

    def do_split(self):
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            if not self.sample_dataset:
                splits_file = join(self.dataset_directory, "splits_last.pkl")
            else:
                splits_file = join(self.dataset_directory, f"splits_sample_{self.num_training_sample}.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                num_train = int(len(all_keys_sorted)*4/5)
                train_keys = np.array(all_keys_sorted)[:num_train]
                test_keys = np.array(all_keys_sorted)[num_train:]

                if self.sample_dataset:
                    train_keys = np.random.choice(train_keys, size=self.num_training_sample,replace = False) \
                        if num_train > self.num_training_sample else train_keys
                    test_keys = np.random.choice(test_keys, size=self.num_val_sample,replace = False) \
                        if len(test_keys) > self.num_val_sample else test_keys

                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                raise ValueError(f'Folder {self.fold} does not equal to zero!')

            self.num_batches_per_epoch = min(int(len(tr_keys) / self.batch_size),self.num_batches_per_epoch)

            if self.fixiteration:
                self.num_batches_per_epoch = 16
                self.current_max_num_epochs = int(self.fixiteration / self.num_batches_per_epoch)
                self.max_num_epochs = self.current_max_num_epochs
            self.num_val_batches_per_epoch = int(len(val_keys) / self.batch_size)

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        print(tr_keys)
        print(val_keys)
        for i in tr_keys:
            for s in range(self.interact_samples):
                index = i + '_' + str(s)
                self.dataset_tr[index] = self.dataset[i][s]
                self.dataset[index] = self.dataset[i][s]
            del self.dataset[i]
        self.dataset_val = OrderedDict()
        s = np.random.choice(self.interact_samples, size=len(val_keys))
        for i, j in enumerate(val_keys):
            self.dataset_val[j] = self.dataset[j][s[i]]
            del self.dataset[j]
            self.dataset[j] = self.dataset_val[j]

    def get_features(self):
        features_all = OrderedDict()
        with torch.no_grad():
            self.network.eval()
            for b in range(self.num_val_batches_per_epoch):
                features_dict = self.sample_feature(self.val_gen)
                for k,feature in features_dict.items():
                    if k in features_all.keys():
                        features_all[k] = np.concatenate((features_all[k],feature),axis = 0)
                    else:
                        features_all[k] = feature
        return features_all

    def sample_feature(self, data_generator):
        data_dict = next(data_generator)
        data = data_dict['data']

        data = maybe_to_torch(data)

        if torch.cuda.is_available():
            data = to_cuda(data)

        if self.fp16:
            with autocast():
                features_dict = self.network.get_feature_map(data)
                del data
        else:
            features_dict = self.network.get_feature_map(data)
            del data

        sampled_features_dict = OrderedDict()
        for key, features in features_dict.items():
            B, C = features.size(0), features.size(1)
            features = features.detach().view(B, C, -1).cpu().numpy()
            spatial_size = features.shape[-1]

            random.seed(0)
            k = min(spatial_size, 500)
            indices = random.sample(range(spatial_size), k=k)
            sampled_features = features[:, :, indices]
            sampled_features = np.reshape(np.transpose(sampled_features, (0, 2, 1)), (-1, C))
            sampled_features_dict[key] = sampled_features
        return sampled_features_dict

    def plot_cluster(self, x, y, level = None):
        # choose a color palette with seaborn
        num_classes = len(np.unique(y))
        palette = np.array(sns.color_palette("hls", num_classes))

        # create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')

        labels = np.unique(y)
        for i,label in enumerate(labels):
            x_i = x[y==label]
            lbl = f'task {int(label)}' if label else 'current task'
            ax.scatter(x_i[:, 0], x_i[:, 1], lw=0, s=40, color=palette[i],label = lbl)

        ax.axis('off')
        ax.legend()
        # ax.axis('tight')
        fig_name = f"tsne_{level}.png"
        f.savefig(join(self.output_folder, fig_name))
        plt.close()

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
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from _warnings import warn
from tqdm import trange
from time import time, sleep
from datetime import datetime
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset_from_list
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import load_dataset_from_list
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
import torch.backends.cudnn as cudnn
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from torch import nn


class nnUNetTrainerV2_MultiDatasets(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, fold, trainer_task_dict, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, args=None):
        super().__init__(None, fold, '', None, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        del self.output_folder
        self.interact_samples = args.num_samples
        self.slice_ratio = np.clip(args.slice_ratio,a_min=0.,a_max = 1.)
        self.nointeract = args.nointeract
        self.sample_dataset = args.sample_dataset
        self.num_training_sample = 50
        self.num_val_sample = int(self.num_training_sample / 4)

        self.trainer_task_dict = trainer_task_dict

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
            self.output_folder_base_dict = OrderedDict()
            self.output_folder_dict = OrderedDict()
            self.plans_dict = OrderedDict()
            self.plans_file_dict = OrderedDict()
            self.patch_size_dict = OrderedDict()
            self.transpose_forward_dict = OrderedDict()
            self.transpose_backward_dict = OrderedDict()
            self.plans_loaded = False

            for taskid in self.trainer_task_dict.keys():
                output_folder = self.trainer_task_dict[taskid]['output_folder']
                maybe_mkdir_p(output_folder)
                self.output_folder_base_dict[taskid] = self.trainer_task_dict[taskid]['output_folder_base']
                self.output_folder_dict[taskid] = output_folder

                plans_file = self.trainer_task_dict[taskid]['plans_file']
                plans = load_pickle(plans_file)

                self.plans_dict[taskid] = plans
                self.plans_file_dict[taskid] = plans_file
                # use the plans of first task except patch_size,transpose_forward,transpose_backward
                if not self.plans_loaded:
                    self.process_plans(plans)
                    self.plans = None
                    self.plans_loaded = True
                    del self.patch_size
                    del self.transpose_forward
                    del self.transpose_backward

                # what stored in the plan should be used?
                # intensity_properties? patch_size,transpose_forward,transpose_backward
                self.patch_size_dict[taskid] = np.array(plans['plans_per_stage'][self.stage]['patch_size']).astype(int)
                self.transpose_forward_dict[taskid] = plans['transpose_forward']
                self.transpose_backward_dict[taskid] = plans['transpose_backward']

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

            self.folder_with_preprocessed_data_dict = OrderedDict()
            for taskid in self.trainer_task_dict.keys():
                self.folder_with_preprocessed_data_dict[taskid] = \
                    [join(self.trainer_task_dict[taskid]['dataset_directory'], self.plans_dict[taskid]['data_identifier'] +
                        f"_sample{i}_ratio{self.slice_ratio}_stage{self.stage}") for i in range(self.interact_samples)]

            if training:
                self.dl_tr_dict, self.dl_val_dict = self.get_basic_generators()
                self.tr_gen_dict = OrderedDict()
                self.val_gen_dict = OrderedDict()
                self.thread_division_factor = len(list(self.trainer_task_dict.keys()))
                for taskid in self.trainer_task_dict.keys():
                    if self.unpack_data:
                        print("unpacking dataset")
                        unpack_dataset_from_list(self.folder_with_preprocessed_data_dict[taskid])
                        print("done")
                    else:
                        print(
                            "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                            "will wait all winter for your model to finish!")

                    self.tr_gen_dict[taskid], self.val_gen_dict[taskid] = get_moreDA_augmentation(
                        self.dl_tr_dict[taskid], self.dl_val_dict[taskid],
                        self.data_aug_params_dict[taskid][
                            'patch_size_for_spatialtransform'],
                        self.data_aug_params_dict[taskid],
                        deep_supervision_scales=self.deep_supervision_scales,
                        pin_memory=self.pin_memory,
                        use_nondetMultiThreadedAugmenter=False,
                        thread_division_factor=self.thread_division_factor,
                        thread_division_factor_train=self.thread_division_factor
                    )

            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))

            self.all_decoder_state_dict = OrderedDict()
            for taskid in self.trainer_task_dict.keys():
                self.all_decoder_state_dict[taskid] = OrderedDict()
                for k,value in self.network.state_dict().items():
                    if k.startswith('seg_outputs'):
                        self.all_decoder_state_dict[taskid][k] = value

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            raise NotImplementedError

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]

        self.data_aug_params["num_cached_per_thread"] = 2

        self.data_aug_params_dict = OrderedDict()
        self.basic_generator_patch_size_dict = OrderedDict()

        for taskid in self.trainer_task_dict.keys():
            self.data_aug_params_dict[taskid] = OrderedDict()
            self.data_aug_params_dict[taskid].update(self.data_aug_params)
            patch_size = self.patch_size_dict[taskid]
            self.data_aug_params_dict[taskid]['patch_size_for_spatialtransform'] = patch_size

            if self.do_dummy_2D_aug:
                self.basic_generator_patch_size_dict[taskid] = get_patch_size(patch_size[1:],
                                                                 self.data_aug_params['rotation_x'],
                                                                 self.data_aug_params['rotation_y'],
                                                                 self.data_aug_params['rotation_z'],
                                                                 self.data_aug_params['scale_range'])
                self.basic_generator_patch_size_dict[taskid] = np.array([patch_size[0]] + list(self.basic_generator_patch_size))
            else:
                self.basic_generator_patch_size_dict[taskid] = get_patch_size(patch_size, self.data_aug_params['rotation_x'],
                                                                 self.data_aug_params['rotation_y'],
                                                                 self.data_aug_params['rotation_z'],
                                                                 self.data_aug_params['scale_range'])
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        dl_tr_dict = OrderedDict()
        dl_val_dict = OrderedDict()

        for taskid in self.trainer_task_dict.keys():
            if self.threeD:
                dl_tr = DataLoader3D(self.dataset_tr_dict[taskid], self.basic_generator_patch_size_dict[taskid],
                                     self.patch_size_dict[taskid], self.batch_size,
                                     False, oversample_foreground_percent=self.oversample_foreground_percent,
                                     pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader3D(self.dataset_val_dict[taskid], self.patch_size_dict[taskid],
                                      self.patch_size_dict[taskid], self.batch_size, False,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            else:
                dl_tr = DataLoader2D(self.dataset_tr_dict[taskid], self.basic_generator_patch_size_dict[taskid],
                                     self.patch_size_dict[taskid], self.batch_size,
                                     False, oversample_foreground_percent=self.oversample_foreground_percent,
                                     pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
                dl_val = DataLoader2D(self.dataset_val_dict[taskid], self.patch_size_dict[taskid],
                                      self.patch_size_dict[taskid], self.batch_size, False,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_dict[taskid] = dl_tr
            dl_val_dict[taskid] = dl_val
        return dl_tr_dict, dl_val_dict

    def load_dataset(self):
        self.dataset_dict = OrderedDict()
        for taskid in self.trainer_task_dict.keys():
            self.dataset_dict[taskid] = load_dataset_from_list(self.folder_with_preprocessed_data_dict[taskid])

    def do_split(self):
        self.dataset_tr_dict = OrderedDict()
        self.dataset_val_dict = OrderedDict()
        self.num_batches_per_epoch_dict = OrderedDict()
        self.num_val_batches_per_epoch_dict = OrderedDict()

        for taskid in self.trainer_task_dict.keys():
            if self.fold == "all":
                # if fold==all then we use all images for training and validation
                tr_keys = val_keys = list(self.dataset[taskid].keys())
            else:
                if not self.sample_dataset:
                    splits_file = join(self.trainer_task_dict[taskid]['dataset_directory'], "splits_last.pkl")
                else:
                    splits_file = join(self.trainer_task_dict[taskid]['dataset_directory'],
                                       f"splits_sample_{self.num_training_sample}.pkl")

                # if the split file does not exist we need to create it
                if not isfile(splits_file):
                    self.print_to_log_file("Creating new 5-fold cross-validation split...")
                    splits = []
                    all_keys_sorted = np.sort(list(self.dataset_dict[taskid].keys()))
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

                self.num_batches_per_epoch_dict[taskid] = int(len(tr_keys) / self.batch_size)
                self.num_val_batches_per_epoch_dict[taskid] = int(len(val_keys) / self.batch_size)

            tr_keys.sort()
            val_keys.sort()
            self.dataset_tr_dict[taskid] = OrderedDict()
            print(tr_keys)
            print(val_keys)
            for i in tr_keys:
                for s in range(self.interact_samples):
                    index = i + '_' + str(s)
                    self.dataset_tr_dict[taskid][index] = self.dataset_dict[taskid][i][s]
                    self.dataset_dict[taskid][index] = self.dataset_dict[taskid][i][s]
                del self.dataset_dict[taskid][i]

            self.dataset_val_dict[taskid] = OrderedDict()
            s = np.random.choice(self.interact_samples, size=len(val_keys))
            for i, j in enumerate(val_keys):
                self.dataset_val_dict[taskid][j] = self.dataset_dict[taskid][j][s[i]]
                del self.dataset_dict[taskid][j]
                self.dataset_dict[taskid][j] = self.dataset_val_dict[taskid][j]

    def load_latest_checkpoint(self, train=True):
        if not hasattr(self,'all_decoder_state_dict'):
            self.all_decoder_state_dict = OrderedDict()

        for taskid in self.trainer_task_dict.keys():
            output_folder = self.output_folder_dict[taskid]
            if isfile(join(output_folder, "model_final_checkpoint.model")):
                self.load_checkpoint(join(output_folder, "model_final_checkpoint.model"), train=train)
            elif isfile(join(output_folder, "model_latest.model")):
                self.load_checkpoint(join(output_folder, "model_latest.model"), train=train)
            else:
                raise RuntimeError("No checkpoint found")

            decoder_state_dict = OrderedDict()
            for k, value in self.network.state_dict().items():
                if k.startswith('seg_outputs'):
                    decoder_state_dict[k] = value
            self.print_to_log_file(f'save decoder {decoder_state_dict.keys()} in the model')
            self.all_decoder_state_dict[taskid] = decoder_state_dict

    def load_best_checkpoint(self, train=True):

        if not hasattr(self, 'all_decoder_state_dict'):
            self.all_decoder_state_dict = OrderedDict()
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        for taskid in self.trainer_task_dict.keys():
            output_folder = self.output_folder_dict[taskid]
            if isfile(join(output_folder, "model_best.model")):
                self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)

                decoder_state_dict = OrderedDict()
                for k, value in self.network.state_dict().items():
                    if k.startswith('seg_outputs'):
                        decoder_state_dict[k] = value
                self.print_to_log_file(f'save decoder {decoder_state_dict.keys()} in the model')
                self.all_decoder_state_dict[taskid] = decoder_state_dict
                return
            else:
                self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                       "back to load_latest_checkpoint")
        self.load_latest_checkpoint(train)

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans_dict']
        del dct['intensity_properties']
        del dct['dataset_dict']
        del dct['dataset_tr_dict']
        del dct['dataset_val_dict']
        if 'all_decoder_state_dict' in dct:
            del dct['all_decoder_state_dict']

        for taskid in self.trainer_task_dict.keys():
            output_folder = self.output_folder_dict[taskid]
            save_json(dct, join(output_folder, "debug.json"))
            import shutil
            plans_file = self.plans_file_dict[taskid]
            output_folder_base = self.output_folder_base_dict[taskid]
            shutil.copy(plans_file, join(output_folder_base, "plans.pkl"))

    def update_decoder_dict(self,taskid):
        for k in self.all_decoder_state_dict[taskid].keys():
            self.all_decoder_state_dict[taskid][k] = self.network.state_dict()[k]
        self.print_to_log_file(f'Updating decoder state dict '
                               f'{self.all_decoder_state_dict[taskid].keys()} for task {taskid}')

    def load_decoder_dict(self,taskid):
        decoder_state_dict = self.all_decoder_state_dict[taskid]
        current_state_dict = self.network.state_dict()
        current_state_dict.update(decoder_state_dict)
        self.network.load_state_dict(current_state_dict)

    def run_training_multi_datasets(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        if not hasattr(self,'all_val_eval_metrics_dict'):
            self.all_val_eval_metrics_dict = OrderedDict()
            self.print_to_log_file('creating all the val metric dict...')
            for taskid in self.trainer_task_dict.keys():
                self.all_val_eval_metrics_dict[taskid] = []

        for taskid in self.trainer_task_dict.keys():
            _ = self.tr_gen_dict[taskid].next()
            _ = self.val_gen_dict[taskid].next()
            maybe_mkdir_p(self.output_folder_dict[taskid])


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.current_max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            for taskid in self.trainer_task_dict.keys():
                self.load_decoder_dict(taskid)
                self.print_to_log_file(f'Load the decoder of task {taskid} for training')

                tr_gen = self.tr_gen_dict[taskid]
                # train one epoch
                self.network.train()
                if self.use_progress_bar:
                    with trange(self.num_batches_per_epoch_dict[taskid]) as tbar:
                        for b in tbar:
                            tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.current_max_num_epochs))

                            l = self.run_iteration(tr_gen, True)

                            tbar.set_postfix(loss=l)
                            train_losses_epoch.append(l)
                else:
                    for _ in range(self.num_batches_per_epoch_dict[taskid]):
                        l = self.run_iteration(tr_gen, True)
                        train_losses_epoch.append(l)

                self.update_decoder_dict(taskid)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                for taskid in self.trainer_task_dict.keys():
                    self.load_decoder_dict(taskid)
                    self.print_to_log_file(f'Load the decoder of task {taskid} for evaluation')

                    val_gen = self.val_gen_dict[taskid]
                    self.network.eval()
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch_dict[taskid]):
                        l = self.run_iteration(val_gen, False, True)
                        val_losses.append(l)
                    self.finish_online_evaluation_on_current_dataset(taskid)

                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.
        # add codes here for continual learning

        if self.save_final_checkpoint:
            for taskid in self.trainer_task_dict.keys():
                output_folder = self.output_folder_dict[taskid]
                self.load_decoder_dict(taskid)
                self.print_to_log_file(f'Load the decoder of task {taskid} to save checkpoint')
                self.save_checkpoint(join(output_folder, "model_final_checkpoint.model"))
            # now we can delete latest as it will be identical with final
            if isfile(join(output_folder, "model_latest.model")):
                os.remove(join(output_folder, "model_latest.model"))
            if isfile(join(output_folder, "model_latest.model.pkl")):
                os.remove(join(output_folder, "model_latest.model.pkl"))

    def on_epoch_end(self):
        self.plot_progress_all_tasks()

        self.update_mean_metric()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training

    def finish_online_evaluation_on_current_dataset(self,task):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics_dict[task].append(np.mean(global_dc_per_class))

        self.print_to_log_file(f"Average global foreground Dice of task {task}:", [np.round(i, 4) for i in global_dc_per_class])

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def plot_progress_all_tasks(self):
        """
        Should probably by improved
        :return:
        """

        colors = ['lightseagreen','g','coral','c','m','y']

        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            for i,taskid in enumerate(self.all_val_eval_metrics_dict.keys()):
                assert len(self.all_val_eval_metrics_dict[taskid]) == len(x_values)
                ax2.plot(x_values, self.all_val_eval_metrics_dict[taskid],
                        color=colors[i], ls='-', label=f"task_{taskid}")


            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=4)
            output_folder = list(self.output_folder_dict.values())[0]
            fig.savefig(join(output_folder, "progress_tasks.png"))

            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            for taskid in self.trainer_task_dict.keys():
                output_folder = self.output_folder_dict[taskid]
                self.load_decoder_dict(taskid)
                self.print_to_log_file(f'Load the decoder of task {taskid} to save checkpoint')
                if not self.save_latest_only:
                    self.save_checkpoint(join(output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
                self.save_checkpoint(join(output_folder, "model_latest.model"))
                self.print_to_log_file("done")

    def update_mean_metric(self):
        last_metrics = []
        for taskid in self.trainer_task_dict.keys():
            last_metrics.append(self.all_val_eval_metrics_dict[taskid][-1])
        self.all_val_eval_metrics.append(sum(last_metrics)/len(last_metrics))
        self.print_to_log_file(f"Average global foreground Dice:",self.all_val_eval_metrics[-1])

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")

                if self.save_best_checkpoint:
                    for taskid in self.trainer_task_dict.keys():
                        output_folder = self.output_folder_dict[taskid]
                        self.load_decoder_dict(taskid)
                        self.print_to_log_file(f'Load the decoder of task {taskid} to save checkpoint')
                        self.save_checkpoint(join(output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            output_folder = list(self.output_folder_dict.values())[0]
            maybe_mkdir_p(output_folder)
            timestamp = datetime.now()
            self.log_file = join(output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))


            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

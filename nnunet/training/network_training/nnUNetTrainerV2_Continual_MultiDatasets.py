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
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation,get_moreDA_augmentation_only_val
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset_from_list
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import load_dataset_from_list
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.dataloading.dataset_loading import delete_npy
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.quant_layers_for_blip import Conv2d_Q,Conv3d_Q
import math
from nnunet.utilities.tensor_utilities import sum_tensor

from torch import nn
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


class nnUNetTrainerV2_Continual_MultiDatasets(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, args=None, previous_task_dict = None):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        # interactive learning related (not used)
        self.interact_samples = args.num_samples
        self.slice_ratio = np.clip(args.slice_ratio, a_min=0., a_max=1.)
        self.nointeract = args.nointeract

        self.previous_task_dict = previous_task_dict
        self.exp_name = args.exp_name
        self.sample_dataset = args.sample_dataset
        self.num_training_sample = 50
        self.num_val_sample = int(self.num_training_sample / 4)

        self.method = args.method
        # EWC related
        self.lambda_ewc = args.lambda_ewc
        self.fisher_storage = args.fisher_storage
        # BLIP related
        self.fisher_prior = args.fisher_prior


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

                self.thread_division_factor = len(self.previous_task_dict.keys()) + 1
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                    thread_division_factor = self.thread_division_factor,
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)

                if len(self.previous_task_dict.keys()):
                    self.load_previous_validation_set()
            else:
                pass

            if self.method == 'BLIP':
                self.initialize_network_blip()
            else:
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
                num_train = int(len(all_keys_sorted) * 4 / 5)
                train_keys = np.array(all_keys_sorted)[:num_train]
                test_keys = np.array(all_keys_sorted)[num_train:]

                if self.sample_dataset:
                    train_keys = np.random.choice(train_keys,size = self.num_training_sample, replace = False) \
                        if num_train > self.num_training_sample else train_keys
                    test_keys = np.random.choice(test_keys,size = self.num_val_sample, replace = False) \
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

            self.num_batches_per_epoch = int(len(tr_keys) / self.batch_size)
            self.num_val_batches_per_epoch = int(len(val_keys) / self.batch_size)

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
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

    def load_best_checkpoint(self, output_folder, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(output_folder, "model_best.model")):
            fname = join(output_folder, "model_best.model")
        elif isfile(join(output_folder, "model_final_checkpoint.model")):
            # self.load_checkpoint(join(self.output_folder, f"model_best_episode{episode}.model"), train=train)
            fname = join(output_folder, "model_final_checkpoint.model")
        else:
            raise ValueError(f"No checkpoint found in {output_folder}")

        self.print_to_log_file("Only loading state_dict from checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)


    def load_decoder(self,output_folder, train = False):
        if isfile(join(output_folder,"model_best.model")):
            fname = join(output_folder, "model_best.model")
        elif isfile(join(output_folder,"model_final_checkpoint.model")):
            fname = join(output_folder, "model_final_checkpoint.model")
        else:
            raise ValueError("No checkpoint found for loading decoder")

        self.print_to_log_file("Only load decoder from checkpoint", fname, "train=", train)
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        current_state_dict = self.network.state_dict()
        for k,value in saved_model['state_dict'].items():
            if k.startswith('seg_outputs'):
                new_state_dict[k] = value
        current_state_dict.update(new_state_dict)
        self.network.load_state_dict(current_state_dict)
        self.print_to_log_file(f'{new_state_dict.keys()} get updated in the network state dict')

    def save_current_decoder(self):
        self.decoder_state_dict = OrderedDict()
        for k,value in self.network.state_dict().items():
            if k.startswith('seg_outputs'):
                self.decoder_state_dict[k] = value
        self.print_to_log_file(f'Storing {self.decoder_state_dict.keys()}')

    def load_current_decoder(self):
        current_state_dict = self.network.state_dict()
        current_state_dict.update(self.decoder_state_dict)
        self.network.load_state_dict(current_state_dict)
        self.print_to_log_file('Recovering the decoder state dict for the current dataset')

    def load_latest_checkpoint(self, output_folder, train=True):
        if isfile(join(output_folder, f"model_final_checkpoint.model")):
            return self.load_checkpoint(join(output_folder, f"model_final_checkpoint.model"),
                                        train=train)
        if isfile(join(output_folder, f"model_latest.model")):
            return self.load_checkpoint(join(output_folder, f"model_latest.model"), train=train)
        raise RuntimeError("No checkpoint found")

    def load_last_dataset_checkpoint(self, train=True):
        last_task_id = list(self.previous_task_dict.keys())[-1]
        last_output_folder = self.previous_task_dict[last_task_id]['output_folder']
        if not self.was_initialized:
            self.initialize(train)
        self.load_best_checkpoint(last_output_folder)
        # reinitialize decoder
        weight_initilizer = InitWeights_He()
        for name,module in self.network.named_modules():
            if name.startswith('seg_outputs'):
                self.print_to_log_file(f'{name} gets reinitialized')
                module.apply(weight_initilizer)

    def EWC_load_previous_episode(self):
        self.optpar_previous = OrderedDict()
        self.fisher_previous = OrderedDict()
        if self.fisher_storage == 'single':
            self.optpar_previous['overall'] = OrderedDict()
            self.fisher_previous['overall'] = OrderedDict()

        assert len(list(self.previous_task_dict.keys())), \
            'Only need to load previous episode when current episode is larger than zero and we are in continual ' \
            'learning setup '
        max_fisher_value = 0.
        for taskid in self.previous_task_dict.keys():
            output_folder = self.previous_task_dict[taskid]['output_folder']
            try:
                ckpt_path = join(output_folder, f"model_best.model")
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                assert 'fisher_dict' in checkpoint.keys(), 'No fisher information stored in the best checkpoint'
            except:
                if isfile(join(output_folder, f"model_final_checkpoint.model")):
                    ckpt_path = join(output_folder, f"model_final_checkpoint.model")
                    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                else:
                    raise RuntimeError(f"No checkpoint found for task {taskid}")
            self.print_to_log_file("loading checkpoint", ckpt_path)

            previous_state_dict = OrderedDict()
            previous_fisher = OrderedDict()
            curr_state_dict_keys = list(self.network.state_dict().keys())

            assert 'fisher_dict' in checkpoint.keys(), 'No fisher information stored in this checkpoint'

            for k, value in checkpoint['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                if not key.startswith('seg_outputs'):
                    previous_state_dict[key] = value
                    previous_fisher[key] = checkpoint['fisher_dict'][k]
                    max_fisher_value = max(torch.max(previous_fisher[key]),max_fisher_value)
            self.print_to_log_file("Loaded optimal params and fisher information from task ", taskid)

            if self.fisher_storage == 'each':
                self.optpar_previous[taskid] = previous_state_dict
                self.fisher_previous[taskid] = previous_fisher
            elif self.fisher_storage == 'single':
                self.optpar_previous['overall'] = previous_state_dict
                if not len(list(self.fisher_previous['overall'].keys())):
                    self.fisher_previous['overall'] = previous_fisher
                else:
                    for key in self.fisher_previous['overall'].keys():
                        self.fisher_previous['overall'][key] += previous_fisher[key]

        if self.fisher_storage == 'single':
            for key in self.fisher_previous['overall'].keys():
                self.fisher_previous['overall'][key] /= len(list(self.previous_task_dict.keys()))

        self.print_to_log_file('Max fisher entry:', max_fisher_value)


    def update_fisher_dict(self, data_generator):
        self.print_to_log_file("Start Updating fisher information")
        self.log_prob = RobustCrossEntropyLoss(reduction = 'none')
        self.log_prob = MultipleOutputLoss2(self.log_prob, self.ds_loss_weights)

        square_grad_params_list = []
        for _ in range(self.num_batches_per_epoch):
            # TODO: what part of data to use to update fisher information
            data_dict = next(data_generator)
            data = data_dict['data']
            target = data_dict['target']

            data = maybe_to_torch(data)
            target = maybe_to_torch(target)

            if torch.cuda.is_available():
                data = to_cuda(data)
                target = to_cuda(target)

            if self.fp16:
                with autocast():
                    output = self.network(data)
                    del data
                    log_prob = self.log_prob(output, target)

                scaled_grad_params = [torch.autograd.grad(outputs = self.amp_grad_scaler.scale(p),
                                                     inputs=self.network.parameters(), allow_unused=True,
                                                    retain_graph = True)
                                  for p in log_prob]

                assert len(scaled_grad_params) == self.batch_size
                inv_scale = 1. / self.amp_grad_scaler.get_scale()
                square_grad_params = [sum((d*inv_scale)**2 if d is not None else torch.tensor(0.) for d in param_ls)/self.batch_size
                                    for param_ls in zip(*scaled_grad_params)]
                # self.amp_grad_scaler.update()

            else:
                output = self.network(data)
                del data
                log_prob = self.log_prob(output, target)
                grad_params = [torch.autograd.grad(outputs=p,
                                                  inputs=self.network.parameters(),
                                                   allow_unused=True,retain_graph = True)
                               for p in log_prob]
                square_grad_params = [
                    sum(d ** 2 if d is not None else torch.tensor(0.) for d in param_ls) / self.batch_size
                    for param_ls in zip(*grad_params)]

            if not len(square_grad_params_list):
                square_grad_params_list.extend(square_grad_params)
            else:
                square_grad_params_list = [square_grad_params_list[i] + square_grad_params[i]
                                           for i, _ in enumerate(square_grad_params_list)]

        square_grad_params_list = [param / self.num_batches_per_epoch for param in square_grad_params_list]

        self.fisher_dict = OrderedDict()
        params = [param[0] for param in self.network.named_parameters()]
        assert len(params) == len(square_grad_params_list)
        for name, grad in zip(params, square_grad_params_list):
            if not name.startswith('seg_outputs'):
                self.fisher_dict[name] = grad.clone()
        self.print_to_log_file("Fisher information gets updated!")
        return self.fisher_dict

    def get_consolidation_loss_EWC(self):
        losses = []
        if self.fisher_storage == 'each':
            for taskid in self.previous_task_dict.keys():
                optpars = self.optpar_previous[taskid]
                fishers = self.fisher_previous[taskid]
                if torch.cuda.is_available():
                    optpars = to_cuda(optpars)
                    fishers = to_cuda(fishers)
                for name, param in self.network.named_parameters():
                    if name.startswith('module.'):
                        name = name[7:]
                    if not name.startswith('seg_outputs'):
                        optpar = optpars[name]
                        fisher = fishers[name]
                        losses.append((fisher * (param - optpar) ** 2).sum())
        elif self.fisher_storage == 'single':
            fishers = self.fisher_previous['overall']
            optpars = self.optpar_previous['overall']
            if torch.cuda.is_available():
                optpars = to_cuda(optpars)
                fishers = to_cuda(fishers)
            for name, param in self.network.named_parameters():
                if name.startswith('module.'):
                    name = name[7:]
                if not name.startswith('seg_outputs'):
                    optpar = optpars[name]
                    fisher = fishers[name]
                    losses.append((fisher * (param - optpar) ** 2).sum())
        loss = sum(losses)
        loss = loss * self.lambda_ewc / 2
        return loss

    def load_previous_validation_set(self):
        self.previous_val_loader = OrderedDict()

        for taskid in self.previous_task_dict.keys():
            self.previous_val_loader[taskid] = OrderedDict()
            dataset_directory = self.previous_task_dict[taskid]['dataset_directory']
            plans_file = self.previous_task_dict[taskid]['plans_file']
            if not self.sample_dataset:
                splits_file = join(dataset_directory, "splits_last.pkl")
            else:
                splits_file = join(dataset_directory, f"splits_sample_{self.num_training_sample}.pkl")

            if not isfile(splits_file):
                raise RuntimeError('There should be a split file already!')
            else:
                self.print_to_log_file("Using previous splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            if self.fold < len(splits):
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d validation cases."
                                       % len(val_keys))

            num_val_batches_per_epoch = int(len(val_keys) / self.batch_size)
            val_keys.sort()
            self.previous_val_loader[taskid]['num_val_batches_per_epoch'] = num_val_batches_per_epoch
            self.previous_val_loader[taskid]['val_keys'] = val_keys

            data_identifier = load_pickle(plans_file)['data_identifier']

            dataset_val = OrderedDict()
            folder_with_preprocessed_data = [join(dataset_directory, data_identifier +
                                                f"_sample{i}_ratio{self.slice_ratio}_stage{self.stage}") for i in
                                                  range(self.interact_samples)]
            dataset = load_dataset_from_list(folder_with_preprocessed_data)
            # TODO: validation strategy
            s = np.random.choice(self.interact_samples, size=len(val_keys))
            for i, j in enumerate(val_keys):
                dataset_val[j] = dataset[j][s[i]]

            stage_plans = load_pickle(plans_file)['plans_per_stage'][self.stage]
            patch_size = np.array(stage_plans['patch_size']).astype(int)

            if self.threeD:
                dl_val = DataLoader3D(dataset_val, patch_size, patch_size, self.batch_size, False,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            else:
                dl_val = DataLoader2D(dataset_val, patch_size, patch_size, self.batch_size,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')

            unpack_dataset_from_list(folder_with_preprocessed_data)

            val_gen = get_moreDA_augmentation_only_val(
                dl_val,
                self.data_aug_params,
                deep_supervision_scales=self.deep_supervision_scales,
                pin_memory=self.pin_memory,
                use_nondetMultiThreadedAugmenter=False,
                thread_division_factor=self.thread_division_factor
            )

            self.previous_val_loader[taskid]['val_gen'] = val_gen

    def eval_on_previous_dataset(self):

        if not hasattr(self,'all_val_eval_metrics_on_previous'):
            self.print_to_log_file('creating previous val metric dict...')
            self.all_val_eval_metrics_on_previous = OrderedDict()
            for taskid in self.previous_val_loader.keys():
                self.all_val_eval_metrics_on_previous[taskid] = []

        if self.epoch % 1 == 0:
            self.save_current_decoder()

            for taskid in self.previous_val_loader.keys():
                output_folder = self.previous_task_dict[taskid]['output_folder']
                self.load_decoder(output_folder)

                num_val_batches_per_epoch = self.previous_val_loader[taskid]['num_val_batches_per_epoch']
                val_gen = self.previous_val_loader[taskid]['val_gen']
                with torch.no_grad():
                    self.network.eval()
                    for b in range(num_val_batches_per_epoch):
                        self.run_iteration(val_gen, False, True)

                    self.finish_online_evaluation_on_previous_dataset(task = taskid)

            self.load_current_decoder()
        else:
            for taskid in self.previous_val_loader.keys():
                self.all_val_eval_metrics_on_previous[taskid].append(
                    self.all_val_eval_metrics_on_previous[taskid][-1])

        self.plot_progress_all_tasks()

    def finish_online_evaluation_on_previous_dataset(self,task):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics_on_previous[task].append(np.mean(global_dc_per_class))

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

        colors = ['b','g','r','c','m','y']

        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)

            x_values = list(range(self.epoch + 1))

            for i,taskid in enumerate(self.all_val_eval_metrics_on_previous.keys()):
                if len(self.all_val_eval_metrics_on_previous[taskid]) != len(x_values):
                    diff = len(x_values) - len(self.all_val_eval_metrics_on_previous[taskid])
                    self.print_to_log_file('Ooops...forget to save previous metrics')
                    self.all_val_eval_metrics_on_previous[taskid] = [0.] * diff + \
                                                                    self.all_val_eval_metrics_on_previous[taskid]
                ax.plot(x_values, self.all_val_eval_metrics_on_previous[taskid],
                        color=colors[i+1], ls='--', label=f"task_{taskid}")

            assert len(self.all_val_eval_metrics) == len(x_values)
            ax.plot(x_values, self.all_val_eval_metrics,
                    color=colors[0], ls='--', label=f"current_task")

            ax.set_xlabel("epoch")
            ax.set_ylabel("evaluation metric")
            ax.legend(loc=4)
            fig.savefig(join(self.output_folder, "progress_tasks.png"))

            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def plot_tsne(self):
        features_all = OrderedDict()
        labels_all = OrderedDict()
        with torch.no_grad():
            self.network.eval()
            for b in range(self.num_val_batches_per_epoch):
                features_dict = self.sample_feature(self.val_gen)
                for k,feature in features_dict.items():
                    if k in features_all.keys():
                        features_all[k] = np.concatenate((features_all[k],feature),axis = 0)
                        labels_all[k] = np.concatenate((labels_all[k],np.zeros(feature.shape[0])))
                    else:
                        features_all[k] = feature
                        labels_all[k] = np.zeros(feature.shape[0])

        for taskid in self.previous_val_loader.keys():
            num_val_batches_per_epoch = self.previous_val_loader[taskid]['num_val_batches_per_epoch']
            val_gen = self.previous_val_loader[taskid]['val_gen']
            with torch.no_grad():
                self.network.eval()
                for b in range(num_val_batches_per_epoch):
                    features_dict = self.sample_feature(val_gen)
                    for k,feature in features_dict.items():
                        features_all[k] = np.concatenate((features_all[k],feature),axis = 0)
                        labels_all[k] = np.concatenate((labels_all[k],taskid * np.ones(feature.shape[0])))


        for k,features in features_all.items():
            self.print_to_log_file(f'Feature_size:', features.shape[1])

            if features.shape[1] > 50:
                pca_50 = PCA(n_components=50)
                pca_result_50 = pca_50.fit_transform(features)
                features_tsne = TSNE(random_state = 0).fit_transform(pca_result_50)
            else:
                features_tsne = TSNE(random_state = 0).fit_transform(features)

            self.print_to_log_file(f'TSNE of level {k} done')
            self.plot_cluster(features_tsne,labels_all[k],level = k)

    def initialize_network_blip(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = Conv3d_Q
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = Conv2d_Q
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def store_current_state_dict(self):
        self.print_to_log_file('Storing the current state dict in case it gets overwritten')
        self.current_state_dict = self.network.state_dict()

    def recover_current_state_dict(self):
        self.print_to_log_file('Recovering the current state dict')
        self.network.load_state_dict(self.current_state_dict)

    def update_blip_params(self, task):
        ''':cvar task: start from 0'''
        fisher_dict = self.update_fisher_dict(self.tr_gen)
        del self.fisher_dict

        for name, m in self.network.named_modules():
            if isinstance(m, (Conv2d_Q, Conv3d_Q)) and not name.startswith('seg_outputs'):
                m.Fisher_w.add_(fisher_dict[name+'.weight'].data)
                m.Fisher_b.add_(fisher_dict[name+ '.bias'].data)

                # update bits according to information gain
                m.update_bits(task=task, C=0.5 / math.log(2))
                # do quantization
                m.sync_weight()
                # update Fisher in the buffer
                m.update_fisher(task=task)

    def sample_feature(self,data_generator):
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
        for key,features in features_dict.items():
            B,C = features.size(0), features.size(1)
            features = features.detach().view(B,C,-1).cpu().numpy()
            spatial_size = features.shape[-1]

            random.seed(0)
            k = min(spatial_size,500)
            indices = random.sample(range(spatial_size),k = k)
            sampled_features = features[:,:,indices]
            sampled_features = np.reshape(np.transpose(sampled_features,(0,2,1)),(-1,C))
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



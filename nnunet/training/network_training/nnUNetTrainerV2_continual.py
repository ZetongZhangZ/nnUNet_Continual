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

import numpy as np
import torch

from torch.cuda.amp import autocast
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset

from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import unpack_dataset_from_list
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import load_dataset_from_list
from nnunet.training.dataloading.dataset_loading import delete_npy
from torch import nn


class nnUNetTrainerV2_Continual(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False, args=None, episode=0):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.interact_samples = args.num_samples
        self.slice_ratio = np.clip(args.slice_ratio, a_min=0., a_max=1.)
        self.nointeract = args.nointeract
        self.episode = episode
        self.num_episodes = args.num_episodes
        self.method = args.method
        self.lambda_ewc = args.lambda_ewc
        self.setup = args.setup
        self.exp_name = args.exp_name
        self.first_stage = args.first_stage
        self.background_cls = 0
        self.ignore_cls = -1

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

            if self.setup == 'single_dataset_multi_heads':
                self.all_classes = [1, 2, 3]
                if self.first_stage == 'full':
                    if self.episode:
                        self.classes = [(self.episode - 1) % 3 + 1]
                    else:
                        assert self.classes == [1, 2, 3]
                    self.current_trained_cls = [1, 2, 3]
                elif self.first_stage == 'incremental':
                    self.classes = [self.episode % 3 + 1]
                    self.current_trained_cls = set(self.classes)  # temporarily
                    # directly calculate num of classes here because it is required to initialize the network
                    self.num_classes = self.episode + 2 if self.episode < 3 else 4  # including background

                self.print_to_log_file('During initializing...')
                self.print_to_log_file('Current class:', self.classes)
                self.print_to_log_file('Current number classes:',
                                       self.num_classes)  # num_classes will be used to generate network
                self.print_to_log_file('Current trained classes:', self.current_trained_cls)

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
                self.print_to_log_file(f"Start training episode {self.episode}")
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
            splits_file = join(self.dataset_directory, "splits_last.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                num_train = int(len(all_keys_sorted) * 4 / 5)
                train_keys = np.array(all_keys_sorted)[:num_train]
                test_keys = np.array(all_keys_sorted)[num_train:]
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
                # split into episodes
                episode_file = join(self.dataset_directory, f"episodes_last_{self.num_episodes}.pkl")
                if not isfile(episode_file):
                    self.print_to_log_file(f"Creating new {self.num_episodes} episodes from dataset...")
                    tr_keys = splits[self.fold]['train']
                    val_keys = splits[self.fold]['val']
                    num_each_episode = int(len(tr_keys) / self.num_episodes)
                    episode_dict = OrderedDict()
                    episode_dict['val'] = val_keys
                    episode_dict['train'] = OrderedDict()
                    for j in range(self.num_episodes):
                        current_keys = np.array(tr_keys)[j * num_each_episode:(j + 1) * num_each_episode]
                        episode_dict['train'][j] = current_keys
                    save_pickle(episode_dict, episode_file)
                else:
                    self.print_to_log_file("Using episodes from existing episode file:", episode_file)
                    episode_dict = load_pickle(episode_file)

                tr_keys = episode_dict['train'][self.episode]
                val_keys = episode_dict['val']
                self.print_to_log_file("This episode has %d training and %d validation cases."
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
                index = i + str(s)
                self.dataset_tr[index] = self.dataset[i][s]
                self.dataset[index] = self.dataset[i][s]
            del self.dataset[i]
        self.dataset_val = OrderedDict()
        s = np.random.choice(self.interact_samples, size=len(val_keys))
        for i, j in enumerate(val_keys):
            self.dataset_val[j] = self.dataset[j][s[i]]
            del self.dataset[j]
            self.dataset[j] = self.dataset_val[j]

    def load_best_checkpoint(self, train=True, episode=0):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, f"model_best_episode{episode}.model")):
            # self.load_checkpoint(join(self.output_folder, f"model_best_episode{episode}.model"), train=train)
            fname = join(self.output_folder, f"model_best_episode{episode}.model")
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

            if 'current_trained_cls' in saved_model.keys() and saved_model['current_trained_cls']:
                self.current_trained_cls = saved_model['current_trained_cls']
                self.current_trained_cls.update(self.classes)
            elif not hasattr(self, 'current_trained_cls'):
                self.current_trained_cls = set(self.classes)

            self.print_to_log_file('Current trained cls', list(self.current_trained_cls))
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(episode=episode)

    def load_latest_checkpoint(self, train=True, episode=0):
        if isfile(join(self.output_folder, f"model_final_checkpoint_episode{episode}.model")):
            return self.load_checkpoint(join(self.output_folder, f"model_final_checkpoint_episode{episode}.model"),
                                        train=train)
        if isfile(join(self.output_folder, f"model_latest_episode{episode}.model")):
            return self.load_checkpoint(join(self.output_folder, f"model_latest_episode{episode}.model"), train=train)
        if isfile(join(self.output_folder, f"model_best_episode{episode}.model")):
            return self.load_best_checkpoint(episode=episode)
        raise RuntimeError("No checkpoint found")

    def load_last_episode_checkpoint(self, train=True, episode=1):
        last_episode = episode - 1
        if isfile(join(self.output_folder, f"model_final_checkpoint_episode{last_episode}.model")):
            self.load_checkpoint(join(self.output_folder, f"model_final_checkpoint_episode{last_episode}.model"),
                                 train=train)  # to match the total epoch
        if isfile(join(self.output_folder, f"model_best_episode{last_episode}.model")):
            return self.load_best_checkpoint(episode=last_episode)

        elif isfile(join(self.output_folder, f"model_latest_episode{last_episode}.model")):
            return self.load_checkpoint(join(self.output_folder, f"model_latest_episode{last_episode}.model"),
                                        train=train)
        else:
            return
        # raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False, episode=0):
        filename = join(self.output_folder, f"model_final_checkpoint_episode{episode}.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def EWC_load_previous_episode(self):
        self.optpar_previous = OrderedDict()
        self.fisher_previous = OrderedDict()

        assert hasattr(self, 'episode') and self.episode >= 1, \
            'Only need to load previous episode when current episode is larger than zero and we are in continual ' \
            'learning setup '
        for e in range(self.episode):
            try:
                ckpt_path = join(self.output_folder, f"model_best_episode{e}.model")
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                assert 'fisher_dict' in checkpoint.keys(), 'No fisher information stored in the best checkpoint'
            except:
                if isfile(join(self.output_folder, f"model_final_checkpoint_episode{e}.model")):
                    ckpt_path = join(self.output_folder, f"model_final_checkpoint_episode{e}.model")
                    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                else:
                    raise RuntimeError(f"No checkpoint found for episode {e}")
            self.print_to_log_file("loading checkpoint", ckpt_path)

            previous_state_dict = OrderedDict()
            previous_fisher = OrderedDict()
            curr_state_dict_keys = list(self.network.state_dict().keys())

            assert 'fisher_dict' in checkpoint.keys(), 'No fisher information stored in this checkpoint'
            assert checkpoint['state_dict'].keys() == checkpoint['fisher_dict'].keys(), \
                'State Dict and Fisher Dict contain different keys'
            for k, value in checkpoint['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                previous_state_dict[key] = value
                previous_fisher[key] = checkpoint['fisher_dict'][k]
            self.print_to_log_file("Loaded optimal params and fisher information from episode ", e)
            self.optpar_previous[e] = previous_state_dict
            self.fisher_previous[e] = previous_fisher

    def update_fisher_dict(self, data_generator):
        if not hasattr(self, 'fisher_dict'):
            self.fisher_dict = None
        self.print_to_log_file("Start Updating fisher information")
        self.log_prob = RobustCrossEntropyLoss()
        self.log_prob = MultipleOutputLoss2(self.log_prob, self.ds_loss_weights)

        grad_params_list = []
        for _ in range(self.num_batches_per_epoch):
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

                scaled_grad_params = torch.autograd.grad(outputs=self.amp_grad_scaler.scale(log_prob),
                                                         inputs=self.network.parameters(), allow_unused=True)
                inv_scale = 1. / self.amp_grad_scaler.get_scale()
                grad_params = [(p * inv_scale) ** 2 if p is not None else torch.tensor(0.) for p in scaled_grad_params]

            else:
                output = self.network(data)
                del data
                log_prob = self.log_prob(output, target)
                grad_params = torch.autograd.grad(outputs=log_prob,
                                                  inputs=self.network.parameters(), allow_unused=True)
                grad_params = [p ** 2 if p is not None else torch.tensor(0.) for p in grad_params]

            if not len(grad_params_list):
                grad_params_list.extend(grad_params)
            else:
                grad_params_list = [grad_params_list[i] + grad_params[i] for i, _ in enumerate(grad_params_list)]

        grad_params_list = [param / self.num_batches_per_epoch for param in grad_params_list]

        self.fisher_dict = OrderedDict()
        params = [param[0] for param in self.network.named_parameters()]
        for name, grad in zip(params, grad_params_list):
            self.fisher_dict[name] = grad.clone()
        self.print_to_log_file("Fisher information gets updated!")

    def get_consolidation_loss_EWC(self):
        losses = []
        for e in range(self.episode):
            optpars = self.optpar_previous[e]
            fishers = self.fisher_previous[e]
            if torch.cuda.is_available():
                optpars = to_cuda(optpars)
                fishers = to_cuda(fishers)
            for name, param in self.network.named_parameters():
                if name.startswith('module.'):
                    name = name[7:]
                optpar = optpars[name]
                fisher = fishers[name]
                losses.append((fisher * (param - optpar) ** 2).sum())
        loss = sum(losses)
        loss = loss * self.lambda_ewc / 2
        return loss

    def generate_pseudo_labels(self):
        # for prediction
        if not self.data_aug_params['do_mirror']:
            raise RuntimeError(
                "We did not train with mirroring so you cannot do inference with mirroring enabled")

        self.folder_with_preprocessed_segmentation = []
        for folder in self.folder_with_preprocessed_data:
            dirpath = os.path.dirname(folder)
            folder_name = os.path.basename(folder).split('stage')[0]
            method = self.method if self.method else ''
            new_folder = os.path.join(dirpath, folder_name + method + '_' + self.first_stage + '_' + self.exp_name)
            self.folder_with_preprocessed_segmentation.append(new_folder)
            maybe_mkdir_p(new_folder)
            delete_npy(new_folder)

        for tr_key, data_dict in self.dataset_tr.items():
            new_data_file, new_property_file = self.generate_one_data(data_dict, 'train')
            self.dataset_tr[tr_key]['data_file'] = new_data_file
            self.dataset_tr[tr_key]['property_file'] = new_property_file
            self.dataset[tr_key] = self.dataset_tr[tr_key]

        for val_key, data_dict in self.dataset_val.items():
            new_data_file, new_property_file = self.generate_one_data(data_dict, 'val')
            self.dataset_val[val_key]['data_file'] = new_data_file
            self.dataset_val[val_key]['property_file'] = new_property_file
            self.dataset[val_key] = self.dataset_val[val_key]

        if self.threeD:
            self.dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                      self.batch_size,
                                      False, oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            self.dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            self.dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size,
                                      self.batch_size,
                                      oversample_foreground_percent=self.oversample_foreground_percent,
                                      pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            self.dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')

        if self.unpack_data:
            print("unpacking dataset")
            unpack_dataset_from_list(self.folder_with_preprocessed_segmentation)
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

    def generate_one_data(self, data_dict, mode='train'):
        data_file = data_dict['data_file']
        property_file = data_dict['properties_file']
        data_name = os.path.basename(data_file)
        property_name = os.path.basename(property_file)
        dir_name = os.path.basename(os.path.dirname(data_file)).split('stage')[0]
        method = self.method if self.method else ''
        new_dir_name = dir_name + method + '_' + self.first_stage + '_' + self.exp_name
        up_dir_name = os.path.dirname(os.path.dirname(data_file))
        assert isdir(os.path.join(up_dir_name, new_dir_name)), 'The directory does not exist'

        new_data_file = os.path.join(up_dir_name, new_dir_name, data_name)
        new_property_file = os.path.join(up_dir_name, new_dir_name, property_name)

        if isfile(new_data_file) and isfile(new_property_file) and mode == 'train':
            return new_data_file, new_property_file

        self.print_to_log_file('Processing ', data_name)
        data = np.load(data_file)['data']
        seg = data[-1]
        image = data[:-1]
        properties = load_pickle(property_file)

        if mode == 'train':
            self.not_current_classes = [c for c in self.all_classes if c not in self.classes]
            properties['classes'] = [c for c in self.all_classes if c in self.current_trained_cls]
            properties['classes'].extend([self.ignore_cls, self.background_cls])

            for cls in self.not_current_classes:
                if cls in properties['class_locations'].keys():
                    del properties['class_locations'][cls]

            if self.episode:
                # from trainer.validate
                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(image,
                                                                                     do_mirroring=True,
                                                                                     mirror_axes=self.data_aug_params[
                                                                                         'mirror_axes'],
                                                                                     use_sliding_window=True,
                                                                                     step_size=0.5,
                                                                                     use_gaussian=True,
                                                                                     all_in_gpu=False,
                                                                                     mixed_precision=self.fp16)[1]
                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                # generate the pseudo labels

                assert softmax_pred.shape[-3:] == seg.shape[-3:], \
                    f"The size of prediction must be the same as segmentation label,{softmax_pred.shape}|{seg.shape}"

                prediction_cls = np.argmax(softmax_pred, axis=0)
                prediction_max = np.max(softmax_pred, axis=0)

            new_seg = np.copy(seg)

            num_difference = 0
            for c in self.not_current_classes:
                new_seg[new_seg == c] = self.background_cls  # should be background, not ignored index
                if self.episode:
                    median = np.median(prediction_max[prediction_cls == c])
                    threshold = min(median,
                                    0.75)  # not exactly the same as PLOP: median of a single image, not entire dataset
                    mask = prediction_max > threshold
                    mask_c = np.bitwise_and(mask, prediction_cls == c)
                    new_seg[mask_c] = c
                    num_difference += np.bitwise_and(np.bitwise_not(mask), prediction_cls == c)

            if num_difference:
                self.print_to_log_file('Number of voxels whose confidence is below median:', num_difference)

            assert np.sum(np.isin(new_seg, np.array(self.not_current_classes))) == 0, 'Something Wrong'

        elif mode == 'val':
            new_seg = np.copy(seg)
            self.not_trained_cls = [c for c in self.all_classes if c not in self.current_trained_cls]

            properties['classes'] = [c for c in self.all_classes if c in self.current_trained_cls]
            properties['classes'].extend([self.ignore_cls, self.background_cls])

            for c in self.not_trained_cls:
                if c in properties['class_locations'].keys():
                    del properties['class_locations'][c]
                new_seg[new_seg == c] = self.background_cls  # background or ignored?

            assert np.sum(np.isin(new_seg, np.array(self.not_trained_cls))) == 0, 'Something Wrong'

        if len(new_seg.shape) == 3:
            new_seg = np.expand_dims(new_seg, axis=0)
        new_data = np.concatenate((image, new_seg), axis=0)

        save_pickle(properties, new_property_file)
        np.savez_compressed(new_data_file, data=new_data.astype(np.float32))
        return new_data_file, new_property_file

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


import argparse
from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration,get_nointeract_configuration
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")

    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')
    parser.add_argument('--nointeract', default = False,action = 'store_true',
                        help='Specify to train without interactions')
    parser.add_argument('--fixiteration', default = None,help = 'Number of total iterations')
    parser.add_argument('--exp_name',default = None)
    parser.add_argument('--sample_dataset', default = False, action = 'store_true',
                        help = 'use only part of image of each dataset')

    args = parser.parse_args()

    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic


    fp32 = args.fp32
    run_mixed_precision = not fp32

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    task_list = [3, 6, 7, 8, 9, 10, 135]
    task_dict = {
        3: {'name': 'Liver', 'training_set_size': 94, 'plan_identifier': 'nnUNetPlans_interactive'},
        6: {'name': 'Lung', 'training_set_size': 50, 'plan_identifier': 'nnUNetPlansv2.1'},
        7: {'name': 'Pancreas', 'training_set_size': 224, 'plan_identifier': 'nnUNetPlans_interactive'},
        8: {'name': 'HepaticVessel', 'training_set_size': 241, 'plan_identifier': 'nnUNetPlansv2.1'},
        9: {'name': 'Spleen', 'training_set_size': 41, 'plan_identifier': 'nnUNetPlansv2.1'},
        10: {'name': 'Colon', 'training_set_size': 126, 'plan_identifier': 'nnUNetPlansv2.1'},
        135: {'name': 'KiTS2021', 'training_set_size': 240, 'plan_identifier': 'nnUNetPlans_interactive'},
    }
    features_all = OrderedDict()
    labels_all = OrderedDict()
    for task in task_list:
        task_id = int(task)
        task = convert_id_to_task_name(task_id)
        plans_identifier = task_dict[task_id]['plan_identifier']

        if not args.nointeract:
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
        else:
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_nointeract_configuration(network, task, network_trainer, plans_identifier)

        if not args.nointeract:
            output_folder_name += f'_ratio_{args.slice_ratio}_sample_{args.num_samples}'
        else:
            output_folder_name += '_nointeract'
            args.slice_ratio = 0.
            args.num_samples = 1

        if args.fixiteration:
            output_folder_name += '_iteration_' + args.fixiteration
            args.fixiteration = int(args.fixiteration)

        if args.exp_name:
            output_folder_name += '_' + args.exp_name

        trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision,args =args)

        trainer.initialize(not validation_only)
        if plans_identifier == 'nnUNetPlans_interactive':
            trainer.gt_niftis_folder = join(trainer.dataset_directory, "gt_segmentations_only_foreground")
        trainer.save_every = 10
        trainer.load_best_checkpoint(train=False)
        trainer._maybe_init_amp()

        features = trainer.get_features()
        for k,feature in features.items():
            if k not in features_all.keys():
                features_all[k] = feature
                labels_all[k] = task_id * np.ones(feature.shape[0])
            else:
                features_all[k] = np.concatenate((features_all[k],feature),axis = 0)
                labels_all[k] = np.concatenate((labels_all[k],task_id * np.ones(feature.shape[0])))

    for k, features in features_all.items():
        trainer.print_to_log_file(f'Feature_size:', features.shape[1])

        if features.shape[1] > 50:
            pca_50 = PCA(n_components=50)
            pca_result_50 = pca_50.fit_transform(features)
            features_tsne = TSNE(random_state=0).fit_transform(pca_result_50)
        else:
            features_tsne = TSNE(random_state=0).fit_transform(features)

        trainer.print_to_log_file(f'TSNE of level {k} done')
        trainer.plot_cluster(features_tsne, labels_all[k], level=k)


if __name__ == "__main__":
    main()

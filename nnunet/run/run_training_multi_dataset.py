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
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration, get_nointeract_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.paths import network_training_output_dir,preprocessing_output_dir



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("--fold", type=str, default='0', choices=['0'])
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    # parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
    #                     default="nnUNetPlans_continual", required=False)
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
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
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
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--num_samples', type=int, default=3,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')

    parser.add_argument('--nointeract', default=False, action='store_true',
                        help='Specify to train without interactions')

    parser.add_argument('--sample_dataset', default = False, action = 'store_true',
                        help = 'use only part of image of each dataset')
    parser.add_argument('--num_epochs',default = 1000, type = int, help = 'number of epochs for each dataset')

    args = parser.parse_args()

    fold = int(args.fold)
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    # plans_identifier = args.p
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    task_list = [3, 6, 7, 8, 9, 10, 135]
    training_task = [3,6,7,9,10,135]
    task_dict = {
        3: {'name': 'Liver', 'training_set_size': 94, 'plan_identifier': 'nnUNetPlans_interactive'},
        6: {'name': 'Lung', 'training_set_size': 50, 'plan_identifier': 'nnUNetPlansv2.1'},
        7: {'name': 'Pancreas', 'training_set_size': 224, 'plan_identifier': 'nnUNetPlans_interactive'},
        8: {'name': 'HepaticVessel', 'training_set_size': 241, 'plan_identifier': 'nnUNet_interactive'},
        9: {'name': 'Spleen', 'training_set_size': 41, 'plan_identifier': 'nnUNetPlansv2.1'},
        10: {'name': 'Colon', 'training_set_size': 126, 'plan_identifier': 'nnUNetPlansv2.1'},
        135: {'name': 'KiTS2021', 'training_set_size': 240, 'plan_identifier': 'nnUNetPlans_interactive'},
    }

    trainer_task_dict = OrderedDict()
    for j in range(len(training_task)):
        task_id = training_task[j]
        trainer_task_dict[task_id] = OrderedDict()

        map_func = lambda x:'Task' + str(x).rjust(3,'0') + '_' + task_dict[x]['name']
        task = map_func(task_id)
        plans_identifier = task_dict[task_id]['plan_identifier']
        training_set_size = task_dict[task_id]['training_set_size']

        if not args.nointeract:
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
        else:
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_nointeract_configuration(network, task, network_trainer, plans_identifier)

        if trainer_class is None:
            raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

        suffix = ''
        if not args.nointeract:
            suffix += f'_ratio_{args.slice_ratio}_sample_{args.num_samples}'  # to be modified
        else:
            suffix += '_nointeract'  # output folder name = trainer + planner，Now choose nnuNet_planner_interactive
            args.slice_ratio = 0.
            args.num_samples = 1

        if args.exp_name:
            suffix += '_' + args.exp_name
        output_folder_name += suffix

        gt_folder = 'gt_segmentations_only_foreground' \
            if task_dict[task_id]['plan_identifier'] == 'nnUNet_interactive' else 'gt_segmentations'

        if j:
            assert batch_dice == trainer_task_dict[training_task[j-1]]['batch_dice'],\
                'batch dice should be the same'
            assert stage == trainer_task_dict[training_task[j-1]]['stage'],\
                'stage should be the same'

        trainer_task_dict[task_id]['output_folder_base'] = output_folder_name
        trainer_task_dict[task_id]['output_folder'] = os.path.join(output_folder_name,f'fold_{fold}')
        trainer_task_dict[task_id]['dataset_directory'] = dataset_directory
        trainer_task_dict[task_id]['plans_file'] = plans_file
        trainer_task_dict[task_id]['plan_identifier'] = task_dict[task_id]['plan_identifier']
        trainer_task_dict[task_id]['gt_niftis_folder'] = os.path.join(dataset_directory,gt_folder)
        trainer_task_dict[task_id]['batch_dice'] = batch_dice
        trainer_task_dict[task_id]['stage'] = stage

    print('trainer task dict',trainer_task_dict)
    print('\n\n\n')
    print('------------------------------------')
    print('\n')
    print(f'Start training all the tasks')
    print('\n')
    print('------------------------------------')
    try:
        trainer = trainer_class(fold = fold, trainer_task_dict = trainer_task_dict,
                                batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                deterministic=deterministic,
                                fp16=run_mixed_precision, args=args)
    except:
        raise ValueError('The specified trainer does not take current arguments')

    trainer.max_num_epochs = args.num_epochs
    trainer.current_max_num_epochs = args.num_epochs

    trainer.initialize(not validation_only)
    trainer.save_every = 10

    if not validation_only:
        if args.continue_training:
            # -c was set, continue a previous training and ignore pretrained weights
            trainer.load_latest_checkpoint()

        elif (not args.continue_training) and (args.pretrained_weights is not None):
            # we start a new training. If pretrained_weights are set, use them
            load_pretrained_weights(args.pretrained_weights)
        else:
            # new training without pretraine weights, do nothing
            pass

        trainer.run_training()

    else:
        trainer.load_best_checkpoint(train=False)

        trainer.network.eval()
        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                         run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                         overwrite=args.val_disable_overwrite)


if __name__ == "__main__":
    main()

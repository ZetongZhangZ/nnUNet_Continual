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
    parser.add_argument('--pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--num_samples', type=int, default=3,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')

    parser.add_argument('--start_task', type=int, required=False, default=3, choices=[3, 6, 7, 9, 10, 135])
    parser.add_argument('--nointeract', default=False, action='store_true',
                        help='Specify to train without interactions')
    parser.add_argument('--method', type=str, default=None, choices=[None, 'EWC', 'PNN'],
                        help='method for continual learning')
    parser.add_argument('--lambda_ewc', type=float, default=0.1,
                        help='lambda_ewc')
    parser.add_argument('--fisher_storage',choices = ['each','single'],help='the method to store fisher information')

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

    start_index = training_task.index(int(args.start_task))
    for j in range(start_index,len(training_task)):
        task_id = training_task[j]

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
        if args.method:
            suffix += '_' + args.method
        if args.exp_name:
            suffix += '_' + args.exp_name
        output_folder_name += suffix

        previous_task_ids = [training_task[k] for k in range(j)] if j else []
        previous_tasks = list(map(map_func, previous_task_ids))
        previous_task_dict = OrderedDict()
        for id,t in enumerate(previous_task_ids):
            previous_task_dict[t] = OrderedDict()
            previous_folder = join(network_training_output_dir, network, previous_tasks[id],
                                   network_trainer + "__" + task_dict[t]['plan_identifier'])
            previous_folder += suffix
            previous_folder = os.path.join(previous_folder,f'fold_{fold}')
            previous_dataset_dir= join(preprocessing_output_dir, previous_tasks[id])

            if args.nointeract:
                previous_plans_file = join(preprocessing_output_dir, previous_tasks[id],
                              task_dict[t]['plan_identifier'] + "_plans_3D_nointeract.pkl")
            else:
                previous_plans_file = join(preprocessing_output_dir, previous_tasks[id],
                                  task_dict[t]['plan_identifier'] + "_plans_3D.pkl")

            previous_gt_folder = 'gt_segmentations_only_foreground' \
                if task_dict[t]['plan_identifier'] == 'nnUNet_interactive' else 'gt_segmentations'

            previous_task_dict[t]['output_folder'] = previous_folder
            previous_task_dict[t]['dataset_directory'] = previous_dataset_dir
            previous_task_dict[t]['plans_file'] = previous_plans_file
            previous_task_dict[t]['plan_identifier'] = task_dict[t]['plan_identifier']
            previous_task_dict[t]['gt_niftis_folder'] = os.path.join(previous_dataset_dir,previous_gt_folder)

        print('previous_task_dict',previous_task_dict)
        print('\n\n\n')
        print('------------------------------------')
        print('\n')
        print(f'Start training {task}...')
        print('\n')
        print('------------------------------------')
        try:
            trainer = trainer_class(plans_file, fold, output_folder=output_folder_name,
                                    dataset_directory=dataset_directory,
                                    batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                    deterministic=deterministic,
                                    fp16=run_mixed_precision, args=args,previous_task_dict = previous_task_dict )
        except:
            raise ValueError('The specified trainer does not take additional arguments')

        trainer.max_num_epochs = args.num_epochs
        trainer.current_max_num_epochs = args.num_epochs
        if plans_identifier == 'nnUNetPlans_interactive':
            trainer.gt_niftis_folder = join(trainer.dataset_directory, "gt_segmentations_only_foreground")
        trainer.initialize(not validation_only)
        trainer.save_every = 10
        trainer._maybe_init_amp()

        trainer.load_best_checkpoint(output_folder=trainer.output_folder, train=False)
        trainer.plot_tsne()




if __name__ == "__main__":
    main()

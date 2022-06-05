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
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration,get_nointeract_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.training.dataloading.dataset_loading import delete_npy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("--fold", type = str, default = '0',choices = ['0'])
    parser.add_argument("--exp_name", type = str, default='')
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
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    parser.add_argument('--num_samples', type=int, default=3,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')

    parser.add_argument('--num_episodes',type = int, required = False, default = 10,
                        help='NUM of episode for continuous learning')
    parser.add_argument('--start_episode', type=int,required = False,default = 0)
    parser.add_argument('--nointeract', default=False, action='store_true',
                        help='Specify to train without interactions')
    parser.add_argument('--method', type = str, default=None, choices=[None,'EWC','PNN'],
                        help='method for continual learning')
    parser.add_argument('--lambda_ewc', type=float,default = 0.1,
                        help='lambda_ewc')

    parser.add_argument('--pseudo_GT',default = False, action = 'store_true',
                        help = 'whether to use pseudo labels generated from interactive learning')
    parser.add_argument('--setup',default = 'single_dataset_multi_heads',
                        choices=['single_dataset_single_head','single_dataset_multi_heads','multi_datasets_multi_heads'],
                        help = 'setup of continual learning')
    parser.add_argument('--first_stage', default='full',
                        choices=['full', 'incremental'],
                        help='how to start training for the setup \'single_dataset_multi_heads\':'
                             'full --> supervision on all the classes at the first stage,'
                             'incremental --> only training on the first foreground class at the first stage')

    args = parser.parse_args()

    task = args.task
    fold = args.fold
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

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    if args.setup == 'single_dataset_single_head':
        plans_identifier = 'nnUNetPlans_continual'
    elif args.setup == 'single_dataset_multi_heads':
        plans_identifier = 'nnUNetPlansv2.1'
    elif args.setup == 'multi_datasets_multi_heads':
        raise NotImplementedError
        plans_identifier = ...

    if not args.nointeract:
        plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)
    else:
        plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_nointeract_configuration(network, task, network_trainer, plans_identifier)

    if not args.nointeract:
        output_folder_name += f'_ratio_{args.slice_ratio}_sample_{args.num_samples}' # to be modified
    else:
        output_folder_name += '_nointeract' # output folder name = trainer + planner，Now choose nnuNet_planner_interactive
        args.slice_ratio = 0.
        args.num_samples = 1

    if args.setup == 'single_dataset_multi_heads':
        output_folder_name += '_' + args.first_stage

    if args.method:
        output_folder_name += '_' + args.method

    if args.exp_name:
        output_folder_name += '_' + args.exp_name

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")


    # continuous learning
    assert 0 <= args.start_episode < args.num_episodes, 'Argument start_episode out of range'

    for i in range(args.start_episode,args.num_episodes):
        print('\n\n\n')
        print('------------------------------------')
        print('\n')
        print(f'Start episode {i}...')
        print('\n')
        print('------------------------------------')
        try:
            trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                                    batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                                    deterministic=deterministic,
                                    fp16=run_mixed_precision,args=args,episode = i)
        except:
            raise ValueError('The specified trainer does not take additional arguments')

        if args.disable_saving:
            trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
            trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
            # self.best_val_eval_criterion_MA
            trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
            # the training chashes
            trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

        trainer.initialize(not validation_only)
        trainer.current_max_num_epochs = 80 * (i + 1)
        trainer.save_every = 10

        if not args.pseudo_GT:
            if args.setup == 'single_dataset_single_head':
                trainer.gt_niftis_folder = join(trainer.dataset_directory, "gt_segmentations_only_foreground")
            elif args.setup == 'single_dataset_multi_heads':
                trainer.gt_niftis_folder = join(trainer.dataset_directory, "gt_segmentations")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if not validation_only:
            if i:
                trainer.load_last_episode_checkpoint(episode = i)
                trainer.best_val_eval_criterion_MA = None
                trainer.best_MA_tr_loss_for_patience = None
                trainer.best_epoch_based_on_MA_tr_loss = None
                if args.method == 'EWC':
                    trainer.EWC_load_previous_episode()

            if args.start_episode == i:
                if args.continue_training :
                    # -c was set, continue a previous training and ignore pretrained weights
                    trainer.load_latest_checkpoint(episode = i)

                elif (not args.continue_training) and (args.pretrained_weights is not None):
                    # we start a new training. If pretrained_weights are set, use them
                    load_pretrained_weights(trainer.network, args.pretrained_weights)
                else:
                    # new training without pretraine weights, do nothing
                    pass

            # add for continual learning
            if args.setup == 'single_dataset_multi_heads':
                trainer.generate_pseudo_labels()

            trainer.run_training()

        else:
            trainer.load_best_checkpoint(train=False,episode = i)
            # trainer.load_final_checkpoint(train=False,episode = i)

            trainer.network.eval()
            # predict validation
            trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                             run_postprocessing_on_folds=not disable_postprocessing_on_folds,
                             overwrite=args.val_disable_overwrite)



if __name__ == "__main__":
    main()

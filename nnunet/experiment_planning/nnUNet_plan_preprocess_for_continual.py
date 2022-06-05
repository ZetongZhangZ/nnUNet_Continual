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


import nnunet
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl", "--planner", type=str, default="ExperimentPlanner_Continual",
                        help="Name of the ExperimentPlanner class for interactive learning. "
                             "Default is ExperimentPlanner_Interactive. Can be 'None', in which case these U-Nets will"
                             "not be configured")
    parser.add_argument("--model_type", type=str, default="3d",
                        help="Choose between 2d and 3d network structures, default is 2d")
    parser.add_argument("-no_pp", action="store_true",default=True,
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument('-fgcls', type=str, default='all',
                        help='which classes would be recognize as foreground, default is all, possible options: [all'
                             '1,2,3')
    parser.add_argument('--num_fg_clicks', type=int, default=5, help='maximal number of positive clicks')
    parser.add_argument('--num_bg_clicks', type=int, default=10,
                        help='maximal number of negative clicks')
    # parser.add_argument('--num_bg_samples_s2', type=int, default=10,
    #                     help='maximal number of negative clicks for strategy 2')
    parser.add_argument('-d', type=int, default=20,
                        help='maximal distance from negative clicks to ground truth boundaries')
    parser.add_argument('-d_step', type=int, default=8,
                        help='minimal distance between two postive(or negative)clicks')
    parser.add_argument('-d_margin', type=int, default=5,
                        help='minimal distance from negative clicks to ground truth boundaries')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')
    parser.add_argument('--nointeract',action='store_true',help='whether to do interactive learning')
    parser.add_argument('--num_episodes', type=int, required=False, default=10,
                        help='NUM of episode for continuous learning')


    args = parser.parse_args()
    args_dict = vars(args)
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    model_type = args.model_type
    assert model_type in ['2d', '3d'], "Please choose model type between 2d and 3d"
    planner_name = args.planner + '_' + model_type.upper() if args.planner != 'None' else None

    assert args.fgcls in ['all', '1', '2', '3'], "Please choose the foreground class(es) according to instruction"
    if args.fgcls == 'all':
        fg_clses = [1, 2, 3]
    else:
        fg_clses = [int(args.fgcls)]
    args_dict['foreground_cls'] = fg_clses

    if args.nointeract:
        args_dict['num_samples'] = 1
        args_dict['slice_ratio'] = 0

    print(args_dict)
    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        task_name = convert_id_to_task_name(i)

        tasks.append(task_name)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name is not None:
        planner = recursive_find_python_class([search_in], planner_name, current_module="nnunet.experiment_planning")
        if planner is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name)
    else:
        planner = None

    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        # splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        # lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)

        # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False,
                                           num_processes=tf)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(
            collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

        maybe_mkdir_p(preprocessing_output_dir_this_task)
        shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

        threads = (tl, tf)

        print("number of threads: ", threads, "\n")
        if planner is not None:
            try:
                exp_planner = planner(cropped_out_dir, preprocessing_output_dir_this_task, args_dict)
            except:
                raise ValueError('The specified planner does not take additional arguments')
            exp_planner.plan_experiment()
            if not dont_run_preprocessing:  # double negative, yooo
                exp_planner.run_preprocessing(threads, args_dict)



if __name__ == "__main__":
    main()

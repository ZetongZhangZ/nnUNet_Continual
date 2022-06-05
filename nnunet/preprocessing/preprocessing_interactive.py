from nnunet.preprocessing.preprocessing import PreprocessorFor2D, resample_patient, GenericPreprocessor
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.configuration import default_num_threads, RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD
from multiprocessing.pool import Pool
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
from scipy.ndimage import distance_transform_bf


class PreprocessorForInteractive3D(GenericPreprocessor):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list),
                 intensityproperties=None, args=None):
        super(PreprocessorForInteractive3D, self).__init__(normalization_scheme_per_modality, use_nonzero_mask,
                                                           transpose_forward, intensityproperties)
        self.foreground_cls = args['foreground_cls']
        self.num_fg_clicks = args['num_fg_clicks']
        self.num_bg_clicks = args['num_bg_clicks']
        self.d = args['d']
        self.d_step = args['d_step']
        self.d_margin = args['d_margin']
        self.num_samples = args['num_samples']
        self.sample_strageties = ['sequential', 'random']
        self.slice_ratio = np.clip(args['slice_ratio'], a_min=0., a_max=1.)
        self.nointeract = args['nointeract']

    def sample_clicks(self, seg, click_from='2d'):
        seg[seg < 0] = 0
        assert len(np.unique(seg)) == 2, f'Segmentation ground truth contains class other than 0 and 1,' \
                                         f' the classes it contains are {np.unique(seg)}'
        seg = seg.squeeze()
        dimz, dimx, dimy = seg.shape
        if click_from == '2d':
            num_z_index = int(self.slice_ratio * dimz)
            chosen_z_index = np.int_(np.linspace(start=0, stop=dimz - 1, num=num_z_index))
            foreground_clicks = []
            background_clicks = []
            for idx in chosen_z_index:
                flag_fg = False  # whether at least one click has been generated on the foreground map
                strategy = np.random.choice(self.sample_strageties)
                sliced_seg = seg[idx]
                dist_fg = distance_transform_bf(sliced_seg)
                xs, ys = np.nonzero(dist_fg > self.d_margin)
                if len(xs):
                    fg_candidates, valid_count = find_valid_click_position([dimx, dimy], [xs, ys], self.d_margin,
                                                                           self.d_step)
                    if valid_count:
                        max_num_fg_clicks = np.minimum(self.num_fg_clicks, valid_count)
                        num_fg_clicks = np.random.randint(1, max_num_fg_clicks + 1)
                        fg_clicks = create_clicks(fg_candidates, num_fg_clicks, sliced_seg, strategy)
                        fg_clicks = np.hstack((np.ones_like(fg_clicks[:, [0]]) * idx, fg_clicks)).astype(np.uint32)
                        foreground_clicks.extend(fg_clicks.tolist())
                        flag_fg = True

                if (not flag_fg) and (np.sum(sliced_seg)):
                    # if there is a foreground region in this map, at least sample one click
                    fg_pixels_x, fg_pixels_y = np.nonzero(sliced_seg == 1)
                    fg_pixels = np.vstack((fg_pixels_x, fg_pixels_y)).T
                    fg_clicks = create_clicks(fg_pixels, 1, sliced_seg, strategy)
                    fg_clicks = np.hstack((np.ones_like(fg_clicks[:, [0]]) * idx, fg_clicks)).astype(np.uint32)
                    foreground_clicks.extend(fg_clicks.tolist())
                    flag_fg = True

                dist_bg = distance_transform_bf(np.logical_not(sliced_seg))
                bg_mask = np.bitwise_and(self.d_margin < dist_bg, dist_bg < self.d)
                xs, ys = np.nonzero(bg_mask)
                if len(xs):
                    bg_candidates, valid_count = find_valid_click_position([dimx, dimy], [xs, ys], self.d_margin,
                                                                           self.d_step)
                    if valid_count:
                        max_num_bg_clicks = np.minimum(self.num_bg_clicks, valid_count)
                        num_bg_clicks = np.random.randint(1, max_num_bg_clicks + 1)
                        bg_clicks = create_clicks(bg_candidates, num_bg_clicks, bg_mask, strategy)
                        bg_clicks = np.hstack((np.ones_like(bg_clicks[:, [0]]) * idx, bg_clicks)).astype(np.uint32)
                        background_clicks.extend(bg_clicks.tolist())
        elif click_from == '3d':
            raise NotImplementedError
        else:
            raise ValueError('You can only choose 2d or 3d view')
        return np.array(foreground_clicks), np.array(background_clicks)

    def generate_distance_map(self, clicks, shape):
        if len(clicks):
            clicks_map = np.zeros(shape).astype(bool)
            for (z, x, y) in clicks:
                clicks_map[0, z, x, y] = 1
            distance_map = distance_transform_bf(np.bitwise_not(clicks_map))
            distance_map[distance_map > 255] = 255
        else:
            distance_map = np.ones_like(clicks) * 255
        return distance_map, clicks_map

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        """

        :param target_spacings: list of lists [[1.25, 1.25, 5]]
        :param input_folder_with_cropped_npz: dim: c, x, y, z | npz_file['data'] np.savez_compressed(fname.npz, data=arr)
        :param output_folder:
        :param num_threads:
        :param force_separate_z: None
        :return:
        """
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        maybe_mkdir_p(output_folder)
        num_stages = len(target_spacings)
        if not isinstance(num_threads, (list, tuple, np.ndarray)):
            num_threads = [num_threads] * num_stages

        assert len(num_threads) == num_stages

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        self.all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']
        self.background_cls = [i for i in self.all_classes if i not in self.foreground_cls]
        all_classes = [1]

        for i in range(num_stages):
            all_args = []
            for k in range(self.num_samples):
                output_folder_stage = os.path.join(output_folder,
                                                   data_identifier + f"_sample{k}_ratio{self.slice_ratio}_stage{i}")
                maybe_mkdir_p(output_folder_stage)
                spacing = target_spacings[i]
                for j, case in enumerate(list_of_cropped_npz_files):
                    case_identifier = get_case_identifier_from_npz(case)
                    args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                    all_args.append(args)
            p = Pool(num_threads[i])
            p.starmap(self._run_internal, all_args)
            p.close()
            p.join()

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        """
        data and seg must already have been transposed by transpose_forward. properties are the un-transposed values
        (spacing etc)
        :param data:
        :param target_spacing:
        :param properties:
        :param seg:
        :param force_separate_z:
        :return:
        """

        # target_spacing is already transposed, properties["original_spacing"] is not so we need to transpose it!
        # data, seg are already transposed. Double check this using the properties
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }

        # remove nans
        data[np.isnan(data)] = 0

        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        use_nonzero_mask = self.use_nonzero_mask

        new_seg = np.copy(seg)
        for c in self.foreground_cls:
            new_seg[seg == c] = 1
        for c in self.background_cls:
            new_seg[seg == c] = 0
        seg = np.copy(new_seg)

        properties['classes'] = np.unique(seg).tolist()

        if not self.nointeract:
            print('Sample clicks...')
            fg_clicks, bg_clicks = self.sample_clicks(np.copy(seg))
            shape = seg.shape
            fg_distance_map, _ = self.generate_distance_map(fg_clicks, shape)
            bg_distance_map, _ = self.generate_distance_map(bg_clicks, shape)
            data = np.concatenate((data, fg_distance_map, bg_distance_map), axis=0)
            print('size after sampling clicks', data.shape)
            properties['fg_clicks_index'] = fg_clicks
            properties['bg_clicks_index'] = bg_clicks
            properties['use_nonzero_mask_for_norm'][4] = False
            properties['use_nonzero_mask_for_norm'][5] = False

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if data[c].min() == data[c].max() == 255:
                    # to deal with constant distance map (very rare actually)
                    # normalize to 1
                    data[c] /= 255
                else:
                    if use_nonzero_mask[c]:
                        mask = seg[-1] >= 0
                        data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                        data[c][mask == 0] = 0
                    else:
                        mn = data[c].mean()
                        std = data[c].std()
                        # print(data[c].shape, data[c].dtype, mn, std)
                        data[c] = (data[c] - mn) / (std + 1e-8)

        return data, seg, properties


class PreprocessorForInteractive2D(PreprocessorForInteractive3D):
    def __init__(self, normalization_scheme_per_modality, use_nonzero_mask, transpose_forward: (tuple, list),
                 intensityproperties=None, args=None):
        super(PreprocessorForInteractive3D, self).__init__(normalization_scheme_per_modality, use_nonzero_mask,
                                                           transpose_forward, intensityproperties, args)

    def run(self, target_spacings, input_folder_with_cropped_npz, output_folder, data_identifier,
            num_threads=default_num_threads, force_separate_z=None):
        print("Initializing to run preprocessing")
        print("npz folder:", input_folder_with_cropped_npz)
        print("output_folder:", output_folder)
        list_of_cropped_npz_files = subfiles(input_folder_with_cropped_npz, True, None, ".npz", True)
        assert len(list_of_cropped_npz_files) != 0, "set list of files first"
        maybe_mkdir_p(output_folder)
        all_args = []
        num_stages = len(target_spacings)

        # we need to know which classes are present in this dataset so that we can precompute where these classes are
        # located. This is needed for oversampling foreground
        self.all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']
        self.background_cls = [i for i in self.all_classes if i not in self.foreground_cls]
        all_classes = [1]

        for i in range(num_stages):
            for k in range(self.num_samples):
                output_folder_stage = os.path.join(output_folder,
                                                   data_identifier + f"_sample{k}_ratio{self.slice_ratio}_stage{i}")
                maybe_mkdir_p(output_folder_stage)
                spacing = target_spacings[i]
                for j, case in enumerate(list_of_cropped_npz_files):
                    case_identifier = get_case_identifier_from_npz(case)
                    args = spacing, case_identifier, output_folder_stage, input_folder_with_cropped_npz, force_separate_z, all_classes
                    all_args.append(args)
        p = Pool(num_threads)
        p.starmap(self._run_internal, all_args)
        p.close()
        p.join()

    def resample_and_normalize(self, data, target_spacing, properties, seg=None, force_separate_z=None):
        original_spacing_transposed = np.array(properties["original_spacing"])[self.transpose_forward]
        before = {
            'spacing': properties["original_spacing"],
            'spacing_transposed': original_spacing_transposed,
            'data.shape (data is transposed)': data.shape
        }
        target_spacing[0] = original_spacing_transposed[0]
        data, seg = resample_patient(data, seg, np.array(original_spacing_transposed), target_spacing, 3, 1,
                                     force_separate_z=force_separate_z, order_z_data=0, order_z_seg=0,
                                     separate_z_anisotropy_threshold=self.resample_separate_z_anisotropy_threshold)
        after = {
            'spacing': target_spacing,
            'data.shape (data is resampled)': data.shape
        }
        print("before:", before, "\nafter: ", after, "\n")

        if seg is not None:  # hippocampus 243 has one voxel with -2 as label. wtf?
            seg[seg < -1] = 0

        properties["size_after_resampling"] = data[0].shape
        properties["spacing_after_resampling"] = target_spacing
        properties['classes'] = np.unique(seg).tolist()
        use_nonzero_mask = self.use_nonzero_mask

        for c in self.foreground_cls:
            seg[seg == c] = 1
        for c in self.background_cls:
            seg[seg == c] = 0

        if not self.nointeract:
            print('Sample clicks...')
            fg_clicks, bg_clicks = self.sample_clicks(np.copy(seg))
            shape = seg.shape
            fg_distance_map, _ = self.generate_distance_map(fg_clicks, shape)
            bg_distance_map, _ = self.generate_distance_map(bg_clicks, shape)
            data = np.concatenate((data, fg_distance_map, bg_distance_map), axis=0)
            print('size after sampling clicks', data.shape)
            properties['fg_clicks_index'] = fg_clicks
            properties['bg_clicks_index'] = bg_clicks
            properties['use_nonzero_mask_for_norm'][4] = False
            properties['use_nonzero_mask_for_norm'][5] = False

        assert len(self.normalization_scheme_per_modality) == len(data), "self.normalization_scheme_per_modality " \
                                                                         "must have as many entries as data has " \
                                                                         "modalities"
        assert len(self.use_nonzero_mask) == len(data), "self.use_nonzero_mask must have as many entries as data" \
                                                        " has modalities"

        print("normalization...")

        for c in range(len(data)):
            scheme = self.normalization_scheme_per_modality[c]
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                mean_intensity = self.intensityproperties[c]['mean']
                std_intensity = self.intensityproperties[c]['sd']
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert self.intensityproperties is not None, "ERROR: if there is a CT then we need intensity properties"
                lower_bound = self.intensityproperties[c]['percentile_00_5']
                upper_bound = self.intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                if use_nonzero_mask[c]:
                    data[c][seg[-1] < 0] = 0
            elif scheme == 'noNorm':
                pass
            else:
                if data[c].min() == data[c].max() == 255:
                    # to deal with constant distance map (very rare actually)
                    # normalize to 1
                    data[c] /= 255
                else:
                    if use_nonzero_mask[c]:
                        mask = seg[-1] >= 0
                    else:
                        mask = np.ones(seg.shape[1:], dtype=bool)
                    data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                    data[c][mask == 0] = 0
        print("normalization done")
        return data, seg, properties


def find_valid_click_position(size, idx, margin, step_size):
    assert len(size) == len(idx)
    num_dim = len(size)

    def ismember(a, b):
        # return True/False for each element in a which indicates whether this element is in b
        ls = [True if i in b else False for i in a]
        return np.array(ls)

    valid_idx = np.bool_(np.ones_like(idx[0]))
    for i, s in enumerate(size):
        maxi = min(max(idx[i]), s - margin)
        mini = max(min(idx[i]), margin)
        candidate = range(mini, maxi, step_size)
        valid = np.intersect1d(candidate, idx[i])
        valid_idx_this_axis = ismember(idx[i], valid)
        valid_idx = np.logical_and(valid_idx, valid_idx_this_axis)
    valid_num = np.sum(valid_idx)
    output = []
    if valid_num:
        shuffle_index = np.random.permutation(range(valid_num))
        for i in range(num_dim):
            output_this_axis = idx[i][valid_idx]
            output.append(output_this_axis[shuffle_index])
        assert len(output) == num_dim
        assert len(output[0]) == valid_num
        output = np.array(output).T
    return output, valid_num


def create_clicks(candidates, num_clicks, gt, strategy):
    if strategy == 'sequential':
        click_positions = click_seq(candidates, num_clicks, gt)
    elif strategy == 'random':
        click_positions = click_rnd(candidates, num_clicks)
    else:
        raise NotImplementedError
    return click_positions


def click_seq(candidates, num_clicks, gt):
    clicks = []
    ndim = candidates.shape[1]
    assert ndim in [2, 3]
    first = np.random.choice(range(candidates.shape[0]))
    clicks.append(candidates[first])
    if ndim == 2:
        gt[candidates[first][0], candidates[first][1]] = 0
    else:
        gt[candidates[first][0], candidates[first][1], candidates[first][2]] = 0
    num_clicks -= 1

    while num_clicks:
        dist = distance_transform_bf(gt)
        dist_candidates = np.array([dist[i[0], i[1]] if ndim == 2 else dist[i[0], i[1], i[2]] for i in candidates])
        if dist_candidates.max() == 0:
            break
        index = np.argmax(dist_candidates)
        if ndim == 2:
            gt[candidates[index][0], candidates[index][1]] = 0
        else:
            gt[candidates[index][0], candidates[index][1], candidates[index][2]] = 0
        clicks.append(candidates[index])
        num_clicks -= 1
    return np.array(clicks)


def click_rnd(candidates, num_clicks):
    length, ndim = candidates.shape
    choice = np.random.choice(range(length), num_clicks)
    return candidates[choice, :]


if __name__ == "__main__":
    normalization_scheme = dict()
    use_nonzero_mask = dict()
    for i in range(6):
        use_nonzero_mask[i] = True
        normalization_scheme[i] = 'nonCT'
    use_nonzero_mask[4] = False
    use_nonzero_mask[5] = False
    transpose_forward = [0, 1, 2]
    intensity_prop = None
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl", "--planner", type=str, default="ExperimentPlanner_Interactive",
                        help="Name of the ExperimentPlanner class for interactive learning. "
                             "Default is ExperimentPlanner_Interactive. Can be 'None', in which case these U-Nets will"
                             "not be configured")
    parser.add_argument("--model_type", type=str, default="2d",
                        help="Choose between 2d and 3d network structures, default is 2d")
    parser.add_argument("-no_pp", action="store_true",
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
    parser.add_argument('-d_margin', type=int, default=3,
                        help='minimal distance from negative clicks to ground truth boundaries')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='the number of image,user interaction pairs')
    parser.add_argument('--slice_ratio', type=float, default=0.1,
                        help='the number of slices across z axis used to generate clicks')
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

    preprocessor = PreprocessorForInteractive3D(normalization_scheme, use_nonzero_mask, transpose_forward,
                                                intensity_prop, args_dict)
    img_index = 5
    img_name = str(img_index).rjust(5, '0')
    data_path = f"../../../dataset/nnUNet_cropped_data/Task500_BraTS2021/BraTS2021_{img_name}.npz"
    data = np.load(data_path)['data']
    image = data[:-1]
    seg = data[-1:]
    all_cls = [1, 2, 3]
    bg_cls = [i for i in all_cls if i not in preprocessor.foreground_cls]
    for c in preprocessor.foreground_cls:
        seg[seg == c] = 1
    for c in bg_cls:
        seg[seg == c] = 0
    print('Sample clicks...')
    fg_clicks, bg_clicks = preprocessor.sample_clicks(seg)
    shape = seg.shape
    fg_distance_map, fg_click_map = preprocessor.generate_distance_map(fg_clicks, shape)
    bg_distance_map, bg_click_map = preprocessor.generate_distance_map(bg_clicks, shape)

    import matplotlib.pyplot as plt
    from scipy import ndimage


    def extract_edge(gt):
        ### extract the object boundaries
        gt = ndimage.binary_fill_holes(gt)  # for visualization
        dt = ndimage.distance_transform_edt(gt)
        # edge = np.logical_and(dt <= 1, gt)
        edge = dt == 1.
        edge_x, edge_y = np.nonzero(edge)
        return edge_x, edge_y


    dimz = shape[1]
    num_z_index = int(args_dict['slice_ratio'] * dimz)
    z_index = np.int_(np.linspace(start=0, stop=dimz - 1, num=num_z_index))
    print(z_index)
    for z in z_index:
        fig, axes = plt.subplots(2, 4)
        bg_points = bg_clicks[bg_clicks[:, 0] == z][:, 1:].T
        bg_x, bg_y = bg_points[0], bg_points[1]
        fg_points = fg_clicks[fg_clicks[:, 0] == z][:, 1:].T
        fg_x, fg_y = fg_points[0], fg_points[1]
        seg_slice = seg[0, z, :, :]
        edge_x, edge_y = extract_edge(seg_slice)
        for i in range(8):
            axes[i // 4, i % 4].axis('off')
            if i != 7:
                # horizontal y,vertical x in our coordinates
                axes[i // 4, i % 4].scatter(bg_y, bg_x, s=5, c='red', marker='x')
                axes[i // 4, i % 4].scatter(fg_y, fg_x, s=5, c='blue', marker='x')
        for i in range(4):
            axes[i // 4, i % 4].imshow(image[i, z, :, :], cmap='gray')
            axes[i // 4, i % 4].scatter(edge_y, edge_x, c='yellow', s=0.5)

        axes[1, 0].imshow(seg_slice)
        axes[1, 1].imshow(fg_distance_map[0, z, :, :])
        axes[1, 2].imshow(bg_distance_map[0, z, :, :])
        plt.show()

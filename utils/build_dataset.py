import argparse
import h5py
import glob
import os
import pandas as pd

from joblib import delayed, Parallel


def get_arguments():

    parser = argparse.ArgumentParser(description='make csv files for dataset')
    parser.add_argument('dataset', type=str, help='make dataset(csvfile) name')
    parser.add_argument('dataset_dir', type=str, help='path to videos(folders with frame)')
    parser.add_argument('video_format', type=str, help='the video format[hdf5, jpg or png]')
    parser.add_argument('--split_name', type=str, default=None, help='if a dataset with split, specify them')
    parser.add_argument('--n_jobs', type=int, default=-1, help='the number of cores to load data')
    parser.add_argument('--save_path', type=str, default='./csv', help='path where you want to save csv files')
    return parser.parse_args()


def check_n_frames(idx, video_path, dataset_dir, video_format):
    # idx is for sorting list in the same order as path
    if video_format == 'hdf5':
        path = os.path.join(dataset_dir, video_path)

        with h5py.File(path, 'r') as f:
            video_data = f['video']
            n_frames = len(video_data)
    else:
        imgs = glob.glob(os.path.join(
            dataset_dir, video_path, '*.{}'.format(video_format)))
        n_frames = len(imgs)

    return idx, n_frames


def main():
    args = get_arguments()

    # get dataset path 
    if args.video_format == 'hdf5':
        paths = glob.glob(os.path.join(args.dataset_dir, '*.hdf5'))
    else:
        paths = glob.glob(os.path.join(args.dataset_dir, '*'))

    # extract only video name from paths
    paths = [os.path.relpath(path, args.dataset_dir) for path in paths]

    # check how much frames
    # n_frames = [(0, frame_num), (1, frame_num), ...]
    n_frames = Parallel(n_jobs=args.n_jobs)([
        delayed(check_n_frames)(
            i, paths[i], args.dataset_dir, args.video_format)
        for i in range(len(paths))
    ])

    # sort frames by idx
    n_frames.sort(key=lambda x: x[0])
    n_frames = [x[1] for x in n_frames]

    # df of video name and frame_num
    df = pd.DataFrame({
        "video": paths,
        "n_frames": n_frames
    })
    # remove videos where the number of frames is smaller than 16
    df = df[df['n_frames'] >= 16]

    # save csv
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.split_name is not None:
        df.to_csv(
            os.path.join(
                args.save_path,
                '{}_{}.csv'.format(args.dataset, args.split_name)
            ), index=None)
    else:
        df.to_csv(
            os.path.join(
                args.save_path,
                '{}.csv'.format(args.dataset)
            ), index=None)
    print('Done!')


if __name__ == '__main__':
    main()

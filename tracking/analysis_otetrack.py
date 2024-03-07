import _init_paths
import matplotlib.pyplot as plt
import argparse
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results,print_results_per_video
from lib.test.evaluation import get_dataset, trackerlist

def run_analyse(tracker_name,tracker_param, dataset_name, test_checkpoint = None):
    trackers = []
    # dataset_name = 'lasot_extension_subset'
    #dataset_name = 'lasot'
    # dataset_name = 'got10k_val'
    # dataset_name = 'got10k_test'
    # ar
    if tracker_name == 'otetrack':
        trackers.extend(trackerlist(name='otetrack', parameter_name=tracker_param, dataset_name=dataset_name,
                                run_ids=None, test_checkpoint = test_checkpoint, display_name='OTETrack_256'))
        
    dataset = get_dataset(dataset_name)

    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))

    # run_per_video
    # print_results_per_video(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),per_video=True)

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--test_checkpoint', type=str, default=None)

    args = parser.parse_args()

    # try:
    #     seq_name = int(args.sequence)
    # except:
    #     seq_name = args.sequence

    run_analyse(args.tracker_name,  args.tracker_param ,args.dataset_name, test_checkpoint = args.test_checkpoint)


if __name__ == '__main__':
    main()




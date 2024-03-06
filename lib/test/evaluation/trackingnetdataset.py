import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os
from lib.test.utils.load_text import load_text


class TrackingNetDataset(BaseDataset):
    """ TrackingNet test set.

    Publication:
        TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.
        Matthias Mueller,Adel Bibi, Silvio Giancola, Salman Al-Subaihi and Bernard Ghanem
        ECCV, 2018
        https://ivul.kaust.edu.sa/Documents/Publications/2018/TrackingNet%20A%20Large%20Scale%20Dataset%20and%20Benchmark%20for%20Object%20Tracking%20in%20the%20Wild.pdf

    Download the dataset using the toolkit https://github.com/SilvioGiancola/TrackingNet-devkit.
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.trackingnet_path

        sets = 'TEST'
        if not isinstance(sets, (list, tuple)):
            if sets == 'TEST':
                sets = ['TEST']
            elif sets == 'TRAIN':
                sets = ['TRAIN_{}'.format(i) for i in range(5)]

        self.sequence_list = self._list_sequences(self.base_path, sets)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(set, seq_name) for set, seq_name in self.sequence_list])

    def _construct_sequence(self, set, sequence_name):
        anno_path = '{}/{}/anno/{}.txt'.format(self.base_path, set, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        frames_path = '{}/{}/frames/{}'.format(self.base_path, set, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'trackingnet', ground_truth_rect.reshape(-1, 4))
    
    def __len__(self):
        return len(self.sequence_list)

    def _list_sequences(self, root, set_ids):
        sequence_list = []

        test_list_special = ['0JOSlkhPOdA_0',
                             '3ynVtFJmIfk_0',
                             '0l7j7i3XhJ0_0',
                             '6xzaKqU-rwI_0',
                             'fAzgoRh2yP0_0'                      
        ]

        test_list = ['0JOSlkhPOdA_0',
        '0l7j7i3XhJ0_0',
        '3jdzVaWohVw_0',
        '3ynVtFJmIfk_0', 
        '4e0D1OyvPrI_0', 
        '6xzaKqU-rwI_0', 
        '8fiL0-tqkRA_0', 
        'A7OzWjZpCWs_0',
        'aAsiYXsj28E_0',
        'AVuFw6MIACg_0']

        new_list = [
            'B_-FCqaj4oc_0',
            'fAzgoRh2yP0_0',
            'fMR7bO9fQMc_0', 
            'fq_rMea3B9s_0',
            '0GER2Qd0vFw_0', 
            '0jgHdaQXpRk_0', 
            '0nlxpC_f0wU_0',
            '0ZzhXi15dvo_0', 
            '1aAJHdfrJuk_0', 
            '1n0JQ2qIqLo_1', 
            '1n0JQ2qIqLo_2', 
            '1qvKZsLFCX4_0', 
            '1s7hqoYecSo_0', 
            '2DwR0E7MySc_0', 
            '2e5XiuDEo5A_0', 
            '2P0ok6kGdPk_0', 
            '2P0ok6kGdPk_1', 
            '2WTV7g1Z0lA_0', 
            '4qXKgKaCd3s_0', 
            '4rT02vTH8qg_0', 
            '4XNrBaxkiHw_0',
            '5AHb4xPDFR8_0', 
            '5RJXgYSJaVE_0', 
            '8sd513xQzV4_0', 
            '8VkHx1GXvmo_0', 
            '9HizwmZHguc_0', 
            '9RmS4wETvRA_0', 
            '9XfvirWNWZA_0', 
            '27HbwIQV92c_0', 
            '36_slnYU-EA_0', 
        ]
        # test_list = test_list + new_list
        test_list = test_list_special
        for s in set_ids:
            anno_dir = os.path.join(root, s, "anno")
            sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if f.endswith('.txt')]
            # sequences_cur_set = [(s, os.path.splitext(f)[0]) for f in os.listdir(anno_dir) if (f.endswith('.txt') and (os.path.splitext(f)[0] in test_list ))]

            sequence_list += sequences_cur_set

        return sequence_list
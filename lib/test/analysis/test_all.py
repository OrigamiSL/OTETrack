import os
import glob
import shutil
import torch
# project_path = os.path.join('../../../',os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

delete_path = project_path+'/output/test'
root = project_path+''
move_path = project_path+'/output/test/tracking_results/otetrack/otetrack_256_full'

checkpoint_all = os.path.join(project_path,'test_checkpoint','OTETrack_all.pth.tar')
checkpoint_got = os.path.join(project_path,'test_checkpoint','OTETrack_got.pth.tar')

print(torch.version.cuda)
dataset_name = ['lasot','lasot_extension_subset','uav','got10k_test','trackingnet']

for dataset in dataset_name:
        
    if dataset == 'got10k_test' or dataset == 'got10k_val':
        checkpoint = checkpoint_got
    else:
        checkpoint = checkpoint_all

    print(checkpoint)
    test_command = 'python tracking/test.py otetrack otetrack_256_full --dataset %s --threads 0 --num_gpus 1 --test_checkpoint %s' \
            %(dataset,checkpoint)
    
    os.system(test_command)

    if dataset == 'trackingnet':
        transform_cmd = 'python lib/test/utils/transform_trackingnet.py --tracker_name otetrack --cfg_name otetrack_256_full'
        os.system(transform_cmd)
    elif dataset == 'got10k_test':
        transform_cmd = 'python lib/test/utils/transform_got10k.py --tracker_name otetrack --cfg_name otetrack_256_full'
        os.system(transform_cmd)
    else:
        analyse = 'python tracking/analysis_otetrack.py otetrack otetrack_256_full --dataset %s --test_checkpoint %s'%(dataset,checkpoint)
        
        with open('test_result.txt', 'a') as f :
            f.write(checkpoint) 
            f.write('\n')
            f.write('total') 
            f.write('\n')
        os.system(analyse)
    
    root_test = os.path.join(root,'all_test_result',dataset,checkpoint.split('/')[-1])
    
    if not os.path.exists(root_test):
        os.makedirs(root_test)
    filelist = os.listdir(move_path)
    for file in filelist:
        shutil.move(os.path.join(move_path,file),os.path.join(root_test,file))
    shutil.rmtree(delete_path) 

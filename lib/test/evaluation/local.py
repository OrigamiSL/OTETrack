from lib.test.evaluation.environment import EnvSettings
import os
# Set your local paths here.
def local_env_settings():
    settings = EnvSettings()

    # root path
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # dataset path
    settings.got10k_path = '/home/lhg/work/data/' 
    settings.lasot_extension_subset_path = '/home/lhg/work/data/track_data/data/LaSOT_Ext'
    settings.lasot_path = '/home/lhg/work/ssd/LaSOTTest/LaSOTTest'
    settings.tnl2k_path = '/home/lhg/work/data/tnl2k'
    settings.trackingnet_path = '/home/lhg/work/data/track_data/trackingnet/TEST/'
    settings.uav_path = '/home/lhg/work/data/track_data/trackingnet/UAV123/Dataset_UAV123/UAV123'
    settings.nfs_path = '/home/lhg/work/ssd/Nfs'
    settings.network_path = os.path.join(project_path,'output/test/networks' )   # Where tracking networks are stored.
    
    # save path
    settings.prj_dir = project_path
    settings.result_plot_path = os.path.join(project_path,'output/test/result_plots') 
    settings.results_path = os.path.join(project_path,'output/test/tracking_results') 
    settings.save_dir = os.path.join(project_path,'output')  
    settings.segmentation_path = os.path.join(project_path,'output/test/segmentation_results' )
    
    return settings


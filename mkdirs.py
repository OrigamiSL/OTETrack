import os

root_dir = os.path.dirname(os.path.abspath(__file__))
# print( root_dir )
pretrained_models_path = os.path.join(root_dir,'pretrained_models')
test_checkpoint_path = os.path.join(root_dir,'test_checkpoint')

if not os.path.exists(pretrained_models_path):
    os.makedirs(pretrained_models_path)

if not os.path.exists(test_checkpoint_path):
    os.makedirs(test_checkpoint_path)

from prototree.prototree import ProtoTree
from util.data import get_dataloaders
from util.visualize_prediction import gen_pred_vis
import torch
import torchvision.transforms as transforms
from PIL import Image
from shutil import copy
from copy import deepcopy
import os
import argparse

# `main_explain_local.py --log_dir ./runs/protoree_cars --dataset CARS --sample_dir ./data/cars/dataset/test/Dodge_Sprinter_Cargo_Van_2009/04003.jpg 
# --prototree ./runs/protoree_cars/checkpoints/pruned_and_projected`

# Directly assign values to your arguments
prototree_dir = './runs/protoree_cars/checkpoints/pruned_and_projected'
log_dir = './runs/run_prototree_xray'
# dataset = 'CUB-200-2011'
sample_dir = '/home/shaijal/ProtoTree/data/xrays/xrays_test/Edema/patient00157_study1_view1_frontal.jpg'
results_dir = './runs/run_prototree_xray/local_explanations'
disable_cuda = False
image_size = 224
dir_for_saving_images = './runs/run_prototree_xray/local_explanations/upsampling_results'
upsample_threshold = 0.98

def explain_local(args):
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')

    # Log which device was actually used
    print('Device used: ', str(device))

    # Load trained ProtoTree
    tree = ProtoTree.load(args.prototree).to(device=device)
    # Obtain the dataset and dataloaders
    args.batch_size = 64  # placeholder
    args.augment = True  # placeholder
    _, _, _, classes, _ = get_dataloaders(args)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transform_no_augment = transforms.Compose([
        transforms.Resize(size=(args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])

    sample = test_transform(Image.open(args.sample_dir)).unsqueeze(0).to(device)

    gen_pred_vis(tree, sample, args.sample_dir, args.results_dir, args, classes)


# Call the function directly with the provided values
args = argparse.Namespace(
    prototree=prototree_dir,
    log_dir=log_dir,
    # dataset=dataset,
    sample_dir=sample_dir,
    results_dir=results_dir,
    disable_cuda=disable_cuda,
    image_size=image_size,
    dir_for_saving_images=dir_for_saving_images,
    upsample_threshold=upsample_threshold
)

try:
    Image.open(args.sample_dir)
    print("Image to explain: ", args.sample_dir)
    explain_local(args)
except:  # folder is not an image
    class_name = args.sample_dir.split('/')[-1]
    if not os.path.exists(os.path.join(os.path.join(args.log_dir, args.results_dir), class_name)):
        os.makedirs(os.path.join(os.path.join(args.log_dir, args.results_dir), class_name))
    for filename in os.listdir(args.sample_dir):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            args_1 = deepcopy(args)
            args_1.sample_dir = args.sample_dir + "/" + filename
            args_1.results_dir = os.path.join(args.results_dir, class_name)
            explain_local(args_1)

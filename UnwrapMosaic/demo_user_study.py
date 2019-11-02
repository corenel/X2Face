import numpy as np
import torch.nn as nn
import os
import torch
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale

VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
VALID_VIDEO_EXTS = ('.mp4', '.avi', '.mov')


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in VALID_IMAGE_EXTS


def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VALID_VIDEO_EXTS


def is_valid_file(path):
    return is_image_file(path) or is_video_file(path)


def list_files(path, filter_func=is_valid_file):
    if os.path.isdir(path):
        # return [os.path.abspath(f) for f in os.listdir(path)]
        return sorted([
            os.path.abspath(os.path.join(root, f))
            for root, dirs, files in os.walk(path) for f in files
            if filter_func(f)
        ])
    elif os.path.isfile(path) and filter_func(path):
        return [path]
    else:
        return list()


def load_img(file_path):
    img = Image.open(file_path).convert('RGB')
    transform = Compose([Scale((256, 256)), ToTensor()])
    return Variable(transform(img)).cuda()


def run_batch(source_images, pose_images, requires_grad=False, volatile=False):
    return model(pose_images, *source_images)


def run_demo(source_img_path, driving_img_path, save_root):
    # Driving the source image with the driving sequence
    source_images = [load_img(source_img_path).unsqueeze(0).repeat(1, 1, 1, 1)]

    driving_images = load_img(driving_img_path).unsqueeze(0)

    # Run the model for each
    result = run_batch(source_images, driving_images)
    result = result.clamp(min=0, max=1)

    # image_list = [s.cpu().data for s in source_images]
    # image_list.extend([driving_images.cpu().data, result.cpu().data])
    # image_grid = torchvision.utils.make_grid(torch.cat(image_list),
    #                                          nrow=len(driving_images))

    save_path = os.path.join(save_root,
                             os.path.relpath(source_img_path, source_path))
    save_dir = os.path.dirname(save_path)
    print(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.utils.save_image(result.cpu().data[0], save_path)


def replace_path(path):
    return path.replace('/media/yuthon/Data', '/app/data')


BASE_MODEL = '../release_models'  # Change to your path
state_dict = torch.load(os.path.join(BASE_MODEL, 'x2face_model.pth'))

model = UnwrappedFaceWeightedAverage(output_num_channels=2,
                                     input_num_channels=3,
                                     inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
model = model.cuda()

model = model.eval()

dataset_mode = 'vox2'
source_path = '/app/data/user_study/real/{}'.format(dataset_mode)
driving_path = '/app/data/user_study/input_label/{}'.format(dataset_mode)
result_root = '/app/data/user_study/synthesized/{}/x2face_{}'.format(
    dataset_mode, dataset_mode)

source_imgs = list_files(source_path, filter_func=is_image_file)
driving_imgs = list_files(driving_path, filter_func=is_image_file)
print('# source data points: {}'.format(len(source_imgs)))
print('# driving data points: {}'.format(len(source_imgs)))

for image_src, image_gt in tqdm(zip(source_imgs, driving_imgs)):
    run_demo(image_src, image_gt, result_root)

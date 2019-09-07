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

BASE_MODEL = '../release_models'  # Change to your path
state_dict = torch.load(os.path.join(BASE_MODEL, 'x2face_model.pth'))

model = UnwrappedFaceWeightedAverage(output_num_channels=2,
                                     input_num_channels=3,
                                     inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
model = model.cuda()

model = model.eval()

# driver_path = './examples/Taylor_Swift/1.6/nuBaabkzzzI/'
# source_path = './examples/Taylor_Swift/1.6/YqaVWHmGgtI/'

# driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))
#               ][0:8]  # 8 driving frames
# source_imgs = [source_path + d for d in sorted(os.listdir(source_path))
#               ][0:3]  # 3 source frames

# image_id = '10'
# driver_imgs = ['./inter_id/{}_gt.jpg'.format(image_id)]  # 1 driving frames
# source_imgs = ['./inter_id/{}_src.jpg'.format(image_id)]  # 1 source frames


def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Scale((256, 256)), ToTensor()])
    return Variable(transform(img)).cuda()


def run_batch(source_images, pose_images, requires_grad=False, volatile=False):
    return model(pose_images, *source_images)


def run_demo(source_imgs, driver_imgs, index, save_dir):
    # Driving the source image with the driving sequence
    source_images = []
    for img in source_imgs:
        source_images.append(
            load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))

    driver_images = None
    for img in driver_imgs:
        if driver_images is None:
            driver_images = load_img(img).unsqueeze(0)
        else:
            driver_images = torch.cat(
                (driver_images, load_img(img).unsqueeze(0)), 0)

    # Run the model for each
    result = run_batch(source_images, driver_images)
    result = result.clamp(min=0, max=1)

    image_list = [s.cpu().data for s in source_images]
    image_list.extend([driver_images.cpu().data, result.cpu().data])
    image_grid = torchvision.utils.make_grid(torch.cat(image_list),
                                             nrow=len(driver_images))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.utils.save_image(
        driver_images.cpu().data[0],
        os.path.join(save_dir, '{}_gt.jpg'.format(index)))
    torchvision.utils.save_image(
        source_images[0].cpu().data,
        os.path.join(save_dir, '{}_src.jpg'.format(index)))
    torchvision.utils.save_image(
        image_grid, os.path.join(save_dir, '{}_grid.jpg'.format(index)))
    torchvision.utils.save_image(
        result.cpu().data[0],
        os.path.join(save_dir, '{}_x2face.jpg'.format(index)))


def replace_path(path):
    return path.replace('/media/data1/vox', '/app/data/vox')


if __name__ == '__main__':
    with open('eval_list.txt', 'r') as f:
        eval_list = [l.strip().split(' ') for l in f.readlines()]
    for image_idx, (image_src, image_gt) in tqdm(enumerate(eval_list)):
        run_demo([replace_path(image_src)], [replace_path(image_gt)],
                 '{:05d}'.format(image_idx), 'eval_results')

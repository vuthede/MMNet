from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm
import cv2
import subprocess

def get_alpha_channel(img_path):
    rgba = Image.open(img_path)
    alpha = rgba.getchannel("A")
    alpha = np.array(alpha)
    assert alpha.shape == (800,600), "Shape image should be (800,600)"
    return alpha

def prepare_all_mask_data_and_image_data(matting_human_half_dir, output_img_dir, output_mat_dir):
    files_img = glob.glob(f'{matting_human_half_dir}/clip_img/*/*/*.jpg')
    files_mat = glob.glob(f'{matting_human_half_dir}/matting/*/*/*.png')

    for f_im, f_mat in tqdm(zip(files_img, files_mat)):
        alpha = get_alpha_channel(f_mat)
        os.makedirs(output_mat_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)
        cv2.imwrite(f'{output_mat_dir}/{os.path.basename(f_mat).replace(".png",".jpg")}', alpha)
        subprocess.call(['cp', f'{f_im}', f'{output_img_dir}/{os.path.basename(f_im)}'])

    
prepare_all_mask_data_and_image_data(matting_human_half_dir="/home/ubuntu/matting_data/kaggle_data/matting_human_half",\
                      output_img_dir="/home/ubuntu/matting_data/kaggle_data/matting_human_half/all_img",\
                      output_mat_dir="/home/ubuntu/matting_data/kaggle_data/matting_human_half/all_mat")

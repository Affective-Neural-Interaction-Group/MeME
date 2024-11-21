import os
import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt
import legacy
import numpy as np
import pickle
from tqdm import tqdm
from dataclasses import dataclass
import pyrallis
import ast
import pandas as pd

@dataclass
class EditConfig:
    # enter the trial number
    trial : str = '6251'
    # enter the number of subjects to average
    avg_num: int = 2

@pyrallis.wrap()
def run(opts: EditConfig):
    gen_avg(trial=opts.trial, avg_num=opts.avg_num)

def lerp(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)

def interpolate(latent_code_1, latent_code_2, num_interps):
    step_size = 1.0/num_interps
    amounts = np.arange(0, 1, step_size)
    interpolated_codes = []
    for seed_idx, alpha in enumerate(tqdm(amounts)):
        interpolated_code = lerp(latent_code_2, latent_code_1, alpha)
        interpolated_codes.append(interpolated_code)
    return interpolated_codes


def gen_avg(trial, avg_num):   
    SAVE_PATH = f'your_directory/{avg_num}avg_{trial}'
    LATENT_DIR = f'directory_of_latent_codes_of_generated_images/{trial}'

    # Load the pretrained the StyleGAN model 
    generator_path = './stylegan3-r-ffhq-1024x1024.pkl'
    print('Loading generator from "%s"...' % generator_path)
    device = torch.device('cuda')
    with open(generator_path, "rb") as f:
        generator = pickle.load(f)['G_ema'].cuda()

    def generate_image(z, device=device):
        z = torch.from_numpy(z).unsqueeze(0).to(device)
        label = torch.zeros([1, generator.c_dim], device=device)
        w = generator.mapping(z, label, truncation_psi=0.8)
        img = generator.synthesis(w, noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        img.save(SAVE_PATH + f'image_{i}.png', format='png')
        return img

    subject_list = ['6002', '6003', '6004', '6005', '6006', '6009', '6010', '6012', '6013', '6018', '6020', '6021', '6022', '6023', '6024', '6025', '6027', '6029', '6030', '6031', '6032', '6033', '6034', '6035', '6036', '6037', '6038', '6039', '6040']
    trial_names = ['1010', '355', '1172', '1111', '1156', '1223', '193', '209', '439', '908', '587', '2220', '6251', '269', '1160']
    df_train = pd.read_csv('directory_of_your_training_resluts.csv')
    df_test = pd.read_csv('/directory_of_your_testing_resluts.csv')
    df_train['trial'] = trial_names
    df_test['trial'] = trial_names
    df_train_trial = df_train[df_train['trial'] == trial]
    df_test_trial = df_test[df_test['trial'] == trial]

    avg_code = []
    trial_list = []
    for i in range(avg_num):
        subject = subject_list[i]
        print(subject)

        list1 = df_train_trial[f'{subject}_1_allchansok'].tolist()
        list2 = df_test_trial[f'{subject}_1_allchansok'].tolist()
        result1 = ast.literal_eval(list1[0])
        result2 = ast.literal_eval(list2[0])

        trials = label_dic[f't{trial}']
        trial_list.extend(trials)
        
        check_list = result1 + result2

        label_dic = pickle.load(open(f'/home/yc47480/mc15539/sub_list/{subject}_label_list.pkl', 'rb'))
        trial_list = label_dic[f't{trial}']

        latent_codes = []
        detected_list = []
        for x in trial_list:
            if x in check_list:
                detected_list.append(x)

        for y in detected_list:
            path = os.path.join(LATENT_DIR, f'{y}.npy')
            # print(path)
            if not os.path.exists(path):
                continue
            else:
                with open(path, 'rb') as f:
                    latent_code = np.load(f)
                    latent_codes.extend(latent_code)

        all_interpolations = []
        total_steps = 200 # total fine steps between codes
        step_at = 60 # start the next interpolation from this step

        if len(latent_codes) > 1:
            interpolated_code = latent_codes[0]  # Set initial code

            for j in range(1, len(latent_codes)):
                # Interpolate from the last stepped latent code to the next one in the list
                fine_interpolations = interpolate(interpolated_code, latent_codes[j], total_steps)

                # Select only every "step_at"-th interpolated code
                selected_interpolations = fine_interpolations[::step_at]
                all_interpolations.extend(selected_interpolations)
            
                if selected_interpolations:
                    interpolated_code = selected_interpolations[-1]
        # print(all_interpolations)
        avg_code.append(all_interpolations[-1])
        print(label_dic)

    final_interp = []
    if len(avg_code) > 1:
        interp_code = avg_code[0]
    for k in range(1, avg_num):
        avg_interp = interpolate(interp_code, avg_code[k], 200) 
        selected_interp = avg_interp[::59]
        final_interp.extend(selected_interp)
            
        if selected_interp:
            interp_code = selected_interp[-1]  
    for code in tqdm(final_interp):
        image = generate_image(code)

if __name__ == '__main__':
    run()

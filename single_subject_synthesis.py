import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import ast
import torch
import PIL.Image

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

def generate_image(z):
    z = torch.from_numpy(z).unsqueeze(0).to(device)
    label = torch.zeros([1, generator.c_dim], device=device)
    w = generator.mapping(z, label, truncation_psi=0.8)
    img = generator.synthesis(w, noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    img.save(SAVE_PATH + f'image_{i}.png', format='png')
    return img


SAVE_PATH = 'define_save_path_here'
# Load the pretrained the StyleGAN model 
generator_path = './stylegan3-r-ffhq-1024x1024.pkl'
print('Loading generator from "%s"...' % generator_path)
device = torch.device('cuda')
with open(generator_path, "rb") as f:
    generator = pickle.load(f)['G_ema'].cuda()


trial_names = ['1010', '355', '1172', '1111', '1156', '1223', '193', '209', '439', '908', '587', '2220', '6251', '269', '1160']
all_sub_dist = {}
dist_dict = {}
df_train = pd.read_csv('directory_of_your_training_results')
df_test = pd.read_csv('directory_of_your_testing_results')
df_train['trial'] = trial_names
df_test['trial'] = trial_names
subject_list = ['6002', '6003', '6004', '6005', '6006', '6009', '6010', '6012', '6013', '6018', '6020', '6021', '6022', '6023', '6024', '6025', '6027', '6029', '6030', '6031', '6032', '6033', '6034', '6035', '6036', '6037', '6038', '6039', '6040']
for subject in subject_list:
    print(subject)
    for j in trial_names:
        list1 = df_train[f'{subject}_1_allchansok'].loc[df_train['trial'] == j].tolist()
        list2 = df_test[f'{subject}_1_allchansok'].loc[df_test['trial'] == j].tolist()
        result1 = ast.literal_eval(list1[0])
        result2 = ast.literal_eval(list2[0])
        check_list = result1 + result2
        label_dic = pickle.load(open(f'Use_the_given_label_lists/{subject}_label_list.pkl', 'rb'))
        trial_list = label_dic[f't{j}']

        LATENT_DIR = f'directory_of_latent_codes/{j}'
        print(LATENT_DIR)
        source_code_dir = f'directory_of_latent_codes/{j}/t000.npy'
        with open(source_code_dir, 'rb') as s:
            source_code = np.load(s)

        latent_codes = []
        detected_list = []
        for i in trial_list:
            if i in check_list:
                detected_list.append(i)

        for i in detected_list:
            path = os.path.join(LATENT_DIR, f'{i}.npy')
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

        for i in range(1, len(latent_codes)):
            # Interpolate from the last stepped latent code to the next one in the list
            fine_interpolations = interpolate(interpolated_code, latent_codes[i], total_steps)

            # Select only every "step_at"-th interpolated code
            selected_interpolations = fine_interpolations[::step_at]
            all_interpolations.extend(selected_interpolations)
        
            if selected_interpolations:
                interpolated_code = selected_interpolations[-1]

        final_code = all_interpolations[-1]
        img = generate_image(final_code)
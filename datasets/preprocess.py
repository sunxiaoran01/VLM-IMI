import os
from natsort import natsorted


def write_image_prompt_paths_to_txt(high_image_folder, low_image_folder, prompt_folder, output_file):
    high_image_files = natsorted(os.listdir(high_image_folder))
    low_image_files = natsorted(os.listdir(low_image_folder))
    prompt_files = natsorted(os.listdir(prompt_folder))

    with open(output_file, 'w') as f:
        for i in range(len(high_image_files)):
            high_image_path = os.path.join(high_image_folder, high_image_files[i])
            low_image_path = os.path.join(low_image_folder, low_image_files[i])
            prompt_path = os.path.join(prompt_folder, prompt_files[i])

            f.write(f'{high_image_path} {low_image_path} {prompt_path}\n')
            print(f'Added: {high_image_path} {low_image_path} {prompt_path}')


high_image_folder = '../data/lowlight/high'
low_image_folder = '../data/lowlight/low'
prompt_folder = '../data/lowlight/text'
output_txt_file = '../data/lowlight_train.txt'

write_image_prompt_paths_to_txt(high_image_folder, low_image_folder, prompt_folder, output_txt_file)
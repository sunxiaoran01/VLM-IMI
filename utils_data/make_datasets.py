import os
import re
import torch
from PIL import Image
from PIL import ImageFile
from transformers import AutoProcessor, LlavaForConditionalGeneration
import transformers
transformers.logging.set_verbosity_info()
ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_and_save_descriptions(image_dir, output_folder):
    prompt = "USER: <image>\nPlease answer the following questions to generate a description of the lighting conditions in this image:" \
             "1. What type of light source is present in the image? (e.g., natural light, artificial light)" \
             "2. How intense is the lighting? (e.g., bright, moderate, soft)" \
             "3. How is the lighting distributed in the scene? (e.g., even lighting, spotlighting, local brightness variations)" \
             "4. Are there any noticeable shadows or reflections? If so, what is the direction and shape of the shadows?" \
             "5. Is the position of the light source visible? Where is the light coming from? (e.g., the light source is from the left, right, top)" \
             "6. Describe the objects or subjects in the scene, their spatial relationships and colors. \nASSISTANT:"

    model = LlavaForConditionalGeneration.from_pretrained("pretrained/llava",
                                                          torch_dtype=torch.float16,
                                                          low_cpu_mem_usage=True,
                                                          attn_implementation="flash_attention_2").to(1)
    processor = AutoProcessor.from_pretrained("pretrained/llava")

    image_names = os.listdir(image_dir)
    image_names = sorted(image_names, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for img_name in image_names:
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(image, prompt, return_tensors='pt').to(1, torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
        generated_text = generated_text.split("ASSISTANT:", 1)[1].strip()

        txt_file_name = os.path.splitext(img_name)[0] + '.txt'
        txt_file_path = os.path.join(output_folder, txt_file_name)

        with open(txt_file_path, 'w') as f:
            f.write(generated_text)


image_directory = ''
output_directory = ''

generate_and_save_descriptions(image_directory, output_directory)
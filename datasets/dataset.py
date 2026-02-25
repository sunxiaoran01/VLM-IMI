import os
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        train_dataset = ImageCaptionDataset(self.config.data.data_dir,
                                            filelist='{}_train.txt'.format(self.config.data.train_dataset))
        val_dataset = ImageCaptionDataset(self.config.data.data_dir,
                                          filelist='{}_val.txt'.format(self.config.data.val_dataset))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, dir, filelist=None):
        super().__init__()

        self.dir = dir

        self.file_list = filelist
        self.train_list = os.path.join(dir, self.file_list)   # file_list 包含图像路径和文本路径
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_names)

    def __getitem__(self, idx):
        input_name = self.input_names[idx].replace('\n', '')

        high_image_name, low_image_name, caption_name = input_name.split(' ')[0], input_name.split(' ')[1], input_name.split(' ')[2]
        image_id = high_image_name.split('/')[-1]

        high_image = Image.open(high_image_name).convert('RGB')
        high_image = self.transform(high_image)
        low_image = Image.open(low_image_name).convert('RGB')
        low_image = self.transform(low_image)

        with open(caption_name, 'r', encoding='utf-8') as file:
            caption = file.read().strip()

        example = dict()
        example["condition_image"] = low_image.squeeze(0) * 2.0 - 1.0
        example["image"] = high_image.squeeze(0) * 2.0 - 1.0
        example["caption"] = caption
        example["image_id"] = image_id

        return example
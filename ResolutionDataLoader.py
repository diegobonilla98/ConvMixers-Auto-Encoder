import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import numpy as np
import albumentations as albu


class ArtDataset(Dataset):
    def __init__(self, is_test=False):
        self.transform = transforms.Compose([
                self.ToNormTensor()
            ])
        self.data_folder = '/media/bonilla/My Book/Human_Segmentation'
        self.augmentation = self.get_training_augmentation()

    @staticmethod
    def get_training_augmentation():
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45, shift_limit=0.1, p=1, border_mode=4),
            albu.Resize(224, 224, always_apply=True),
            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5),
            albu.OneOf(
                [
                    albu.CLAHE(p=1),
                    albu.RandomBrightness(p=1),
                    albu.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.IAASharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            albu.OneOf(
                [
                    albu.RandomContrast(p=1),
                    albu.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ]
        return albu.Compose(train_transform)

    @staticmethod
    class ToNormTensor(object):
        def __call__(self, sample):
            image, mask = sample['image'], sample['mask']
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            mask = torch.from_numpy(mask.transpose((2, 0, 1)))
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            return {'image': image,
                    'mask': mask}

    def __len__(self):
        return 34747

    def __getitem__(self, i):
        image_name = np.random.choice(glob.glob(os.path.join(np.random.choice(glob.glob(os.path.join(
            np.random.choice(glob.glob(os.path.join(self.data_folder, 'clip_img', '*'))), '*'))), '*')))
        image = cv2.imread(image_name)[:, :, ::-1]
        # image = cv2.resize(image, (224, 224))[:, :, ::-1].astype(np.float32) / 255.

        mask_name = image_name.replace('clip_img', 'matting').replace('clip_', 'matting_').replace('.jpg', '.png')
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        # mask = cv2.resize(mask, (224, 224))[:, :, 3:].astype(np.float32) / 255.

        sample = self.augmentation(image=image, mask=mask[:, :, 3:].astype(np.float32) / 255.)
        image, mask = sample['image'], sample['mask']

        # plt.figure(0)
        # plt.imshow(image)
        # plt.figure(1)
        # plt.imshow(mask[:, :, 0])
        # plt.show()

        sample = {'image': image.astype(np.float32) / 255., 'mask': mask}
        sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    dl = ArtDataset()
    a = dl[0]
    print()

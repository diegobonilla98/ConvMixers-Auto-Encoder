import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable


USE_CUDA = torch.cuda.is_available()
weights_path = './checkpoints/people_seg_epoch_9.pth'
model = torch.load(weights_path)
print(model)
model = model.eval()
if USE_CUDA:
    model = model.cuda()

cam = cv2.VideoCapture(1)
background = None
while True:
    ret, frame = cam.read()
    if background is None:
        background = cv2.imread('../test_art/il_fullxfull.2842036312_34gb.jpg')
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    image_org = frame.copy()
    image = cv2.resize(image_org[:, :, ::-1].copy(), (224, 224)).astype(np.float32) / 255.
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    if USE_CUDA:
        image = image.cuda()
    image = Variable(image)
    mask_output = model(input_data=image)
    mask_output = mask_output.cpu().data.numpy()[0, 0, :, :]
    mask_output = np.uint8(mask_output * 255.)
    mask_output = cv2.resize(mask_output, (frame.shape[1], frame.shape[0]), cv2.INTER_LANCZOS4)
    mask_output = cv2.cvtColor(mask_output, cv2.COLOR_GRAY2BGR).astype('float32') / 255.

    comb = np.uint8(mask_output * image_org + (1. - mask_output) * background)

    cv2.imshow("Webcam", image_org)
    cv2.imshow("Mask", comb)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

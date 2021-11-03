import torch.nn
import numpy as np
from ModelAE import ConvMixer
from ResolutionDataLoader import ArtDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from torch.autograd import Variable
from LovaszLoss import LovaszLoss


BATCH_SIZE = 10
LEARNING_RATE = 1e-4
USE_CUDA = torch.cuda.is_available()
N_EPOCHS = 100

data_set = ArtDataset()
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

model = ConvMixer()
print(model)

if USE_CUDA:
    model = model.cuda()

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
lambda1 = lambda epoch: 0.65 ** epoch
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

loss = LovaszLoss('BINARY_MODE')
if USE_CUDA:
    loss = loss.cuda()

for p in model.parameters():
    p.requires_grad = True

for epoch in range(N_EPOCHS + 1):
    data_iter = iter(data_loader)
    i = 0
    epoch_errors = []
    while i < len(data_loader):
        sample = next(data_iter)
        s_image, s_mask = sample['image'], sample['mask']

        model.zero_grad()
        if USE_CUDA:
            s_image = s_image.cuda()
            s_mask = s_mask.cuda()

        s_image_v = Variable(s_image)
        s_mask_v = Variable(s_mask)

        s_class_output = model(input_data=s_image_v)
        err = loss(s_class_output, s_mask_v)

        err.backward()
        optimizer.step()

        i += 1
        errr = err.cpu().data.numpy()
        epoch_errors.append(errr)

        print(f'[Epoch: {epoch}/{N_EPOCHS}], [It: {i}/{len(data_loader)}], [Err_label: {errr}]')

    scheduler.step()
    print(f'[Epoch: {epoch}/{N_EPOCHS}], [Err_label: {np.mean(epoch_errors)}]')
    torch.save(model, f'./checkpoints/people_seg_epoch_{epoch}.pth')

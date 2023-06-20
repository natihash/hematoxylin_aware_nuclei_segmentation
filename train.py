import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import glob 
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from custom_dataset import CustomImageDataset

image_names = sorted(glob.glob("Monuseg_dataset/Tissue Images/*"))
hema_names = sorted(glob.glob("modified dataset/hemas/*"))
sema_names = sorted(glob.glob("modified dataset/semas/*"))
mark_names = sorted(glob.glob("modified dataset/markers/*"))
weight_names = sorted(glob.glob("modified dataset/weights/*"))

class MyRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

transform_train = transforms.Compose([transforms.RandomVerticalFlip(),MyRotationTransform([0, 90, 180, 270]),
                                     transforms.RandomHorizontalFlip()])

final_data = CustomImageDataset(image_names, mark_names, sema_names, hema_names, weight_names, transform_train)
train_loader = torch.utils.data.DataLoader(final_data, batch_size=12, shuffle=True)

class weighted_cross_entropy(nn.Module):
    def __init__(self):
        super(weighted_cross_entropy, self).__init__()

    def forward(self, logits, labels, weights):
        cross_entropy_loss = F.cross_entropy(logits, labels, reduction='none')
        weighted_loss = cross_entropy_loss * weights
        loss = torch.mean(weighted_loss)
        return loss
myloss = weighted_cross_entropy()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from models import NestedUNet
model = NestedUNet(num_classes=[1, 3], input_channels=4, deep_supervision=True)
model = nn.DataParallel(model)
model.to(device);

## epoch, learning rate, save model name, delete tensors, lr scedule, 
loss_list = []
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-04, betas=(0.9, 0.999), eps=1e-07)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
model.train()
for i in range(80):
    tot_loss = 0
    count = 0
    for img_, sema_, mark_, weight_ in train_loader:
        img_ = img_.to(device, dtype=torch.float32)
        out1, out2 = model(img_)
        del img_
        
        mark_ = mark_.to(device, dtype=torch.long)
        sema_ = sema_.to(device, dtype=torch.float32)
        weight_ = weight_.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        
        loss1 = F.binary_cross_entropy_with_logits(out1.squeeze(1), sema_, weight_)
        loss2 = myloss(out2, mark_, weight_)
        loss = loss1+loss2
        del sema_, mark_, weight_, out1, out2

        loss.backward()
        
        tot_loss = loss+tot_loss
        optimizer.step()
        count = count+1
        if count%10 == 0:
            print(count, end=" ")
    scheduler.step()
    loss_list.append(tot_loss)
    print("epoch "+str(i)+" ended")
    if torch.isnan(tot_loss):
        break

model.eval();
torch.save(model.state_dict(), "models/trained_weights")
## don't forget to load weights if you are resuming training

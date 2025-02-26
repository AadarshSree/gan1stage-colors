import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import torchvision.models as models
import os
from torch.utils import data
from model import generator
import numpy as np
from PIL import Image
from skimage.color import rgb2yuv,yuv2rgb
import cv2
import matplotlib.pyplot as plt
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN based model")
    parser.add_argument("-d",
                        "--training_dir",
                        type=str,
                        required=True,
                        help="Training directory (folder contains all 256*256 images)")
    parser.add_argument("-t",
                        "--test_image",
                        type=str,
                        default=None,
                        help="Test image location")
    parser.add_argument("-c",
                        "--checkpoint_location",
                        type=str,
                        required=True,
                        help="Place to save checkpoints")
    parser.add_argument("-e",
                        "--epoch",
                        type=int,
                        default=120,
                        help="Epoches to run training")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="which GPU to use?")
    parser.add_argument("-b",
                        "--batch_size",
                        type=int,
                        default=20,
                        help="batch size")
    parser.add_argument("-w",
                        "--num_workers",
                        type=int,
                        default=6,
                        help="Number of workers to fetch data")
    parser.add_argument("-p",
                        "--pixel_loss_weights",
                        type=float,
                        default=1000.0,
                        help="Pixel-wise loss weights")
    parser.add_argument("--g_every",
                        type=int,
                        default=1,
                        help="Training generator every k iteration")
    parser.add_argument("--g_lr",
                        type=float,
                        default=1e-4,
                        help="learning rate for generator")
    parser.add_argument("--d_lr",
                        type=float,
                        default=1e-4,
                        help="learning rate for discriminator")
    parser.add_argument("-i",
                        "--checkpoint_every",
                        type=int,
                        default=200,
                        help="Save checkpoint every k iteration (checkpoints for same epoch will overwrite)")
    parser.add_argument("--d_init",
                        type=str,
                        default=None,
                        help="Init weights for discriminator")
    parser.add_argument("--g_init",
                        type=str,
                        default=None,
                        help="Init weights for generator")
    args = parser.parse_args()
    return args

# define data generator
class img_data(data.Dataset):
    def __init__(self, path):
        files = os.listdir(path)
        self.files = [os.path.join(path,x) for x in files]
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        yuv = rgb2yuv(img)
        y = yuv[...,0]-0.5
        u_t = yuv[...,1] / 0.43601035
        v_t = yuv[...,2] / 0.61497538
        return torch.Tensor(np.expand_dims(y,axis=0)),torch.Tensor(np.stack([u_t,v_t],axis=0))


args = parse_args()
if not os.path.exists(os.path.join(args.checkpoint_location,'weights')):
    os.makedirs(os.path.join(args.checkpoint_location,'weights'))

# Define G, same as torch version
G = generator().cuda(args.gpu)

# define D
D = models.resnet18(pretrained=False,num_classes=2)
D.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
D = D.cuda(args.gpu)

trainset = img_data(args.training_dir)
params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers}
training_generator = data.DataLoader(trainset, **params)
if args.test_image is not None:
    test_img = Image.open(args.test_image).convert('RGB').resize((256,256))
    test_yuv = rgb2yuv(test_img)
    test_inf = test_yuv[...,0].reshape(1,1,256,256)
    test_var = Variable(torch.Tensor(test_inf-0.5)).cuda(args.gpu)
if args.d_init is not None:
    D.load_state_dict(torch.load(args.d_init, map_location='cuda:0'))
if args.g_init is not None:
    G.load_state_dict(torch.load(args.g_init, map_location='cuda:0'))

# save test image for beginning
if args.test_image is not None:
    test_res = G(test_var)
    uv=test_res.cpu().detach().numpy()
    uv[:,0,:,:] *= 0.436
    uv[:,1,:,:] *= 0.615
    test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,256,256)
    test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
    cv2.imwrite(os.path.join(args.checkpoint_location,'test_init.jpg'),(test_rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])

# Lists to plot Ls
d_loss_list = []
g_loss_list = []

start_time = time.time()

i=0
adversarial_loss = torch.nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(0.5, 0.999))

for epoch in range(args.epoch):
    for y, uv in training_generator:
        # Adversarial ground truths
        valid = Variable(torch.Tensor(y.size(0), 1).fill_(1.0), requires_grad=False).cuda(args.gpu)
        fake = Variable(torch.Tensor(y.size(0), 1).fill_(0.0), requires_grad=False).cuda(args.gpu)

        yvar = Variable(y).cuda(args.gpu)
        uvvar = Variable(uv).cuda(args.gpu)
        real_imgs = torch.cat([yvar,uvvar],dim=1)

        optimizer_G.zero_grad()
        uvgen = G(yvar)
        # Generate a batch of images
        gen_imgs = torch.cat([yvar.detach(),uvgen],dim=1)

        # Loss measures generator's ability to fool the discriminator
        g_loss_gan = adversarial_loss(D(gen_imgs), valid)
        g_loss = g_loss_gan + args.pixel_loss_weights * torch.mean((uvvar-uvgen)**2)
        if i%args.g_every==0:
            g_loss.backward()
            optimizer_G.step()

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(D(real_imgs), valid)
        fake_loss = adversarial_loss(D(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        i+=1

        #print(f" [{i}] ",end=" ")
        d_loss_list.append(d_loss.item())
        g_loss_list.append(g_loss.item())

        if i%args.checkpoint_every==0:
            print ("[%d]Epoch: %d: [D loss: %f] [G total loss: %f] [G GAN Loss: %f]" % (i,epoch, d_loss.item(), g_loss.item(), g_loss_gan.item()))

            with open(os.path.join(args.checkpoint_location,'D_loss_'+str(epoch)), "w") as fp:
                json.dump(d_loss_list, fp)

            with open(os.path.join(args.checkpoint_location,'G_loss_'+str(epoch)), "w") as fp:
                json.dump(g_loss_list, fp)
            
            print("Time Elapsed:")
            print("--- %s seconds ---" % (time.time() - start_time))
            
            torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'weights','D'+str(epoch)+'.pth'))
            torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'weights','G'+str(epoch)+'.pth'))
            if args.test_image is not None:
                test_res = G(test_var)
                uv=test_res.cpu().detach().numpy()
                uv[:,0,:,:] *= 0.436
                uv[:,1,:,:] *= 0.615
                test_yuv = np.concatenate([test_inf,uv],axis=1).reshape(3,256,256)
                test_rgb = yuv2rgb(test_yuv.transpose(1,2,0))
                cv2.imwrite(os.path.join(args.checkpoint_location,'test_epoch_'+str(epoch)+'.jpg'),(test_rgb.clip(min=0,max=1)*256)[:,:,[2,1,0]])


plt.figure(figsize=(10, 5))
plt.plot(d_loss_list, label='Discriminator Loss')
plt.plot(g_loss_list, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('GAN Training Loss over Epochs')
plt.legend()
plt.savefig(os.path.join(args.checkpoint_location,'gan_losses_plot.png'))
# plt.show()

#Time Calc:
print("Training Completed in:")
print("--- %s seconds ---" % (time.time() - start_time))

with open(os.path.join(args.checkpoint_location,'D_loss'), "w") as fp:
    json.dump(d_loss_list, fp)

with open(os.path.join(args.checkpoint_location,'G_loss'), "w") as fp:
    json.dump(g_loss_list, fp)

#Save the Final Model
torch.save(D.state_dict(), os.path.join(args.checkpoint_location,'D_final.pth'))
torch.save(G.state_dict(), os.path.join(args.checkpoint_location,'G_final.pth'))

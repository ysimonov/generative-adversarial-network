import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import cv2
import gc
import random
torch.cuda.empty_cache()

DATA_PATH = './data/celeba'

SELECTED_DEVICE = 'cuda'
AVAILABLE_GPUS = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
NGPU = len(AVAILABLE_GPUS)

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

BATCH_SIZE = 128
NUM_EPOCHS = 2

NOISE_CHANNELS = 100
DISCRIMINATOR_FEATURES = 64
GENERATOR_FEATURES = 64

# hyperparameters
BETA1 = 0.5
BETA2 = 0.999
DISCRIMINATOR_LEARNING_RATE = 0.0002
GENERATOR_LEARNING_RATE = 0.0002


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train():

    device = torch.device('cuda:0' if torch.cuda.is_available() and NGPU > 0 and SELECTED_DEVICE != 'cpu' else 'cpu')

    dataset = ImageFolder(
        root=DATA_PATH, 
        transform=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    )
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=4)

    # plot some training images
    data_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                data_batch[0][:64], 
                padding=2, normalize=True).cpu(),
            (1, 2, 0)
        )
    )
    plt.show()
    plt.close()

    # create generator and descriminator
    net_disc = Discriminator(
        img_channels=IMAGE_CHANNELS, 
        num_features=DISCRIMINATOR_FEATURES,
        ngpu=NGPU
    ).to(device)
    if (device.type == 'cuda') and (NGPU > 1):
        net_disc = nn.DataParallel(module=net_disc, device_ids=list(range(NGPU)))
    net_disc.apply(weight_init)

    net_gen = Generator(
        noise_channels=NOISE_CHANNELS, 
        img_channels=IMAGE_CHANNELS, 
        num_features=GENERATOR_FEATURES,
        ngpu=NGPU
    ).to(device)
    if (device.type == 'cuda') and (NGPU > 1):
        net_gen = nn.DataParallel(module=net_gen, device_ids=list(range(NGPU)))
    net_gen.apply(weight_init)

    criterion = nn.BCELoss()

    # set convention for real and fake labels
    real_label = 1.0
    fake_label = 0.0

    # batch of latent vectors
    fixed_noise = torch.randn(64, NOISE_CHANNELS, 1, 1, device=device)

    # set optimizers for generator and discriminator
    optim_disc = optim.Adam(
        net_disc.parameters(), 
        lr=DISCRIMINATOR_LEARNING_RATE, 
        betas=(BETA1, BETA2))

    optim_gen = optim.Adam(
        net_gen.parameters(), 
        lr=GENERATOR_LEARNING_RATE, 
        betas=(BETA1, BETA2))


    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(NUM_EPOCHS):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            net_disc.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = net_disc(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, NOISE_CHANNELS, 1, 1, device=device)
            # Generate fake image batch with G
            fake = net_gen(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = net_disc(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optim_disc.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            net_gen.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = net_disc(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optim_gen.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]   Loss_D: %.4f   Loss_G: %.4f   D(x): %.4f   D(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = net_gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
                filename = f'./results/successful_image_{i}.jpg'
                with open(filename, mode='wb') as image_file:
                    plt.imsave(image_file, np.transpose(img_list[-1].numpy(),(1,2,0)))

            iters += 1


    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.plot()

    # print('Starting training ...')
    # for epoch in range(NUM_EPOCHS):
    #     for batch_idx, data in enumerate(dataloader):

    #         # set all gradients to zero
    #         net_gen.zero_grad()

    #         data_device = data[0].to(device)
    #         batch_size = data_device.size(0)
                    
    #         #!> Train Discriminator: maximize log(D(x)) + log(1-D(G(z)))

    #         # forward pass data batch through discriminator
    #         output = net_disc.forward(data_device).view(-1)

    #         # calculate loss between discriminator output and target (real) labels
    #         label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
    #         loss_disc_real = criterion(output, label)

    #         # calculate gradients for discriminator in backward pass
    #         loss_disc_real.backward()
    #         disc_x = output.mean().item()

    #         # generate batch of latent (noise) vectors
    #         noise = torch.randn(batch_size, NOISE_CHANNELS, 1, 1, device=device)

    #         # generate fake image batch with generator
    #         fake = net_gen.forward(noise)

    #         # classify all fake labels with discriminator
    #         output = net_disc(fake.detach()).view(-1)

    #         # calculate loss between discriminator output and target (fake) labels
    #         label.fill_(fake_label)
    #         loss_disc_fake = criterion(output, label)

    #         # calculate gradients for fake, summed with real gradients
    #         loss_disc_fake.backward()
    #         loss_disc = loss_disc_real + loss_disc_fake

    #         # update discriminator
    #         optim_disc.step()
    
    #         #!> Train Generator: maximize log(D(G(z)))
    #         net_gen.zero_grad()
    #         label.fill_(real_label)

    #         # perform forward pass of all fake image batch through discriminator
    #         output = net_disc.forward(fake).view(-1)

    #         # calculate generator's loss
    #         loss_gen = criterion(output, label)

    #         # calculate gradients for generator
    #         loss_gen.backward()

    #         # update generator
    #         optim_gen.step()

    #         # display training statistics
    #         if batch_idx % 100 == 0:
    #             print(  f'Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} '
    #                     f'Loss D: {loss_disc.item():.4f}, Loss G: {loss_gen.item():.4f} D(x): {disc_x:.4f}')
                
        # save weights and image every epoch
        # with torch.no_grad():

        #     model_path = f"./weights/celeb_dcgan_dis_{epoch}.pt"
        #     torch.save(net_disc.state_dict(), model_path)

        #     model_path = f"./weights/celeb_dcgan_gen_{epoch}.pt"
        #     torch.save(net_gen.state_dict(), model_path)

        #     fake = net_gen(fixed_noise).detach().cpu()
        #     img_fake = np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(), (1, 2, 0))

        #     filename = f'./results/{epoch}_{batch_idx}_fake.jpg'
        #     with open(filename, mode='wb') as image_file:
        #         plt.imsave(image_file, img_fake)
                

if __name__ == '__main__':
    try:
        train()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
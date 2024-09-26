'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
import os
from models import *
from utils import progress_bar
import shutil
from diffusion_lib import DiffusionMLP, GaussianDiffusion

def eu_dis(a,b,p=2):
    return torch.norm(a[:,None]-b,dim=2,p=p)

def sim_matrix( a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_rec = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        optimizer2.zero_grad()
        outputs = net(inputs)

        # minimize the distance between the noise and the estimated noise
        loss_rec = 128 * gaussian_diffusion.ddpm_forward_oversample(ddpm_model, zs_weight.t()[targets], outputs)


        # Dis part: This slightly improves the performance but gives larger computational cost
        # embed_forward = gaussian_diffusion.sample(ddpm_model, outputs, clip_denoised='clamp') # clamp
        # loss_cls = F.mse_loss(zs_weight.t()[targets], outputs)
        # loss_cls = 128 * F.mse_loss(zs_weight.t()[targets], embed_forward)

        # loss = loss_rec + loss_cls
        loss =  loss_rec 

        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0, norm_type=2.0)
        torch.nn.utils.clip_grad_norm_(ddpm_model.parameters(), max_norm=1.0, norm_type=2.0)

        optimizer.step()
        optimizer2.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 20 == 0:
            print(f'[Epoch: {epoch}] {batch_idx}/{len(trainloader)}: Loss {loss_rec:.2f}')


def test(epoch, best_acc):
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
                
            outputs = gaussian_diffusion.sample(ddpm_model, outputs, clip_denoised='none') # clamp
            outputs =  eu_dis(outputs, zs_weight.t())
            # loss = criterion(outputs, targets)
            _, predicted = outputs.min(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'ddpm_model': ddpm_model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }    
        torch.save(state, args.dir+'/ckpt.pth')
        best_acc = acc
    return best_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--dir',default='./experiments/',
                        help='resume from checkpoint')
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    log_file_path = os.path.join(args.dir, 'log.txt')
    sys.stdout = open(log_file_path, 'w')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18CLIP()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    path_dir = args.dir

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    shutil.copy('./train_net_diffusion.py', path_dir)
    shutil.copy('./diffusion_lib.py', path_dir)


    num_ddpm_steps = 10
    ddpm_model = DiffusionMLP(512, 1024, 512, 2, num_ddpm_steps).to(device)

    gaussian_diffusion = GaussianDiffusion(timesteps=num_ddpm_steps, beta_schedule ='cosine')
    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt_clip_diff_ad.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     ddpm_model.load_state_dict(checkpoint['ddpm_model'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    emb = torch.load('text.pt').float() # 10 512
    zs_weight = F.normalize(emb,dim=-1).t().to(device)
    # Training
    optimizer2 = optim.SGD(ddpm_model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        if epoch % 1 == 0 and epoch>0:
            best_acc = test(epoch, best_acc)
        scheduler.step()

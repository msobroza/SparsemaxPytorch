from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data
import random
import numpy as np
from random import randint

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

def generateRandomCliqueVector(clusters, nodes_per_cluster):
	result = np.zeros(clusters*nodes_per_cluster)
	for i in xrange(clusters):
		j = random.randint(0,nodes_per_cluster-1)
		result[i*nodes_per_cluster+j]=1.0
	return result

class Sparsemax(nn.Module):
    def __init__(self, num_clusters, num_neurons_per_cluster):
        super(Sparsemax, self).__init__()
        self.num_clusters = num_clusters
        self.num_neurons_per_cluster = num_neurons_per_cluster
        
    def forward(self, input):

        input_reshape = torch.zeros(input.size())
        input_reshape = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        #print(input_reshape)
        dim = 2
        #translate for numerical stability
        input_shift = input_reshape # - torch.max(input_reshape, dim)[0].expand_as(input_reshape)

        #sorting input in descending order
        z_sorted = torch.sort(input_shift, dim=dim, descending=True)[0]
        input_size = input_shift.size()[dim]	
        range_values = Variable(torch.arange(1, input_size+1), requires_grad=False)
        range_values = range_values.expand_as(z_sorted)

        #Determine sparsity of projection
        bound = Variable(torch.zeros(z_sorted.size()),requires_grad=False)

        #z_sorted = z_sorted.type_as(bound)
        bound = 1 + torch.addcmul(bound, range_values, z_sorted)
        cumsum_zs = torch.cumsum(z_sorted, dim)
        is_gt = torch.gt(bound, cumsum_zs).type(torch.FloatTensor)
        valid = Variable(torch.zeros(range_values.size()),requires_grad=False)	
        valid = torch.addcmul(valid, range_values, is_gt)
        k_max = torch.max(valid, dim)[0]
        zs_sparse = Variable(torch.zeros(z_sorted.size()),requires_grad=False)
        zs_sparse = torch.addcmul(zs_sparse, is_gt, z_sorted)
        sum_zs = (torch.sum(zs_sparse, dim) - 1)
        taus = Variable(torch.zeros(k_max.size()),requires_grad=False)
        taus = torch.addcdiv(taus, (torch.sum(zs_sparse, dim) - 1), k_max)
        taus_expanded = taus.expand_as(input_reshape)
        output = Variable(torch.zeros(input_reshape.size()))
        output = torch.max(output, input_shift - taus_expanded)
        #self.save_for_backward(output)
        #loss = sparseMaxLoss(taus)
        return output.view(-1, self.num_clusters*self.num_neurons_per_cluster), zs_sparse,taus, is_gt
		 

    def backward(self, grad_output):
        #output_forward, = self.saved_tensors

        self.output = self.output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        grad_output = grad_output.view(-1,self.num_clusters, self.num_neurons_per_cluster)
        dim = 2
        non_zeros = Variable(torch.ne(self.output, 0).type(torch.FloatTensor), requires_grad=False)
        mask_grad = Variable(torch.zeros(self.output.size()), requires_grad=False)
        mask_grad = torch.addcmul(mask_grad, non_zeros, grad_output)
        sum_mask_grad = torch.sum(mask_grad, dim)
        l1_norm_non_zeros = torch.sum(non_zeros, dim)
        sum_v = Variable(torch.zeros(sum_mask_grad.size()), requires_grad=False)
        sum_v = torch.addcdiv(sum_v, sum_mask_grad, l1_norm_non_zeros)
        self.gradInput = Variable(torch.zeros(grad_output.size()))
        self.gradInput = torch.addcmul(self.gradInput, non_zeros, grad_output - sum_v.expand_as(grad_output))
        self.gradInput = self.gradInput.view(-1, self.num_clusters*self.num_neurons_per_cluster)
        return self.gradInput

class MultiLabelSparseMaxLoss(nn.Module):

    def __init__(self, num_clusters, num_neurons_per_cluster):
        super(MultiLabelSparseMaxLoss, self).__init__()
        self.num_clusters = num_clusters
        self.num_neurons_per_cluster = num_neurons_per_cluster

    def forward(self, input, zs_sparse, target, output_sparsemax, taus, is_gt):
        self.output_sparsemax = output_sparsemax
        input = input.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        self.target = target.view(-1, self.num_clusters, self.num_neurons_per_cluster)
        batch_size = input.size(0)
        dim = 2
        target_times_input = torch.sum(self.target * input, dim)
        target_inner_product = torch.sum(self.target * self.target, dim)
        zs_squared = zs_sparse * zs_sparse
        taus_squared = (taus * taus).expand_as(zs_squared)
        taus_squared = taus_squared * is_gt
        sum_input_taus = torch.sum(zs_squared - taus_squared, dim) 
        sparsemax_loss = - target_times_input + 0.5*sum_input_taus + 0.5*target_inner_product
        sparsemax_loss = torch.sum(sparsemax_loss)/(batch_size * self.num_clusters)
        return sparsemax_loss

    def backward(self):
        grad_output = (- self.target + self.output_sparsemax)/(batch_size * self.num_clusters)
        return grad_output


class Net(nn.Module):
    def __init__(self, H_clusters, H_neurons_per_cluster):
        super(Net, self).__init__()
        self.H_clusters=H_clusters
        self.H_neurons_per_cluster=H_neurons_per_cluster
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50,self.H_clusters*self.H_neurons_per_cluster)

        self.sparsemaxActivation = Sparsemax(self.H_clusters,self.H_neurons_per_cluster)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        y_pred, zs_sparse, taus, is_gt = self.sparsemaxActivation(x)
        return x, y_pred, zs_sparse, taus, is_gt

H_clusters, H_neurons_per_cluster, N_class = 10, 10, 10
model = Net(H_clusters, H_neurons_per_cluster)
sparsemaxMulticlassLoss = MultiLabelSparseMaxLoss(H_clusters, H_neurons_per_cluster)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
code_target_class = np.zeros((N_class,H_clusters*H_neurons_per_cluster), dtype='float32')

for i in xrange(N_class):
	code_target_class[i] = generateRandomCliqueVector(H_clusters,H_neurons_per_cluster).reshape((H_clusters*H_neurons_per_cluster))


table_embedding = nn.Embedding(N_class, H_clusters*H_neurons_per_cluster, sparse=True).cuda()
table_embedding.volatile=True
table_embedding.requires_grad=False
table_embedding.weight = nn.Parameter(torch.from_numpy(code_target_class).cuda())
table_embedding.weight.requires_grad=False
table_embedding.weight.cuda()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        code_target = table_embedding(target)
        optimizer.zero_grad()
        input_sparsemax, y_pred, zs_sparse, taus, is_gt = model(data)
        loss = sparsemaxMulticlassLoss(input_sparsemax, zs_sparse, code_target, y_pred, taus, is_gt)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
#test()

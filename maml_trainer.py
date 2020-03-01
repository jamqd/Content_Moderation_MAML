import numpy as np
import torch
import torchvision

import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MAMLTrainer():

    def __init__(self, model, data, training_options, logging_options, baseline=False):
        '''
        training_options = {
            pretrain_iterations: number of pre-training iterations,
            metatrain_iterations: number of metatraining iterations,

            meta_batch_size: number of tasks sampled per meta-update,
            meta_lr: the base learning rate of the generator,

            update_batch_size: number of examples used for inner gradient update (K for K-shot learning),
            update_lr: step size alpha for inner gradient update,

            num_updates: number of inner gradient updates during training
        }

        logging_options = {
            log: if false, do not log summaries, for debugging code
            logdir: /tmp/data, directory for summaries and checkpoints
            save_interval: number of iterations to save after
        }
        '''

        self.model = model
        self.data = data

        self.pretrain_iterations = training_options['pretrain_iterations']
        self.metatrain_iterations = training_options['metatrain_iterations']

        self.logdir = logging_options['logdir']
        self.log = logging_options['log']

    def generate_batch(self):
        pass

    def train(self):
        # if self.log:
        #     train_writer = tf.summary.create_file_writer(self.logdir + '/' + datetime.datetime.now().strftime('%H:%M:%S'))

        pass

class MyDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

num_iterations = 100

mnist = torchvision.datasets.CIFAR10(root="/tmp/cifar", train=True, download=True)

mnist = l2l.data.MetaDataset(mnist)
train_tasks = l2l.data.TaskDataset(mnist,
                                   task_transforms=[
                                        NWays(mnist, n=3),
                                        KShots(mnist, k=1),
                                        LoadData(mnist),
                                   ],
                                   num_tasks=10)
model = Net()
maml = l2l.algorithms.MAML(model, lr=1e-3, first_order=False)
opt = optim.Adam(maml.parameters(), lr=4e-3)

for iteration in range(num_iterations):
    learner = maml.clone()  # Creates a clone of model
    for task in train_tasks:
        # Split task in adaptation_task and evalutation_task
        # Fast adapt
        for step in range(adaptation_steps):
            error = compute_loss(adaptation_task)
            learner.adapt(error)

        # Compute evaluation loss
        evaluation_error = compute_loss(evaluation_task)
        print(evaluation_error)

        # Meta-update the model parameters
        opt.zero_grad()
        evaluation_error.backward()
        opt.step()

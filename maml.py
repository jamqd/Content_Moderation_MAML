# sudo CFLAGS=-stdlib=libc++ python3 maml.py

import argparse
import random
import pandas as pd

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l

import torchtext
from torchtext.datasets import text_classification

from torch.utils.data import Dataset

class Net(nn.Module):
    def __init__(self, roberta, finetune=False):
        super(Net, self).__init__()
        self.roberta = roberta.model
        if not finetune:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, tokens_list):
        sentence_embeddings = []
        for tokens in tokens_list:
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if tokens.size(-1) > self.roberta.max_positions():
                raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                    tokens.size(-1), self.roberta.max_positions()
                ))
            x, extra = self.roberta(
                tokens,
                features_only=True,
                return_all_hiddens=True,
            )
            inner_states = extra['inner_states']
            pooling_layer = inner_states[-2].transpose(0, 1)
            sentence_embeddings.append(pooling_layer.mean(1).view(1024))
        
        x = torch.stack(sentence_embeddings)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


class MAMLDataset(Dataset):
   
    def __init__(self, dataset_name):
        self.dataset = None
        self.dataset_name = dataset_name
        if dataset_name == 'SST':
            self.dataset = torchtext.datasets.SST("./269_datasets/SST/train.txt", torchtext.data.Field(sequential=False), torchtext.data.Field(sequential=False))
        elif dataset_name == 'toxic_comment':
            self.dataset = pd.read_csv("./269_datasets/jigsaw-toxic-comment-classification-challenge/train.csv")
        elif dataset_name == '4054689':
            comments = pd.read_csv("./269_datasets/4054689/attack_annotated_comments.tsv", sep='\t')
            annotations = pd.read_csv("./269_datasets/4054689/attack_annotations.tsv", sep='\t')
            self.dataset = comments.merge(annotations, how='inner', on='rev_id')
        elif dataset_name == 'detecting-insults-in-social-commentary':
            self.dataset = pd.read_csv("./269_datasets/detecting-insults-in-social-commentary/train.csv")
        elif dataset_name == 'GermEval-2018-Data-master':
            self.dataset = None
        elif dataset_name == 'hate-speech-and-offensive-language':
            self.dataset = pd.read_csv("./269_datasets/hate-speech-and-offensive-language/labeled_data.csv")
        elif dataset_name == 'hate-speech-dataset-master':
            annotations_metadata = pd.read_csv("./269_datasets/hate-speech-dataset-master/annotations_metadata.csv")
            ids = []
            comments = []
            for index, row in annotations_metadata.iterrows():
                with open("./269_datasets/hate-speech-dataset-master/all_files/" + row['file_id'] + ".txt") as f:
                    ids.append(row['file_id'])
                    comments.append(f.read().strip())
            comments_data = pd.DataFrame.from_dict({'file_id' : ids, 'comment' : comments})
            self.dataset = annotations_metadata.merge(comments_data, how='inner', on='file_id')
        elif dataset_name == 'IWG_hatespeech_public-master':
            self.dataset = pd.read_csv('./269_datasets/IWG_hatespeech_public-master/german hatespeech refugees.csv')
        elif dataset_name == 'quora-insincere-questions-classification':
            self.dataset = pd.read_csv('./269_datasets/quora-insincere-questions-classification/train.csv')
        elif dataset_name == 'twitter-sentiment-analysis-hatred-speech':
            self.dataset = pd.read_csv('./269_datasets/twitter-sentiment-analysis-hatred-speech/train.csv')

        if dataset_name != 'SST':
            print(self.dataset.iloc[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_name == 'SST':
            tokens = getattr(self.dataset[idx], 'text')
            labels = int(getattr(self.dataset[idx], 'label') == 'negative')
        elif self.dataset_name == 'toxic_comment':
            tokens = self.dataset.iloc[idx]['comment_text']
            labels = self.dataset.iloc[idx]['identity_hate']
        elif self.dataset_name == '4054689':
            tokens = self.dataset.iloc[idx]['comment']
            labels = self.dataset.iloc[idx]['attack']
        elif self.dataset_name == 'detecting-insults-in-social-commentary':
            tokens = self.dataset.iloc[idx]['Comment']
            labels = self.dataset.iloc[idx]['Insult']
        elif self.dataset_name == 'GermEval-2018-Data-master':
            tokens = None
            labels = None
        elif self.dataset_name == 'hate-speech-and-offensive-language':
            tokens = self.dataset.iloc[idx]['tweet']
            labels = int(self.dataset.iloc[idx]['class'] == 0)
        elif self.dataset_name == 'hate-speech-dataset-master':
            tokens = self.dataset.iloc[idx]['comment']
            labels = int(self.dataset.iloc[idx]['label'] == 'hate')
        elif self.dataset_name == 'IWG_hatespeech_public-master':
            tokens = self.dataset.iloc[idx]['Tweet']
            labels =int(self.dataset.iloc[idx]['HatespeechOrNot (Expert 1)'] == 'YES')
        elif self.dataset_name == 'quora-insincere-questions-classification':
            tokens = self.dataset.iloc[idx]['question_text']
            labels = self.dataset.iloc[idx]['target']
        elif self.dataset_name == 'twitter-sentiment-analysis-hatred-speech':
            tokens = self.dataset.iloc[idx]['tweet']
            labels = self.dataset.iloc[idx]['label']

        return (tokens, labels)

def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=2, shots=5, tps=32, fas=5, device=torch.device("cpu"),
         download_location='~/data'):

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    datasets = ['SST', 'toxic_comment', '4054689', 'detecting-insults-in-social-commentary', \
                'hate-speech-and-offensive-language', 'hate-speech-dataset-master', \
                'quora-insincere-questions-classification', 'twitter-sentiment-analysis-hatred-speech']
    iterations = len(datasets)
    # for iteration in range(iterations):
    #     train = l2l.data.MetaDataset(MAMLDataset(datasets[iteration]))

    #     train_tasks = l2l.data.TaskDataset(train,
    #                                        task_transforms=[
    #                                             l2l.data.transforms.NWays(train, ways),
    #                                             l2l.data.transforms.KShots(train, 2 * shots),
    #                                             l2l.data.transforms.LoadData(train),
    #                                             l2l.data.transforms.RemapLabels(train),
    #                                             l2l.data.transforms.ConsecutiveLabels(train),
    #                                        ],
    #                                        num_tasks=50)
    #     # Sanity check
    #     train_tasks.sample()

    model = Net(roberta)
    # model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0

        print('\n\n### Dataset: ' + datasets[iteration] + '###\n\n')

        train = l2l.data.MetaDataset(MAMLDataset(datasets[iteration]))

        train_tasks = l2l.data.TaskDataset(train,
                                           task_transforms=[
                                                l2l.data.transforms.NWays(train, ways),
                                                l2l.data.transforms.KShots(train, 2 * shots),
                                                l2l.data.transforms.LoadData(train),
                                                l2l.data.transforms.RemapLabels(train),
                                                l2l.data.transforms.ConsecutiveLabels(train),
                                           ],
                                           num_tasks=50)

        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            # data = data.to(device)
            # labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(len(data), dtype=bool)
            adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = ~adaptation_indices
            adaptation_indices = adaptation_indices
            data = np.array(data)
            labels = np.array(labels)

            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            adaptation_data = [roberta.encode(elem)[:roberta.model.max_positions()] for elem in adaptation_data]
            adaptation_labels = torch.LongTensor(adaptation_labels)

            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
            evaluation_data = [roberta.encode(elem)[:roberta.model.max_positions()] for elem in evaluation_data]
            evaluation_labels = torch.LongTensor(evaluation_labels)

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error, allow_unused=True, allow_nograd=True)

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            iteration_error += valid_error
            iteration_acc += valid_accuracy

        iteration_error /= tps
        iteration_acc /= tps
        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn SST Example')

    parser.add_argument('--ways', type=int, default=2, metavar='N',
                        help='number of ways (default: 2)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 5)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=32, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(lr=args.lr,
         maml_lr=args.maml_lr,
         iterations=args.iterations,
         ways=args.ways,
         shots=args.shots,
         tps=args.tasks_per_step,
         fas=args.fast_adaption_steps,
         device=device,
         download_location=args.download_location)

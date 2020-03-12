# sudo CFLAGS=-stdlib=libc++ python3 maml.py

import argparse
import random
import pandas as pd
import pickle

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import learn2learn as l2l

import torchtext
from torchtext.datasets import text_classification

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class Net(nn.Module):
    def __init__(self, roberta, finetune=False):
        super(Net, self).__init__()
        self.roberta = roberta.model
        if not finetune:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
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

        x = F.relu(self.bn1(self.fc1(x)))
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

def maml(lr=0.005, maml_lr=0.01, iterations=5, ways=2, shots=5, tps=5, fas=5, device=torch.device("cpu")):
    
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    datasets = ['SST', 'toxic_comment', '4054689', 'detecting-insults-in-social-commentary', \
                'hate-speech-and-offensive-language', 'hate-speech-dataset-master', \
                'quora-insincere-questions-classification']
    # datasets = ['SST', 'twitter-sentiment-analysis-hatred-speech']
    # tps = 2

    train_tasks_collection = []
    for idx in range(len(datasets)):
        print('\n\n### Dataset: ' + datasets[idx] + '###\n\n')
        train = l2l.data.MetaDataset(MAMLDataset(datasets[idx]))

        train_tasks = l2l.data.TaskDataset(train,
                                           task_transforms=[
                                                l2l.data.transforms.NWays(train, ways),
                                                l2l.data.transforms.KShots(train, 2 * shots),
                                                l2l.data.transforms.LoadData(train),
                                                l2l.data.transforms.RemapLabels(train),
                                                l2l.data.transforms.ConsecutiveLabels(train),
                                           ],
                                           num_tasks=50)
        train_tasks_collection.append(train_tasks)

    model = Net(roberta)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)

    opt = optim.Adam(meta_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=0, last_epoch=-1)

    loss_func = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0

        train_tasks_sampled = random.sample(train_tasks_collection, tps)

        for tps_i in range(tps):
            iteration_errors = torch.zeros(tps * fas)
            error_weights = torch.rand(tps * fas, requires_grad=True)

            learner = meta_model.clone()
            learner.to(device)
            train_task = train_tasks_sampled[tps_i].sample()
            data, labels = train_task

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(len(data), dtype=bool)
            adaptation_indices[np.arange(shots*ways) * 2] = True
            evaluation_indices = ~adaptation_indices
            adaptation_indices = adaptation_indices
            data = np.array(data)
            labels = np.array(labels)

            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            torch_adaptation_data = torch.zeros((len(adaptation_data), roberta.model.max_positions()), dtype=torch.long)
            for i, elem in enumerate(adaptation_data):
                encoding = roberta.encode(elem)[:roberta.model.max_positions()]
                torch_adaptation_data[i, :len(encoding)] = encoding
            adaptation_data = torch_adaptation_data.to(device)
            torch_adaptation_labels = torch.LongTensor(adaptation_labels)
            adaptation_labels = torch_adaptation_labels.to(device)

            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
            torch_evaluation_data = torch.zeros((len(evaluation_data), roberta.model.max_positions()), dtype=torch.long)
            for i, elem in enumerate(evaluation_data):
                encoding = roberta.encode(elem)[:roberta.model.max_positions()]
                torch_evaluation_data[i, :len(encoding)] = encoding
            evaluation_data = torch_evaluation_data.to(device)
            torch_evaluation_labels = torch.LongTensor(evaluation_labels)
            evaluation_labels = torch_evaluation_labels.to(device)

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error, allow_unused=True, allow_nograd=True)

                # Compute validation loss
                # MAML MSL
                predictions = learner(evaluation_data)
                valid_error = loss_func(predictions, evaluation_labels)
                valid_error /= len(evaluation_data)
                valid_accuracy = accuracy(predictions, evaluation_labels)
                iteration_errors[fas * tps_i + step] = valid_error
                iteration_acc += valid_accuracy

            iteration_error += torch.dot(error_weights, iteration_errors)

            del adaptation_data
            del adaptation_labels
            del evaluation_data
            del evaluation_labels
            del learner

        iteration_error /= tps
        iteration_acc /= (tps * fas)
        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()
        scheduler.step()

        error_weights.data = error_weights.data - lr * error_weights.grad.data


    torch.save(model.state_dict(), './models/maml.pt')

def pretrain(lr=0.005, iterations=5, shots=5, fas=5, device=torch.device("cpu")):

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    pretrain_data = MAMLDataset('hate-speech-dataset-master')
    pretrain_dataset = DataLoader(pretrain_data, batch_size=2*2*shots, shuffle=True)
    iter_pretrain_dataset = iter(pretrain_dataset)

    model = Net(roberta)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0

        data, labels = None, None
        try:
            data, labels = next(iter_pretrain_dataset)
        except:
            pretrain_dataset = DataLoader(pretrain_data, batch_size=2*2*shots, shuffle=True)
            iter_pretrain_dataset = iter(pretrain_dataset)
            data, labels = next(iter_pretrain_dataset)

        torch_data = torch.zeros((len(data), roberta.model.max_positions()), dtype=torch.long)
        for i, elem in enumerate(data):
            encoding = roberta.encode(elem)[:roberta.model.max_positions()]
            torch_data[i, :len(encoding)] = encoding
        torch_labels = torch.LongTensor(labels)
        data = torch_data.to(device)
        labels = torch_labels.to(device)

        for step in range(fas):
            predictions = model(data)
            train_error = loss_func(predictions, labels)
            train_acc = accuracy(predictions, labels)

            opt.zero_grad()
            train_error.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.data = param.data - lr * param.grad.data
            iteration_error += train_error / len(data)
            iteration_acc += train_acc

        iteration_error /= fas
        iteration_acc /= fas
        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        del data
        del labels

    torch.save(model.state_dict(), './models/pretrain.pt')
    del model

def train(lr=0.005, iterations=5, shots=5, device=torch.device("cpu"), filepath=None):

    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    data = MAMLDataset('twitter-sentiment-analysis-hatred-speech')

    # 90-10 train-test split
    train_size = 9 * len(data) // 10
    test_size = len(data) - train_size
    train_data_split, test_data_split = torch.utils.data.random_split(data, [train_size, test_size])

    train_dataset = DataLoader(train_data_split, batch_size=2*2*shots, shuffle=True)
    # train_dataset = DataLoader(train_data_split, batch_size=128, shuffle=True)
    iter_train_dataset = iter(train_dataset)

    test_dataset = DataLoader(test_data_split, batch_size=len(test_data_split))

    model = Net(roberta)
    if filepath is not None:
        model.load_state_dict(torch.load(filepath))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

    train_accs = []
    test_accs = []

    train_losses = []
    test_losses = []

    for iteration in range(iterations):

        train_data, train_labels = None, None
        try:
            train_data, train_labels = next(iter(train_dataset))
        except:
            train_dataset = DataLoader(train_data_split, batch_size=2*2*shots, shuffle=True)
            # train_dataset = DataLoader(train_data_split, batch_size=128, shuffle=True)
            iter_train_dataset = iter(train_dataset)
            train_data, train_labels = next(iter(train_dataset))

        torch_train_data = torch.zeros((len(train_data), roberta.model.max_positions()), dtype=torch.long)
        for i, elem in enumerate(train_data):
            encoding = roberta.encode(elem)[:roberta.model.max_positions()]
            torch_train_data[i, :len(encoding)] = encoding
        torch_train_labels = torch.LongTensor(train_labels)
        train_data = torch_train_data.to(device)
        train_labels = torch_train_labels.to(device)

        train_predictions = model(train_data)
        train_error = loss_func(train_predictions, train_labels)
        train_acc = accuracy(train_predictions, train_labels)

        opt.zero_grad()
        train_error.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data
        # train_error /= len(train_data)
        print('Train Loss : {:.3f} Train Acc : {:.3f}'.format(train_error.item(), train_acc))

        train_losses.append(train_error)
        train_accs.append(train_acc)

        del train_data
        del train_labels

    test_data, test_labels = next(iter(test_dataset))
    torch_test_data = torch.zeros((len(test_data), roberta.model.max_positions()), dtype=torch.long)
    for i, elem in enumerate(test_data):
        encoding = roberta.encode(elem)[:roberta.model.max_positions()]
        torch_test_data[i, :len(encoding)] = encoding
    torch_test_labels = torch.LongTensor(test_labels)
    test_data = torch_test_data.to(device)
    test_labels = torch_test_labels.to(device)

    test_predictions = model(test_data)
    test_error = loss_func(test_predictions, test_labels)
    test_acc = accuracy(test_predictions, test_labels)

    test_losses.append(test_error)
    test_accs.append(test_acc)

    # test_error /= len(test_data)
    print('Test Loss : {:.3f} Test Acc : {:.3f}'.format(test_error.item(), test_acc))

    del test_data
    del test_labels

    suffix = ''
    if filepath == './models/maml.pt':
        suffix = 'maml'
    elif filepath == './models/pretrain.pt':
        suffix = 'pretrain'
    with open('./models/train_losses_' + suffix + '.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('./models/train_accs_' + suffix + '.pkl', 'wb') as f:
        pickle.dump(train_accs, f)
    with open('./models/test_losses_' + suffix + '.pkl', 'wb') as f:
        pickle.dump(test_losses, f)
    with open('./models/test_accs_' + suffix + '.pkl', 'wb') as f:
        pickle.dump(test_accs, f)
    
    del model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn SST Example')

    parser.add_argument('--ways', type=int, default=2, metavar='N',
                        help='number of ways (default: 2)')
    parser.add_argument('--shots', type=int, default=5, metavar='N',
                        help='number of shots (default: 5)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=5, metavar='N',
                        help='tasks per step (default: 5)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=5, metavar='N',
                        help='number of iterations (default: 5)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

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

    # print("Training MAML")
    # maml(lr=args.lr,
    #      maml_lr=args.maml_lr,
    #      iterations=args.iterations,
    #      ways=args.ways,
    #      shots=args.shots,
    #      tps=args.tasks_per_step,
    #      fas=args.fast_adaption_steps,
    #      device=device)

    # print("Training pretrain")
    # pretrain(lr=args.lr,
    #      iterations=args.iterations,
    #      shots=args.shots,
    #      fas=args.fast_adaption_steps,
    #      device=device)

    print("Training from MAML")
    train(lr=args.lr,
         iterations=args.iterations,
         shots=args.shots,
         device=device,
         filepath='./models/maml.pt')

    print("Training from pretrain")
    train(lr=args.lr,
         iterations=args.iterations,
         shots=args.shots,
         device=device,
         filepath='./models/pretrain.pt')

    print("Training from scratch")
    train(lr=args.lr,
         iterations=args.iterations,
         shots=args.shots,
         device=device,
         filepath=None)

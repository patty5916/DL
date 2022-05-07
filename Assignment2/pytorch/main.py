import argparse
import copy
import timeit
from memory_profiler import profile

from utils import *
from model import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='For DL Assignment2')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--train_batch', type=int, default=1024, help='batch of training set')
    parser.add_argument('--val_batch', type=int, default=128, help='batch of validation set')

    return parser.parse_args()



@profile(stream=open('pytorch_LeNet5.log', 'w+'))
def main():
    args = parse_args()
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_data = ImageDataset('train', transform)
    val_data = ImageDataset('val', transform)

    train_loader = DataLoader(train_data, args.train_batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, 1, shuffle=False, num_workers=0)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = LeNet5()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)

    total_time = 0
    # train
    train_accList= []
    val_accList = []
    lossList = []
    for epoch in range(args.epochs):
        start = timeit.default_timer()
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0
        best_acc = 0
        best_model_parameters = copy.deepcopy(model.state_dict())

        model.train()
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_acc += torch.sum(torch.argmax(output, axis=1) == label).cpu().numpy()
        stop = timeit.default_timer()

        # validation
        model.eval()
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            val_loss += loss.item() * data.size(0)
            val_acc += torch.sum(torch.argmax(output, axis=1) == label).cpu().numpy()

        train_loss /= len(train_loader.sampler)
        lossList.append(train_loss)

        train_acc /= len(train_loader.sampler)
        val_acc /= len(val_loader.sampler)
        train_accList.append(train_acc)
        val_accList.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_parameters = copy.deepcopy(model.state_dict())

        total_time += (stop - start)
        print(f'Epochs: {epoch}\t|Loss: {train_loss}\t|Train Acc: {train_acc}\t|Val Acc: {val_acc}\t|Time: {stop - start}')

    print(f'Best validation accuracy: {best_acc}')
    model.load_state_dict(best_model_parameters)
    torch.save(model.state_dict(), 'pytorch_LeNet5.pt')

    # plot
    plot_acc(train_accList, val_accList, list(range(args.epochs)), 'pytorch_LeNet5_acc.png')
    plot_loss(lossList, 'pytorch_LeNet5_loss.png')

    # test
    test_data = ImageDataset('test', transform)
    test_loader = DataLoader(test_data, 1, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load('pytorch_LeNet5.pt'))
    model.to(device)

    model.eval()
    test_acc = 0
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        test_acc += torch.sum(torch.argmax(output, axis=1) == label).cpu().numpy()

    test_acc /= len(test_loader.sampler)
    print(f'Test accuracy: {test_acc}')
    print(f'Total Time: {total_time}')


if __name__ == '__main__':
    main()
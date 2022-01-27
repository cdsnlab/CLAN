import argparse
from sklearn.compose import TransformedTargetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR
import random
import os
import numpy as np
from data.dataloader import splitting_data
#from model.transformer import TransformerEncoder 
from model.lossfunction import ConTimeLoss, SupConLoss
from tqdm.notebook import tqdm

#from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def training(model, train_data, args):
    print("Start Training----------------")
    # train mode
    model.train()   
    
    #optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(model, 0.1)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # loss function
    criterion = nn.CrossEntropyLoss()
    #criterion = ConTimeLoss()
   
    # scheduler
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

def testing():
    print("Start Testing----------------")
    # train mode
    model.eval()  
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




if __name__ == "__main__":    
    parser = argparse.ArgumentParser(prog='ConDaT', description='Contrastive learning with duration-aware Transformer for novelty detection in time series sensor data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for scehme
    parser.add_argument('--dataset', type=str, default='lapras', help='choose one of them: lapras, casas, aras_a, aras_b, opportunity')
    parser.add_argument('--padding', type=str, default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, default=1000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=10, help='choose of the minimum number of samples in each label')

    parser.add_argument('--test_ratio', type=float, default=0.1, help='choose the number of test ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='choose the number of vlaidation ratio')
    parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
    parser.add_argument('--encoder', type=str, default='transformer', help='choose one of them: simple, transformer')

    # for training   
    parser.add_argument('--loss', type=str, default='ConDaT', help='choose one of them: simple, transformer')
    parser.add_argument('--optimizer', type=str, default='', help='choose one of them: simple, transformer')
    parser.add_argument('--epochs', type=int, default=20, help='choose the number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='choose the number of batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='choose the number of learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='choose the number of gamma')
    parser.add_argument('--seed', type=int, default=42, help='choose the number of seed')

    args = parser.parse_args()
    #parser.print_help()

    #parser.add_argument('integers', metavar='N', type=int, nargs='+',                        help='an integer for the accumulator')
    #parser.add_argument('--sum', dest='accumulate', action='store_const',                        const=sum, default=max,                        help='sum the integers (default: find the max)')


    # check gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    
    #print(args.integers)
    #print(args.accumulate(args.integers))   

    seed_everything(args.seed)
    # Dataset extraction   
    train_list, valid_list, test_list, train_label_list, valid_label_list, test_label_list = splitting_data(args.dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, args.timespan, args.min_seq, args.min_samples)

    # model
    # if args.encoder == 'simple':
    #     model = TransformerEncoder(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    # elif args.encoder == 'transformer':
    #     model = TransformerEncoder(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)


    # training(model, train_data, args)
    


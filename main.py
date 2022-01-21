import argparse
from sklearn.compose import TransformedTargetRegressor
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataloader import laprasLoader, casasLoader, arasLoader, opportunityLoader
from model.transformer import TransformerEncoder 
from model.lossfunction import ConTimeLoss, SupConLoss

#from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def training(model, train_data, args):
    print("Start Training----------------")
    # train mode
    model.train()   
    
    if args.optimizer == 'SGD':
        optimizer = SGD(model, 0.1)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    criterion = nn.CrossEntropyLoss()
    #criterion = ConTimeLoss()
   
    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    
    for epoch in range(args.epochs):
        for input, labels in dataset:
            
            # Make gradient as 0
            optimizer.zero_grad()

            # Forward, Backward and Optimization
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        scheduler1.step()
        scheduler2.step()

    # Save learned model
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

def testing():
    print("Start Testing----------------")
    # train mode
    model.eval()  
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(prog='ConDaT', description='Contrastive learning with duration-aware Transformer for novelty detection in time series sensor data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # for scehme
    parser.add_argument('--dataset', type=str, default='lapras', help='choose one of them: lapras, casas, aras, opportunity')
    parser.add_argument('--overlapped ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
    parser.add_argument('--encoder', type=str, default='transformer', help='choose one of them: simple, transformer')

    # for training
    parser.add_argument('--loss', type=str, default='ConDaT', help='choose one of them: simple, transformer')
    parser.add_argument('--optimizer', type=str, default='', help='choose one of them: simple, transformer')
    parser.add_argument('--epochs', type=int, default=100, help='choose the number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='choose the number of batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='choose the number of learning rate')


    args = parser.parse_args()
    #parser.print_help()

    #parser.add_argument('integers', metavar='N', type=int, nargs='+',                        help='an integer for the accumulator')
    #parser.add_argument('--sum', dest='accumulate', action='store_const',                        const=sum, default=max,                        help='sum the integers (default: find the max)')


    # check gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    #print(args.integers)
    #print(args.accumulate(args.integers))   

    # Dataset
    if args.dataset == 'lapras':
        train_data, test_data = laprasLoader()
    elif args.dataset == 'casas':
         train_data, test_data = casasLoader()
    elif args.dataset == 'aras':
         train_data, test_data = arasLoader()
    elif args.dataset == 'opportunity':
         train_data, test_data = opportunityLoader()

    # model
    if args.encoder == 'simple':
        model = TransformerEncoder(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    elif args.encoder == 'transformer':
        model = TransformerEncoder(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)


    training(model, train_data, args)
    


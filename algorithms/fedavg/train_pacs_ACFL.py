import os
import argparse
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../..')
from network.get_network_ACFL import GetNetwork
from network.ResNet_ACFL import MultiLayerMLPDiscriminator
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from data.officehome_dataset import OfficeHome_FedDG
from data.terra_incognita_dataset import TerraInc_FedDG
from data.Representation import Representation
from utils.classification_metric import Classification 
from utils.log_utils import *
from utils.fed_merge import Cal_Weight_Dict, FedAvg, FedUpdate
from utils.trainval_func_ACFL import site_evaluation, site_train, GetFedModel, SaveCheckPoint, update_global_discriminator
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs', 'OfficeHome', 'terrainc'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=200)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)

    # new
    parser.add_argument('--dis_learning_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('--dis_momentum', help='momentum', type=float, default=0.9)
    parser.add_argument('--dis_weight_decay', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--r_bsz', help='batch size for representation', type=int, default=128)
    parser.add_argument('--temp', help='temp for discriminator', type=int, default=0.07)
    parser.add_argument('--rk_iters', help='rk_iters for training discriminator', type=int, default=500)
    parser.add_argument('--print_freq', help='print_freq for training discriminator', type=int, default=100)


    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='ACFL')
    parser.add_argument('--display', help='display in controller', action='store_true')
    return parser.parse_args()

from sklearn.mixture import GaussianMixture
import torch
def _fit_GMM(representation):
    representation = torch.tensor(representation)
    numpy_representation = representation.cpu().numpy()
    gmm = GaussianMixture(n_components=3)
    gmm.fit(numpy_representation)
    r, _ = gmm.sample(300)
    return r

def main():
    '''log part'''
    # Set up the log folder and tensorboard
    file_name = 'fedavg_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    if args.dataset == 'pacs':
        dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    elif args.dataset == 'OfficeHome':
        dataobj = OfficeHome_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    elif args.dataset == 'terrainc':
        dataobj = TerraInc_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)

    dataloader_dict, dataset_dict = dataobj.GetData()
    
    '''model'''
    metric = Classification()
    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)

    # new gobal discriminator
    input_dim = 2048
    output_dim = 128
    hidden_dim = [512, 256, 128]
    discriminator = MultiLayerMLPDiscriminator(input_dim, output_dim, hidden_dim)
    
    pro_r = None  # Global feature prototype

    weight_dict = Cal_Weight_Dict(dataset_dict, site_list=dataobj.train_domain_list)
    best_val = 0.

    client_name_idx = {client_name: idx for idx, client_name in enumerate(dataobj.train_domain_list)}

    for i in range(args.comm+1):
        # Initialize the local dataset of the client
        r_locals = []  # Local virtual feature set
        l_locals = [] # Local virtual feature set label

        for domain_name in dataobj.train_domain_list:
            r = site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name],dataloader_dict[domain_name]['train'], log_ten, metric, pro_r, client_name_idx, discriminator)
            # Local training is conducted, and verification is carried out immediately after training
            site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='val')

            re = _fit_GMM(r)
            r_locals.append(re)
            y = [client_name_idx[domain_name] for i in range(len(re))]
            l_locals.append(y)

        # Update the global discriminator and the global feature prototype
        r_locals = np.concatenate(r_locals, axis=0)
        l_locals = np.concatenate(l_locals, axis=0)
        r_dataset = Representation(r_locals, l_locals)
        r_dataloader = DataLoader(r_dataset, batch_size=args.r_bsz, shuffle=True)

        pro_r = update_global_discriminator(args, discriminator, r_dataloader)

        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            results_dict = site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='test')
            fed_val+= results_dict['acc']*weight_dict[domain_name]
        # val result
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            for domain_name in dataobj.train_domain_list: 
                SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'best_val_{domain_name}_model')
                
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    for domain_name in dataobj.train_domain_list: 
        SaveCheckPoint(args, model_dict[domain_name], args.comm, os.path.join(log_dir, 'checkpoints'), note=f'last_{domain_name}_model')
    
if __name__ == '__main__':
    main()
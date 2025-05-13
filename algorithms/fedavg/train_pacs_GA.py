import os
import argparse
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../..')
from utils.log_utils import *
from torch.utils.tensorboard.writer import SummaryWriter
from data.pacs_dataset import PACS_FedDG
from utils.classification_metric import Classification 
from utils.fed_merge import FedAvg, FedUpdate
from utils.trainval_func import site_train, site_evaluation, GetFedModel, SaveCheckPoint
from utils.weight_adjust import refine_weight_dict_by_GA
def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='pacs', choices=['pacs'], help='Name of dataset')
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'], help='model name')
    parser.add_argument("--test_domain", type=str, default='p',
                        choices=['p', 'a', 'c', 's'], help='the domain name for testing')
    parser.add_argument('--num_classes', help='number of classes default 7', type=int, default=7)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--local_epochs', help='epochs number', type=int, default=5)
    parser.add_argument('--comm', help='epochs number', type=int, default=40)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--step_size', help='rate weight step', type=float, default=0.2)
    parser.add_argument("--lr_policy", type=str, default='step', choices=['step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--fair", type=str, default='acc', choices=['acc', 'loss'],
                        help="the fairness metric for FedAvg")
    parser.add_argument('--note', help='note of experimental settings', type=str, default='generalization_adjustment')
    parser.add_argument('--display', help='display in controller', action='store_true') 

    return parser.parse_args()


def main():
    file_name = 'GA_'+os.path.split(__file__)[1].replace('.py', '')
    args = get_argparse()
    log_dir, tensorboard_dir = Gen_Log_Dir(args, file_name=file_name)
    log_ten = SummaryWriter(log_dir=tensorboard_dir)
    log_file = Get_Logger(file_name=log_dir + 'train.log', display=args.display)
    Save_Hyperparameter(log_dir, args)
    
    '''dataset and dataloader'''
    dataobj = PACS_FedDG(test_domain=args.test_domain, batch_size=args.batch_size)
    dataloader_dict, dataset_dict = dataobj.GetData()
    
    '''model'''
    metric = Classification()

    global_model, model_dict, optimizer_dict, scheduler_dict = GetFedModel(args, args.num_classes)
    weight_dict = {}
    site_results_before_avg = {}
    site_results_after_avg = {}

    for site_name in dataobj.train_domain_list:
        weight_dict[site_name] = 1./3.
        site_results_before_avg[site_name] = None
        site_results_after_avg[site_name] = None
        

    FedUpdate(model_dict, global_model)
    best_val=0.
    step_size_decay = args.step_size / args.comm
    for i in range(args.comm+1):
        for domain_name in dataobj.train_domain_list:
            site_train(i, domain_name, args, model_dict[domain_name], optimizer_dict[domain_name], 
                       scheduler_dict[domain_name],dataloader_dict[domain_name]['train'], log_ten, metric)
            
            site_results_before_avg[domain_name] = site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric, note='before_fed')
        
        FedAvg(model_dict, weight_dict, global_model)
        FedUpdate(model_dict, global_model)
        
        fed_val = 0.
        for domain_name in dataobj.train_domain_list:
            site_results_after_avg[domain_name] = site_evaluation(i, domain_name, args, model_dict[domain_name], dataloader_dict[domain_name]['val'], log_file, log_ten, metric)
            fed_val+= site_results_after_avg[domain_name]['acc']*weight_dict[domain_name]
            
        if fed_val >= best_val:
            best_val = fed_val
            SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='best_val_model')
            log_file.info(f'Model saved! Best Val Acc: {best_val*100:.2f}%')
        site_evaluation(i, args.test_domain, args, model_dict[args.test_domain], dataloader_dict[args.test_domain]['test'], log_file, log_ten, metric, note='test_domain')
        
        weight_dict = refine_weight_dict_by_GA(weight_dict, site_results_before_avg, site_results_after_avg, args.step_size - (i-1)*step_size_decay, fair_metric=args.fair)
        log_str = f'Round {i} FedAvg weight: {weight_dict}'
        log_file.info(log_str)
        
    SaveCheckPoint(args, global_model, args.comm, os.path.join(log_dir, 'checkpoints'), note='last_model')
    
if __name__ == '__main__':
    main()
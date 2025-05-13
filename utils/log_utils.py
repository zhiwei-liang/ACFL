from configs.default import log_count_path
import os
import logging


import json
import time
import argparse

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def _Get_Log_Num(log_count_file):
    if os.path.exists(log_count_file):
        with open(log_count_file, 'r') as f:
            file_content = f.readlines()
        log_num = int(file_content[-1]) + 1
        with open(log_count_file, 'a') as f:
            f.writelines(str(log_num) + '\n')
    else:
        with open(log_count_file, 'w') as f:
            f.writelines('0\n')
        log_num = 0
    return log_num

def _Get_Log_Name(args, log_num, file_name='train'):
    start_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime())
    log_name = f"{log_num:d}-{start_time}-{file_name}-{args.dataset if 'dataset' in args else ''}-{args.model if 'model' in args else ''}"\
        + f"-{args.lr if 'lr' in args else ''}-bs{args.batch_size if 'batch_size' in args else ''}-comm{args.comm if 'comm' in args else ''}"\
        +f"-{args.note if 'note' in args else ''}"
    return log_name

def Get_Logger(file_name, file_save=True, display=True):
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%Y-%m-%d %H:%M")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # file handle
    if file_save:
        if os.path.isfile(file_name):
            fh = logging.FileHandler(file_name, mode='a', encoding='utf-8')
        else:
            fh = logging.FileHandler(file_name, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # controler handle
    if display:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger

def Save_Hyperparameter(log_dir, args):
    with open(log_dir + 'hyper_parameter.json', 'w') as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))

def Load_args(log_dir):
    with open(log_dir + 'hyper_parameter.json', 'r') as f:
        json_file = f.read()
    args_dict = json.loads(json_file)
    args = argparse.Namespace(**args_dict)
    return args


def Gen_Log_Dir(args, file_name='train', tensorboard_subdir=True):

    log_count_file = log_count_path + 'log_count.txt'
    log_num = _Get_Log_Num(log_count_file)
    log_name = _Get_Log_Name(args=args, log_num=log_num, file_name=file_name)

    log_dir = log_count_path + log_name + '/'    
    mkdirs(log_dir)

    if tensorboard_subdir: 
        tensorboard_dir = log_dir + '/tensorboard/'
        mkdirs(tensorboard_dir)
        return log_dir, tensorboard_dir
    else:
        return log_dir

def Default_Config(args, default_dict):
    args = _Add_Config(args, default_dict)
    
    return args

def _Add_Config(args, config_dict):
    var_args = vars(args)
    key_list = var_args.keys()
    for key, var in config_dict.items():
        if key in key_list:
            pass
        else:
            var_args[key] = var
    return args

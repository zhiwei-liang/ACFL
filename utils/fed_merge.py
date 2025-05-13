def Dict_weight(dict_in, weight_in):
    # 遍历dict_in中所有参数进行加权weight_in
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in
    
def Dict_Add(dict1, dict2):
    # 两个权重对应健，值相加
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    # 两个权重对应健，值相减
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Cal_Weight_Dict(dataset_dict, site_list=None):
    # 根据每个客户端数据集中测试集的大小计算权重
    if site_list is None:
        site_list = list(dataset_dict.keys())
    weight_dict = {}
    total_len = 0
    for site_name in site_list:
        total_len += len(dataset_dict[site_name]['test'])
    for site_name in site_list:
        site_len = len(dataset_dict[site_name]['test'])
        weight_dict[site_name] = site_len/total_len
    return weight_dict

def FedAvg(model_dict, weight_dict, global_model=None):
    '''联邦平均算法'''
    new_model_dict = None
    for model_name in weight_dict.keys():
        model = model_dict[model_name]
        model_state_dict = model.state_dict()
        if new_model_dict is None:
            # 全局模型为空时，直接赋值
            new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
        else:
            new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
    
    if global_model is None:
        return new_model_dict
    else:
        global_model.load_state_dict(new_model_dict)
        return new_model_dict

def FedUpdate(model_dict, global_model):
    '''使用全局模型初始化本地模型'''
    global_model_parameters = global_model.state_dict()
    for site_name in model_dict.keys():
        model_dict[site_name].load_state_dict(global_model_parameters)
    return None


def MomentumUpdate(model, teacher, alpha=0.99):
    teacher_dict = teacher.state_dict()
    model_dict = model.state_dict()
    for k,v in teacher_dict.items():
        teacher_dict[k] = alpha * v + (1-alpha)*model_dict[k]
    teacher.load_state_dict(teacher_dict)
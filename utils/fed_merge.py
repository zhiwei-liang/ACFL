def Dict_weight(dict_in, weight_in):
    # Traverse all the parameters in dict_in for weighting weight_in
    for k,v in dict_in.items():
        dict_in[k] = weight_in*v
    return dict_in
    
def Dict_Add(dict1, dict2):
    # Two weights correspond to keys, and the values are added together
    for k,v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1

def Dict_Minus(dict1, dict2):
    # Two weights correspond to keys, and the values are subtracted
    for k,v in dict1.items():
        dict1[k] = v - dict2[k]
    return dict1

def Cal_Weight_Dict(dataset_dict, site_list=None):
    # Calculate the weights based on the size of the test set in each client dataset
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
    '''FedAvg'''
    new_model_dict = None
    for model_name in weight_dict.keys():
        model = model_dict[model_name]
        model_state_dict = model.state_dict()
        if new_model_dict is None:
            # When the global model is empty, assign values directly
            new_model_dict = Dict_weight(model_state_dict, weight_dict[model_name])
        else:
            new_model_dict = Dict_Add(new_model_dict, Dict_weight(model_state_dict, weight_dict[model_name]))
    
    if global_model is None:
        return new_model_dict
    else:
        global_model.load_state_dict(new_model_dict)
        return new_model_dict

def FedUpdate(model_dict, global_model):
    '''Initialize the local model using the global model'''
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
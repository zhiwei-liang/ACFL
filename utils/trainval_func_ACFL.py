import os
from network.get_network_ACFL import GetNetwork
import torch
from configs.default import *
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
from utils.losses import con_loss

def Shuffle_Batch_Data(data_in):
    len_total = len(data_in)
    idx_list = list(range(len_total))
    random.shuffle(idx_list)
    return data_in[idx_list]

def epoch_site_train(epochs, site_name, model, optimzier, scheduler, dataloader, log_ten, metric, pro_r, client_name_idx, discriminator):
    model.train()
    for i, data_list in enumerate(dataloader):
        imgs, labels, domain_labels = data_list
        imgs = imgs.cuda()
        labels = labels.cuda()
        domain_labels = domain_labels.cuda()
        optimzier.zero_grad()
        output, representation = model(imgs, "main")
        loss = F.cross_entropy(output, labels)
        if pro_r is not  None:
            with torch.no_grad():
                r = discriminator(representation, 'get_representations')
            loss_con = con_loss(client_name_idx[site_name], r, pro_r, imgs.shape[0], temperature=0.07)
                            
            loss += 0.3 * loss_con
        loss.backward()
        optimzier.step()
        log_ten.add_scalar(f'{site_name}_train_loss', loss.item(), epochs*len(dataloader)+i)
        metric.update(output, labels)
    
    log_ten.add_scalar(f'{site_name}_train_acc', metric.results()['acc'], epochs)
    scheduler.step()
    
def site_train(comm_rounds, site_name, args, model, optimizer, scheduler, dataloader, log_ten, metric, pro_r, client_name_idx, discriminator):
    tbar = tqdm(range(args.local_epochs))
    for local_epoch in tbar:
        tbar.set_description(f'{site_name}_train')
        epoch_site_train(comm_rounds*args.local_epochs + local_epoch, site_name, model, optimizer, scheduler, dataloader, log_ten, metric, pro_r, client_name_idx, discriminator)

    # After the training is completed, return the feature set
    representations = []
    with torch.no_grad():
        for i, data_list in enumerate(dataloader):
            imgs, labels, domain_labels = data_list
            imgs = imgs.cuda()
            r = model(imgs, "get_representations")
            representations.append(r)

    representations = np.concatenate(representations, axis=0)
    return representations



def site_evaluation(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

    return results_dict

def site_evaluation_class_level(epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
    log_ten.add_scalar(f'{note}_{site_name}_class_acc', results_dict['class_level_acc'], epochs)
    log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}% | C Acc: {results_dict["class_level_acc"]*100:.2f}%')

    return results_dict

def site_only_evaluation(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for imgs, labels, domain_labels, in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)
            metric.update(output, labels)
    results_dict = metric.results()
    return results_dict

def GetFedModel(args, num_classes, is_train=True):
    global_model, feature_level = GetNetwork(args, args.num_classes, True)
    global_model = global_model.cuda()
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}
    
    if args.dataset == 'pacs':
        domain_list = pacs_domain_list
    elif args.dataset == 'officehome':
        domain_list = officehome_domain_list
    elif args.dataset == 'domainNet':
        domain_list = domainNet_domain_list
    elif args.dataset == 'terrainc':
        domain_list = terra_incognita_list
        
    for domain_name in domain_list:
        model_dict[domain_name], _ = GetNetwork(args, num_classes, is_train)
        model_dict[domain_name] = model_dict[domain_name].cuda()
        optimizer_dict[domain_name] = torch.optim.SGD(model_dict[domain_name].parameters(), lr=args.lr, momentum=0.9,
                                                      weight_decay=5e-4)
        total_epochs = args.local_epochs * args.comm
        if args.lr_policy == 'step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.StepLR(optimizer_dict[domain_name], step_size=int(total_epochs *0.8), gamma=0.1)
        elif args.lr_policy == 'mul_step':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.MultiStepLR(optimizer_dict[domain_name], milestones=[int(total_epochs*0.3), int(total_epochs*0.8)], gamma=0.1)
        elif args.lr_policy == 'exp95':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.95)
        elif args.lr_policy == 'exp98':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.98)
        elif args.lr_policy == 'exp99':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.ExponentialLR(optimizer_dict[domain_name], gamma=0.99)   
        elif args.lr_policy == 'cos':
            scheduler_dict[domain_name] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dict[domain_name], T_max=total_epochs)
            
    return global_model, model_dict, optimizer_dict, scheduler_dict

def SaveCheckPoint(args, model, epochs, path, optimizer=None, schedule=None, note='best_val'):
    check_dict = {'args':args, 'epochs':epochs, 'model':model.state_dict(), 'note': note}
    if optimizer is not None:
        check_dict['optimizer'] = optimizer.state_dict()
    if schedule is not None:
        check_dict['shceduler'] = schedule.state_dict()
    if not os.path.isdir(path):
        os.makedirs(path)
        
    torch.save(check_dict, os.path.join(path, note+'.pt'))
    

# Update the global discriminator
from utils.losses import SupConLoss
def update_global_discriminator(args, discriminator, r_dataloader):
    criterion = SupConLoss(temperature=args.temp)
    criterion = criterion.cuda()
    discriminator.cuda()
    discriminator.train()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.dis_learning_rate, weight_decay= 1e-6, amsgrad=True)

    for epoch in range(args.rk_iters):
        for batch in r_dataloader:
            inputs, labels = batch
            bsz = labels.shape[0]
            inputs, labels = inputs.cuda(), labels.cuda()
            
            y_preds = discriminator(inputs) # y_preds torch.Size([64, 128])
            
            # Supervised contrastive learning
            loss = criterion(y_preds, labels=labels, test=None)

            discriminator_optimizer.zero_grad()
            loss.backward()
            discriminator_optimizer.step()
        
        if (epoch + 1) % args.print_freq == 0:
            print('Global discriminator train: [{0}]\t'
                'loss: {loss:.3f}'.format(
                epoch + 1, loss=loss.item()))
                
    # Return to the global feature prototype
    pro_representations = _get_pro_representations(r_dataloader, discriminator)

    return pro_representations

def _get_pro_representations(r_dataloader, discriminator):
    model = discriminator
    model.cuda()
    model.eval()

    # get representations
    # For different categories, the prototypes of different categories are obtained through the averaging method.
    pro_representations = {}
    representations = {}
    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(r_dataloader):
            model.zero_grad()
            x = x.cuda()
            r = model(x, "get_representations")
            r = r.cpu().numpy()
            labels = labels.cpu().numpy()
            for i in range(len(labels)):
                label = labels[i]
                feature = r[i]
                if label not in representations:
                    representations[label] = []
                representations[label].append(feature)

    from sklearn.cluster import KMeans
    for label in representations:
        kmeans = KMeans(n_clusters=4, n_init='auto')
        kmeans.fit(representations[label])
        # Obtain the clustering centers
        cluster_centers = kmeans.cluster_centers_
        pro_representations[label] = torch.from_numpy(cluster_centers).cuda()
    return pro_representations


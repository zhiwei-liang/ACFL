import torch.nn.functional as F

class Classification(object):
    def __init__(self):
        self.init()
    
    
    def init(self):
        self.pred_list = []
        self.label_list = []
        self.correct_count = 0
        self.total_count = 0
        self.loss = 0
    
    def update(self, pred, label, easy_model=False):
        pred = pred.cpu()
        label = label.cpu()
        
        if easy_model:
            pass
        else:
            loss = F.cross_entropy(pred, label).item() * len(label)
            self.loss += loss
            pred = pred.data.max(1)[1]
        self.pred_list.extend(pred.numpy())
        self.label_list.extend(label.numpy())
        self.correct_count += pred.eq(label.data.view_as(pred)).sum()
        self.total_count += len(label)
            
    def results(self):
        result_dict = {}
        result_dict['acc'] = float(self.correct_count) / float(self.total_count)
        result_dict['loss'] = float(self.loss) / float(self.total_count)
        self.init()
        return result_dict

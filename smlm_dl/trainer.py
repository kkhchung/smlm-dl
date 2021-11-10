import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from functools import partial
    
class FittingTrainer(object):
    _default_optimizer = partial(torch.optim.Adam, lr=1e-4)    
    
    def __init__(self, model, train_data_loader, valid_data_loader=None, optimizer=None, loss_function=nn.MSELoss(), try_cuda=True):
        
        self.model = model
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        
        if optimizer is None:
            self.optimizer = self._default_optimizer(model.parameters())
        else:
            self.optimizer = optimizer
            
        self.loss_function = loss_function
        # self.logger = logger
        
        self.set_device(try_cuda)
    
    def set_device(self, try_cuda=True):        
        if try_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA not available. Defaulting to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        print("Device: {}".format(self.device))
        
    def train_single_epoch(self, epoch_i=0, log_interval=10, tb_logger=None, tb_log_limit_images=16):
        self.model.to(self.device)
        self.model.train()
        
        # print("-"*100)
        print("Starting training Epoch # {}".format(epoch_i))
        
        for batch_i, (x, y) in enumerate(self.train_data_loader):
            x = x.to(self.device)
            # y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(x)
            
            loss = self.loss_function(pred, x)
            loss.backward()
            self.optimizer.step()
            
            if ((batch_i+1) % log_interval == 0) or ((batch_i+1)==len(self.train_data_loader)):
                print("Epoch # {}, Batch # {} ({}/{}), loss = {:.6f}".format(epoch_i, batch_i,
                                                                     self.train_data_loader.batch_size * (batch_i+1),
                                                                     len(self.train_data_loader.dataset),
                                                                     loss))
                
                if not tb_logger is None:
                    n_iter = epoch_i * len(self.train_data_loader) + batch_i + 1
                    tb_logger.add_scalar("Training/loss", loss, n_iter)
                    tb_logger.add_images("Training/data", x.detach()[:tb_log_limit_images], n_iter)
                    tb_logger.add_images("Training/pred", pred.detach()[:tb_log_limit_images], n_iter)
                    
                    if hasattr(self.model, 'get_suppl'):
                        suppls_dict = self.model.get_suppl()
                        for i, (key, val) in enumerate(suppls_dict.items()):
                            tb_logger.add_image("Training/{}".format(key), val, n_iter, dataformats="HW")
                
        # print("-"*100)
                
    def validate(self, n_iter=0, tb_logger=None, tb_log_limit_images=16):
        self.model.to(self.device)
        self.model.eval()
        
        sum_loss = 0
        
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(self.valid_data_loader):
                x = x.to(self.device)
                pred = self.model(x)
                sum_loss += self.loss_function(pred, x)
        
        loss = sum_loss / len(self.valid_data_loader)
        
        print("*"*100)
        print("Validation, average loss = {:.6f}".format(loss))
        print("*"*100)
        
        if not tb_logger is None:
            tb_logger.add_scalar("Validate/Loss", loss, n_iter) # avg loss
            tb_logger.add_images("Validate/data", x.detach()[:tb_log_limit_images], n_iter) # only images of the last batch
            tb_logger.add_images("Validate/pred", pred.detach()[:tb_log_limit_images], n_iter) # only images of the last batch
            
            if hasattr(self.model, 'get_suppl'):
                suppls_dict = self.model.get_suppl()
                for i, (key, val) in enumerate(suppls_dict.items()):
                    tb_logger.add_image("Validate/{}".format(key), val, n_iter, dataformats="HW")
        
    def train_and_validate(self, n_epoch=100, validate_interval=10, tb_logger=SummaryWriter()):
        """
        
        """
        for epoch_i in range(n_epoch):
            self.train_single_epoch(epoch_i, tb_logger=tb_logger)
            
            if ((epoch_i+1) % validate_interval == 0) or ((epoch_i+1)==n_epoch):
                self.validate((epoch_i+1) * len(self.train_data_loader), tb_logger=tb_logger)
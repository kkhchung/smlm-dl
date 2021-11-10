import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from functools import partial
    
class FittingTrainer(object):
    _default_optimizer = partial(torch.optim.Adam, lr=1e-4)    
    
    def __init__(self, model, data_loader, optimizer=None, loss_function=nn.MSELoss(), try_cuda=True):
        
        self.model = model
        self.data_loader = data_loader
        
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
        
    def train_single_epoch(self, epoch_i=0, log_interval=10, tb_logger=None):
        self.model.to(self.device)
        self.model.train()
        
        print("-"*100)
        print("Starting training Epoch # {}".format(epoch_i))
        
        for batch_i, (x, y) in enumerate(self.data_loader):
            x = x.to(self.device)
            # y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(x)
            
            loss = self.loss_function(pred, x)
            loss.backward()
            self.optimizer.step()
            
            if ((batch_i+1) % log_interval == 0) or ((batch_i+1)==len(self.data_loader)):
                print("Epoch # {}, Batch # {} ({}/{}), loss = {:.6f}".format(epoch_i, batch_i,
                                                                     self.data_loader.batch_size * (batch_i+1),
                                                                     len(self.data_loader.dataset),
                                                                     loss))
                
                if not tb_logger is None:
                    n_iter = epoch_i * len(self.data_loader) + batch_i + 1
                    tb_logger.add_scalar("Training/loss", loss, n_iter)
                    tb_logger.add_images("Training/data", x.detach(), n_iter)
                    tb_logger.add_images("Training/pred", pred.detach(), n_iter)
                
        print("-"*100)
                
    def validate(self, n_iter=0, tb_logger=None):
        self.model.to(self.device)
        self.model.eval()
        
        sum_loss = 0
        
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(self.data_loader):
                x = x.to(self.device)
                pred = self.model(x)
                sum_loss += self.loss_function(pred, x)
        
        loss = sum_loss / len(self.data_loader)
        
        print("*"*100)
        print("Validation, average loss = {:.6f}".format(loss))
        print("*"*100)
        
        if not tb_logger is None:
            tb_logger.add_scalar("Validate/Loss", loss, n_iter) # avg loss
            tb_logger.add_images("Validate/data", x.detach(), n_iter) # only images of the last batch
            tb_logger.add_images("Validate/pred", pred.detach(), n_iter) # only images of the last batch
        
    def train_and_validate(self, n_epoch=100, validate_interval=10, tb_logger=SummaryWriter()):
        """
        For the fitting model, there is no validation dataset.
        The only difference with validation is that the model is in eval mode (no dropouts, etc) and no_grad.
        
        """
        for epoch_i in range(n_epoch):
            self.train_single_epoch(epoch_i, tb_logger=tb_logger)
            
            if ((epoch_i+1) % validate_interval == 0) or ((epoch_i+1)==n_epoch):
                self.validate((epoch_i+1) * len(self.data_loader), tb_logger=tb_logger)
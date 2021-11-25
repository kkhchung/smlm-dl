import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from functools import partial
import os
import time

import numpy as np
from matplotlib import pyplot as plt
    
class FittingTrainer(object):
    _default_optimizer = partial(torch.optim.Adam, lr=1e-4)
    current_state = {}
    
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
                    tb_logger.add_images("Training/data", self.normalize_images(x.detach()[:tb_log_limit_images]), n_iter)
                    tb_logger.add_images("Training/pred", self.normalize_images(pred.detach()[:tb_log_limit_images]), n_iter)
                    tb_logger.add_images("Training/diff", self.normalize_images(x.detach()[:tb_log_limit_images]-pred.detach()[:tb_log_limit_images]), n_iter)
                    
                    if hasattr(self.model, 'get_suppl'):
                        suppls_dict = self.model.get_suppl()
                        for i, (key, val) in enumerate(suppls_dict.items()):
                            tb_logger.add_image("Training/{}".format(key), self.normalize_images(val), n_iter, dataformats="HW")
                
        # print("-"*100)
        
        self.current_state['epoch'] = epoch_i
        self.current_state['loss'] = loss
        if not tb_logger is None:
            self.current_state['filename'] = os.path.split(tb_logger.get_logdir())[-1]
            self.current_state['log_path'] = tb_logger.log_dir
                
    def validate(self, n_iter=0, tb_logger=None, tb_log_limit_images=16, show_images=True):
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
            tb_logger.add_images("Validate/data", self.normalize_images(x.detach()[:tb_log_limit_images]), n_iter) # only images of the last batch
            tb_logger.add_images("Validate/pred", self.normalize_images(pred.detach()[:tb_log_limit_images]), n_iter) # only images of the last batch
            tb_logger.add_images("Validate/diff", self.normalize_images(x.detach()[:tb_log_limit_images]-pred.detach()[:tb_log_limit_images]), n_iter) # only images of the last batch
            
            if hasattr(self.model, 'get_suppl'):
                suppls_dict = self.model.get_suppl()
                fig, axes = plt.subplots(1, len(suppls_dict), figsize=(len(suppls_dict)*4, 3), squeeze=False)
                for i, (key, val) in enumerate(suppls_dict.items()):
                    tb_logger.add_image("Validate/{}".format(key), self.normalize_images(val), n_iter, dataformats="HW")
                    im=axes[0,i].imshow(val)
                    plt.colorbar(im, ax=axes[0,i])
                    axes[0,i].set_title(key)
        
        if show_images:
            fig, axes = plt.subplots(2, 1, figsize=(min([x.shape[0], tb_log_limit_images])*4, 3*2))
            im=axes[0].imshow(np.hstack(x.detach()[:tb_log_limit_images].mean(axis=1), ))
            plt.colorbar(im, ax=axes[0])
            axes[0].set_title('data')
            axes[1].imshow(np.hstack(pred.detach()[:tb_log_limit_images].mean(axis=1), ))
            plt.colorbar(im, ax=axes[1])
            axes[1].set_title('pred')
        
    def train_and_validate(self, n_epoch=100, validate_interval=10, tb_logger=None, checkpoint_interval=1000):
        """
        
        """        
        if tb_logger is None:
            tb_logger = SummaryWriter()
        
        if True: # always pickle the model on running
            self.save_model(os.path.split(tb_logger.get_logdir())[-1])
        
        for epoch_i in range(n_epoch):
            self.train_single_epoch(epoch_i, tb_logger=tb_logger)
            
            if ((epoch_i+1) % validate_interval == 0) or ((epoch_i+1)==n_epoch):
                self.validate((epoch_i+1) * len(self.train_data_loader), tb_logger=tb_logger, show_images=(epoch_i+1)==n_epoch)
                
            if ((epoch_i+1) % checkpoint_interval == 0) or ((epoch_i+1)==n_epoch):
                self.save_checkpoint()
                
    def normalize_images(self, img, vmin=None, vmax=None):
        if vmin is None:
            vmin = img.min()
        if vmax is None:
            vmax = img.max()
        return (img - vmin) / (vmax - vmin)
    
    def save_checkpoint(self, filename=None):
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_function_state_dict": self.loss_function.state_dict(),
        }
        state_dict.update(self.current_state)
        if not filename is None:
            state_dict["filename"] = filename
        save_path = os.path.join("checkpoints", state_dict["filename"] + ".ptc")
        torch.save(state_dict, save_path)
        
        print("Saved to : {}".format(save_path))
        for key, val in state_dict.items():
            if isinstance(val, dict):
                print("{}: {}".format(key, val.keys()))
            else:
                print("{}: {}".format(key, val))
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint.pop("model_state_dict"))
        self.optimizer.load_state_dict(checkpoint.pop("optimizer_state_dict"))
        self.loss_function.load_state_dict(checkpoint.pop("loss_function_state_dict"))
        
        print("Loaded from {}, last modified: {}".format(filepath, time.ctime(os.path.getmtime(filepath))))
        print(summary(self.model))
        print(self.optimizer)
        print(self.loss_function)
        for key, val in checkpoint.items():            
            self.current_state['key'] = val
        print(self.current_state)
        
    def save_model(self, filename=None):
        path = self.current_state.pop("filename", None)
        if not filename is None:
            path = filename
        save_path = os.path.join("models", path + ".ptm")
        torch.save(self.model, save_path)
        print("Saved to : {}".format(save_path))
        
    @staticmethod
    def from_model_file(filepath, *args, **kwargs):
        model = torch.load(filepath)
        trainer = FittingTrainer(model, *args, **kwargs)
        print("Loaded from {}, last modified: {}".format(filepath, time.ctime(os.path.getmtime(filepath))))
        print(summary(model))
        return trainer
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

from functools import partial
import os
import time

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import util
    
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
                    self.log_to_tensorboard(tb_logger, "Training", n_iter, loss, x[:tb_log_limit_images], pred[:tb_log_limit_images])                    
                    self.log_param_to_tensorboard(tb_logger, "Training", n_iter, y, self.model.mapped_params)
                
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
        y_params = {}
        pred_params = {}
        
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(self.valid_data_loader):
                x = x.to(self.device)
                pred = self.model(x)
                sum_loss += self.loss_function(pred, x)                
                
                for old_dict, new_dict in [(y_params, y), (pred_params, self.model.mapped_params)]:
                    for key, val in new_dict.items():
                        if key in old_dict:
                            old_dict[key] = torch.cat([old_dict[key], torch.atleast_1d(torch.as_tensor(val))])
                        else:
                            old_dict[key] = torch.atleast_1d(torch.as_tensor(val))
        
        loss = sum_loss / len(self.valid_data_loader)
        
        print("*"*100)
        print("Validation, average loss = {:.6f}".format(loss))
        print("*"*100)
        
        if not tb_logger is None:
            self.log_to_tensorboard(tb_logger, "Validate", n_iter, loss, x[:tb_log_limit_images], pred[:tb_log_limit_images])
            self.log_param_to_tensorboard(tb_logger, "Validate", n_iter, y_params, pred_params)
        
        if show_images:
            x_numpy = x[:tb_log_limit_images].detach().numpy().mean(axis=1)
            pred_numpy = pred[:tb_log_limit_images].detach().numpy().mean(axis=1)
            vmin = min(x_numpy.min(), pred_numpy.min())
            vmax = max(x_numpy.max(), pred_numpy.max())
            
            n_col = 8
            n_row = 0
            x_numpy_tiled, _n_col, _n_row = util.tile_images(x_numpy, n_col=n_col)
            n_row += _n_row
            pred_numpy_tiled, _n_col, _n_row = util.tile_images(pred_numpy, n_col=n_col)
            n_row += _n_row
            
            fig, axes = plt.subplots(2, 1, figsize=(n_col*2, n_row*1.5))
            im=axes[0].imshow(x_numpy_tiled, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=axes[0])
            axes[0].set_title('data')
            axes[1].imshow(pred_numpy_tiled, vmin=vmin, vmax=vmax)
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
                
    def log_to_tensorboard(self, tb_logger, label, n_iter, loss, x, pred):
        tb_logger.add_scalar("{}/loss".format(label), loss, n_iter)
        x_numpy = x.detach().numpy()
        pred_numpy = pred.detach().numpy()
        vmin = min(x_numpy.min(), pred_numpy.min())
        vmax = max(x_numpy.max(), pred_numpy.max())
        tb_logger.add_images("{}/data".format(label), util.color_images(x_numpy, vmin=vmin, vmax=vmax), n_iter)
        tb_logger.add_images("{}/pred".format(label), util.color_images(pred_numpy, vmin=vmin, vmax=vmax), n_iter)
        tb_logger.add_images("{}/diff".format(label), util.color_images(pred_numpy-x_numpy, vmin=vmin, vmax=vmax, vsym=True), n_iter)
        
        if hasattr(self.model, 'get_suppl'):
            suppl_dict = self.model.get_suppl(colored=True)
            if 'images' in suppl_dict:
                for i, (key, (img, norm, cmap)) in enumerate(suppl_dict['images'].items()):
                    tb_logger.add_image("{}/{}".format(label, key), img, n_iter, dataformats="HWC")
                
    def log_param_to_tensorboard(self, tb_logger, label, n_iter, y, pred, params=['x','y','z']):
        for param in params:
            if param in pred and param in y:
                param_y = y[param].squeeze().detach().numpy()
                param_pred = pred[param].squeeze().detach().numpy()
                error = np.std(param_pred - param_y)
                tb_logger.add_scalar("{}/error_{}".format(label, param), error, n_iter)
    
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
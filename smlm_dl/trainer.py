import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

import socket, datetime
import pathlib
from functools import partial
import os
import time

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from . import util, config

from tqdm.auto import tqdm, trange
    
class FittingTrainer(object):
    _default_optimizer = partial(torch.optim.Adam, lr=1e-4)
    current_state = {}
    
    def __init__(self, model, train_data_loader=None, valid_data_loader=None, optimizer=None, loss_function=nn.MSELoss(), try_cuda=True):
        
        self.model = model
        if train_data_loader is None:
            print("No training data supplied. Remember to set prior to training.")
        if valid_data_loader is None:
            print("No validation data supplied. Remember to set prior to training.")
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
        
    def train_single_epoch(self, epoch_i=0, log_interval=10, tb_logger=None, tb_log_limit_images=9, t=None):
        self.model.to(self.device)
        self.model.train()
        
        # print("-"*100)
        # print("Starting training Epoch # {}".format(epoch_i))
        
        if t is None:
            _t = tqdm(total=len(self.train_data_loader))
        else:
            _t = t
            _t.reset(total=len(self.train_data_loader))
        
        for batch_i, (x, y) in enumerate(self.train_data_loader):
            _t.set_description("Training Batch #{}".format(batch_i))
            _t.update()

            x = x.to(self.device)
            # y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model.call_auto(x, y)            
            loss = self.loss_function(pred, x)
            loss.backward()
            self.optimizer.step()
            
            if ((batch_i+1) % log_interval == 0) or ((batch_i+1)==len(self.train_data_loader)):
                # print("Epoch # {}, Batch # {} ({}/{}), loss = {:.6f}".format(epoch_i, batch_i,
                #                                                      self.train_data_loader.batch_size * (batch_i+1),
                #                                                      len(self.train_data_loader.dataset),
                #                                                      loss))

                if not tb_logger is None:
                    n_iter = epoch_i * len(self.train_data_loader) + batch_i + 1
                    self.log_images_to_tensorboard(tb_logger, "Training", n_iter, loss, x[:tb_log_limit_images], pred[:tb_log_limit_images])
                    self.log_params_to_tensorboard(tb_logger, "Training", n_iter, y, self.model.mapped_params)
                    if self.train_data_loader.dataset.target_is_image is True:
                        self.log_images_to_tensorboard(tb_logger, "Training/Ref", n_iter, loss, y[:tb_log_limit_images], pred[:tb_log_limit_images])
                    
            _t.set_postfix(train_loss=loss.detach().cpu().numpy())
        
        if t is None:
            _t.close()
                
        # print("-"*100)
        
        self.current_state['epoch'] = epoch_i
        self.current_state['loss'] = loss
                
    def validate(self, n_iter=0, tb_logger=None, tb_log_limit_images=9, show_images=True):
        self.model.to(self.device)
        self.model.eval()
        
        sum_loss = 0
        y_params = {}
        pred_params = {}
        
        with torch.no_grad():
            for batch_i, (x, y) in enumerate(self.valid_data_loader):
                x = x.to(self.device)
                pred = self.model.call_auto(x, y)
                sum_loss += self.loss_function(pred, x)                
                
                if self.model.encoder.image_input is False:
                    for old_dict, new_dict in [(y_params, y), (pred_params, self.model.mapped_params)]:
                        for key, val in new_dict.items():
                            if key in old_dict:
                                if val.ndim > 0:
                                    old_dict[key] = torch.cat([old_dict[key], val])
                                else:
                                    old_dict[key] = val # should be identical to the old value
                            else:
                                old_dict[key] = val
        
        loss = sum_loss / len(self.valid_data_loader)
        
        # print("*"*100)
        # print("Validation, average loss = {:.6f}".format(loss))
        # print("*"*100)
        
        if not tb_logger is None:
            self.log_images_to_tensorboard(tb_logger, "Validate", n_iter, loss, x[:tb_log_limit_images], pred[:tb_log_limit_images])
            self.log_params_to_tensorboard(tb_logger, "Validate", n_iter, y_params, pred_params)
            if self.valid_data_loader.dataset.target_is_image is True:
                self.log_images_to_tensorboard(tb_logger, "Validate/Ref", n_iter, loss, y[:tb_log_limit_images], pred[:tb_log_limit_images])
        
        if show_images:
            img_limit = 16
            x_numpy = x[:img_limit].detach().cpu().numpy().mean(axis=1, keepdims=True)
            pred_numpy = pred[:img_limit].detach().cpu().numpy().mean(axis=1, keepdims=True)
            vmin = min(x_numpy.min(), pred_numpy.min())
            vmax = max(x_numpy.max(), pred_numpy.max())
            
            diff = pred_numpy - x_numpy
            diff_vmax = max(-diff.min(), diff.max())
            diff_vmin = -diff_vmax
            
            n_col = min(8, img_limit)
            n_row = 0
            x_numpy_tiled, _n_col, _n_row = util.tile_images(x_numpy, n_col=n_col, full_output=True)
            n_row += _n_row
            pred_numpy_tiled, _n_col, _n_row = util.tile_images(pred_numpy, n_col=n_col, full_output=True)
            n_row += _n_row
            diff_tiled, _n_col, _n_row = util.tile_images(diff, n_col=n_col, full_output=True)
            n_row += _n_row
            
            fig, axes = plt.subplots(3, 1, figsize=(n_col*4, n_row*3))
            for i, (label, (img, vmin, vmax, vsym)) in enumerate({'data': (x_numpy_tiled, vmin, vmax, False),
                                                         'pred': (pred_numpy_tiled, vmin, vmax, False),
                                                         'diff': (diff_tiled, diff_vmin, diff_vmax, True),
                                                         }.items()):
                colored_img, norm, cmap = util.color_images(img, vmin=vmin, vmax=vmax, vsym=vsym, full_output=True)
                colored_img = np.moveaxis(colored_img[0], 0, -1)
                axes[i].imshow(colored_img)
                plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[i])
                axes[i].set_title(label)
            
        return loss
        
    def train_and_validate(self, n_epoch=100, training_interval=10, validate_interval=100,
                           checkpoint_interval=1000, label=None, tb_logger=True, tb_log_limit_images=9):
        """
        
        """
        self.set_logpath(label)
        
        if tb_logger is True:
            tb_logger = SummaryWriter(log_dir=self.current_state["log_path"])
        
        if True: # always pickle the model on running
            self.save_model()
        
        with trange(n_epoch) as t0:
            t1 = tqdm()
            for epoch_i in t0:
                t0.set_description("Training Epoch #{}".format(epoch_i))
                
                self.train_single_epoch(epoch_i,
                                        tb_logger=tb_logger if (epoch_i % training_interval == 0) else None,
                                        tb_log_limit_images=tb_log_limit_images, t=t1)

                if (not self.valid_data_loader is None) and (((epoch_i+1) % validate_interval == 0) or ((epoch_i+1)==n_epoch)):
                    validation_loss = self.validate((epoch_i+1) * len(self.train_data_loader), tb_logger=tb_logger,
                                                    show_images=(epoch_i+1)==n_epoch, tb_log_limit_images=tb_log_limit_images)
                    t0.set_postfix(val_loss=validation_loss.detach().cpu().numpy())

                if ((epoch_i+1) % checkpoint_interval == 0) or ((epoch_i+1)==n_epoch):
                    self.save_checkpoint()
            t1.close()
                
    def log_images_to_tensorboard(self, tb_logger, label, n_iter, loss, x, pred):
        tb_logger.add_scalar("{}/loss".format(label), loss, n_iter)
        x_numpy = x.detach().cpu().numpy()
        x_numpy = util.reduce_images_dim(x_numpy, 'skip')
        pred_numpy = pred.detach().cpu().numpy()
        pred_numpy = util.reduce_images_dim(pred_numpy, 'skip')
        vmin = min(x_numpy.min(), pred_numpy.min())
        vmax = max(x_numpy.max(), pred_numpy.max())
        
        diff = pred_numpy - x_numpy
        diff_vmax = max(-diff.min(), diff.max())
        diff_vmin = -diff_vmax
        
        tb_logger.add_image("{}/data".format(label),
                            util.color_images(util.tile_images(x_numpy,3), vmin=vmin, vmax=vmax)[0], n_iter)
        tb_logger.add_image("{}/pred".format(label),
                            util.color_images(util.tile_images(pred_numpy,3), vmin=vmin, vmax=vmax)[0], n_iter)
        tb_logger.add_image("{}/diff".format(label),
                            util.color_images(util.tile_images(diff,3), vmin=diff_vmin, vmax=diff_vmax, vsym=True)[0], n_iter)
        
        if hasattr(self.model, 'get_suppl'):
            suppl_dict = self.model.get_suppl(colored=True)
            if 'images' in suppl_dict:
                for i, (key, (img, norm, cmap)) in enumerate(suppl_dict['images'].items()):
                    tb_logger.add_image("{}/{}".format(label, key), img, n_iter, dataformats="HWC")
                
    def log_params_to_tensorboard(self, tb_logger, label, n_iter, y, pred, params=['x','y','z']):
        for param in params:
            if param in pred and param in y:
                param_y = y[param].squeeze().detach().to(self.device)
                param_pred = torch.as_tensor(pred[param], device=self.device).squeeze().detach()                
                if param_y.ndim > 1 or param_pred.ndim > 1:
                    # estimate of error based on nearest neighbour
                    for i in range(3 - param_y.ndim):
                        param_y = param_y.unsqueeze(-1)
                    for i in range(3 - param_pred.ndim):
                        param_pred = param_pred.unsqueeze(-1)
                    pairwise_distances = torch.cdist(param_y, param_pred)
                    sorted_values, indices = torch.sort(pairwise_distances, dim=1)
                    error = sorted_values[:, :, 0]
                    error = error.std()
                else:
                    error = torch.std(param_pred - param_y)
                tb_logger.add_scalar("{}/error_{}".format(label, param), error, n_iter)
    
    def save_checkpoint(self, filepath=None):
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_function_state_dict": self.loss_function.state_dict(),
        }
        state_dict.update(self.current_state)
        current_path = state_dict["log_path"]
        if not filepath is None:
            current_path = filepath
        save_path = os.path.join(current_path, "checkpoint.ptc")
        torch.save(state_dict, save_path)
        
        print("Saved to : {}".format(save_path))
        for key, val in state_dict.items():
            if isinstance(val, dict):
                print("{}: {}".format(key, val.keys()))
            else:
                print("{}: {}".format(key, val))
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint.get("model_state_dict"))
        self.model = self.model.to(self.device)
        self.loss_function.load_state_dict(checkpoint.get("loss_function_state_dict"))
        self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
        
        print("Loaded from {}, last modified: {}".format(filepath, time.ctime(os.path.getmtime(filepath))))
        print(summary(self.model))
        print(self.optimizer)
        print(self.loss_function)
        for key, val in checkpoint.items():            
            self.current_state['key'] = val
        print(self.current_state)
        
    def save_model(self, filepath=None):
        current_path = self.current_state["log_path"]
        if not filepath is None:
            current_path = filepath
        save_path = os.path.join(current_path, "model.ptm")
        torch.save(self.model, save_path)
        print("Saved to : {}".format(save_path))
        
    def set_logpath(self, label=None):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = "{}_{}".format(current_time, config.config["ID"]["computer"])
        if not label is None:
            filename += "_{}".format(label)
        log_path = os.path.join(config.config["LOG_PATH"]["run"], filename)
        self.current_state["log_path"] = log_path
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def from_model_file(filepath, *args, **kwargs):
        model = torch.load(filepath)
        trainer = FittingTrainer(model, *args, **kwargs)
        print("Loaded from {}, last modified: {}".format(filepath, time.ctime(os.path.getmtime(filepath))))
        print(summary(model))
        return trainer
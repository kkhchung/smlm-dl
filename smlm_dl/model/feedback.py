import torch
from torch import nn

from . import base


class FeedbackModel(base.BaseModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        super().__init__()
        
    def forward(self, x, feedback):
        raise NotImplementedError()
        
    
class DirectConcatFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        if not all([img_size[d]==feedback_size[d] for d in range(len(img_size))]):
            raise Exception("Input and feedback need to have identical H and W.")
        super().__init__()
        self.norm = nn.GroupNorm(1, 1)
        
    def forward(self, x, feedback):
        feedback = torch.tile(feedback.detach(), (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
    
class CropAndConcatFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        super().__init__()
        self.norm = nn.GroupNorm(1, 1)
        self.padding = [int((feedback_size[d]-img_size[d])/2) for d in range(len(feedback_size))]
        
    def forward(self, x, feedback):
        feedback = torch.tile(feedback[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]], (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
    

class DenseFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32), feedback_features=3, depth=1):
        super().__init__()
        self.norm = nn.GroupNorm(1, 1)
        self.feedback = nn.ModuleDict()
        for i in range(depth):
            self.feedback['dense_layer_{}'.format(i)] = self.dense_block(feedback_size[-1], feedback_size[-1], feedback_features)
        
    def forward(self, x, feedback):
        feedback = feedback.detach()
        for key, val in self.feedback.items():
            feedback = val(feedback)
        feedback = torch.tile(feedback, (x.shape[0], 1, 1, 1))
        x = torch.cat([self.norm(x), feedback], dim=1)
        return x
            
    def dense_block(self, in_channels, out_channels, features):
        return nn.Sequential(
            nn.GroupNorm(1, features),
            nn.Linear(in_channels, out_channels, ),
            nn.ReLU(),
            nn.GroupNorm(1, features),
            nn.Linear(out_channels, out_channels, ),
            nn.ReLU(),
            nn.Dropout2d(0.5),
        )

    
class DiffFeedbackModel(FeedbackModel):
    def __init__(self, img_size=(32,32), feedback_size=(32,32)):
        super().__init__()
        
    def forward(self, x, feedback):
        feedback = feedback.detach()
        # feedback = self.normalize(feedback.detach())
        # feedback = torch.tile(feedback, (x.shape[0], 1, 1, 1))
        # feedback = self.scale_to(feedback, x)
        x = torch.cat([x, feedback-x], dim=1)
        return x
    
    def normalize(self, arr):
        arr -= arr.min()
        arr /= arr.max()
        return arr
            
    def scale_to(self, feedback, x):
        feedback += torch.amin(x, dim=(-2,-1), keepdim=True)
        feedback *= torch.amax(x, dim=(-2,-1), keepdim=True) - torch.amin(x, dim=(-2,-1), keepdim=True)
        return feedback
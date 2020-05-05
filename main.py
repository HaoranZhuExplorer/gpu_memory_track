import torch
import inspect
from torchvision import models
from gpu_mem_track import  MemTracker

device = torch.device('cuda:0')

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU
cnn = models.resnet50(pretrained=True).to(device)
gpu_tracker.track()                     # run function between the code line where uses GPU

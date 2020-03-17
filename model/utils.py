import h5py
import torch
import shutil
import os, os.path

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, epoch, filename='checkpoint.pth', path = "."):
    if not os.path.exists(path):
            os.makedirs(path)        
    if is_best: 
        torch.save(state,path + "/" + task_id + "_" + str(epoch)+ filename)
        shutil.copyfile(path + "/" + task_id + "_" + str(epoch)+ filename, path + "/" + task_id+'model_best.pth')            
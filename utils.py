import torch
import os
import h5py
import numpy as np
import pickle
import torchvision

class RunningAverage:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, data, n):
        data = data.detach().item()
        self.count = self.count + n
        self.sum = self.sum + data * n
        self.avg = self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

def save_checkpoint(num_epoch, num_iter, model, best_val_accuracy, optimizer, device, config, is_best, lr_record, train_accuracy_label1, train_accuracy_label2, val_accuracy_label1, val_accuracy_label2, train_loss, val_loss, val_iter, rng_state):
    state_best_checkpoint = {'epochs': num_epoch,
             'iterations': num_iter,
             'model_state_dict': model.state_dict(),
             'best_val_accuracy': best_val_accuracy,
             'optimizer_state_dict': optimizer.state_dict(),
             'device': str(device),
             'config': config,
             'lr_record': lr_record}
    state_last_checkpoint = {'epochs': num_epoch,
             'iterations': num_iter,
             'model_state_dict': model.state_dict(),
             'best_val_accuracy': best_val_accuracy,
             'optimizer_state_dict': optimizer.state_dict(),
             'device': str(device),
             'config': config,
             'lr_record': lr_record,
             'train_accuracy_label1': train_accuracy_label1,
             'train_accuracy_label2': train_accuracy_label2,
             'val_accuracy_label1': val_accuracy_label1,
             'val_accuracy_label2': val_accuracy_label2,
             'train_loss': train_loss,
             'val_loss': val_loss,
             'val_iter': val_iter,
             'rng_state': rng_state}
    checkpoint_dir = config.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    save_last_checkpoint_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state_last_checkpoint, save_last_checkpoint_path)

    if is_best:
        save_best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        torch.save(state_best_checkpoint, save_best_checkpoint_path)

def load_checkpoint(path, device, model, optimizer=None):
    checkpoint_path = path
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        torch.set_rng_state(state['rng_state'].cpu())

    return state

def save_prediction(save_file_path, save_file_name, name1=None, data1=None, name2=None, data2=None, name3=None, data3=None, name4=None, data4=None, name5=None, data5=None):
    with h5py.File(os.path.join(save_file_path, save_file_name), 'w') as file:
        if name1 is not None:
            file.create_dataset(name1,data=data1)
        if name2 is not None:
            file.create_dataset(name2,data=data2)
        if name3 is not None:
            file.create_dataset(name3,data=data3)
        if name4 is not None:
            file.create_dataset(name4,data=data4)
        if name5 is not None:
            file.create_dataset(name5,data=data5)

def learning_rate_scheduler(optimizer, epoch, initial_lr, lr_decay_epoch = 4, mode = 'fixed'):
    """ decay learning rate by a factor of 0.1 every lr_decay_epoch"""
    assert mode in ['fixed', 'accumulated']
    if mode == 'fixed':
        lr = initial_lr * (0.1**(epoch // lr_decay_epoch))
        if epoch % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        lr = initial_lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
 
def save_trends(dict_data, save_index, save_data, save_name):
    """ save loss/accuracy values as a dictionary """
    if save_index not in dict_data:
        dict_data[save_index] = save_data
        f = open(save_name,"wb")
        pickle.dump(dict_data,f)
        f.close()
    return dict_data

def visualize_prediction(input, target, output, save_index, save_path, num_per_row=4):
    for batch_idx in list(range(input.shape[0])): 
        img1 = input[batch_idx, :, :, :].squeeze().cpu()
        img2 = torchvision.utils.make_grid(img1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True) 
        torchvision.utils.save_image(img2,os.path.join(save_path,'image_' + str(save_index) + '_' + str(batch_idx) + '.tiff'), nrow=num_per_row)
        
        pred1 = np.argmax(output[batch_idx,:,:,:,:].squeeze().cpu(), axis=0)
        pred1 = pred1.type(torch.DoubleTensor)
        pred2 = torchvision.utils.make_grid(pred1.unsqueeze(dim=1), nrow=num_per_row, normalize=True) 
        torchvision.utils.save_image(pred2,os.path.join(save_path,'pred_' + str(save_index) + '_' + str(batch_idx)+ '.tiff'), nrow=num_per_row)	

        label1 = target[batch_idx, :, :, :].squeeze().cpu()
        label2 = torchvision.utils.make_grid(label1.unsqueeze(dim=1), nrow=num_per_row, normalize=True) 
        torchvision.utils.save_image(label2,os.path.join(save_path,'label_' + str(save_index) + '_' + str(batch_idx)+ '.tiff'), nrow=num_per_row)

def visualize_prediction_ur(save_index, save_path, num_per_row=8, im1=None, name1=None, im2=None, name2=None, im3=None, name3=None):
    if name1 is not None:
        for batch_idx in list(range(im1.shape[0])): 
            temp1 = im1[batch_idx, :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True) 
            torchvision.utils.save_image(temp2,os.path.join(save_path, name1 + '_' + str(save_index) + '_' + str(batch_idx) + '.tiff'), nrow=num_per_row)
    if name2 is not None:
        for batch_idx in list(range(im2.shape[0])): 
            temp1 = im2[batch_idx,  :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row, normalize=True) 
            torchvision.utils.save_image(temp2,os.path.join(save_path, name2 + '_' + str(save_index) + '_' + str(batch_idx)+ '.tiff'), nrow=num_per_row)	
    if name3 is not None:
        for batch_idx in list(range(im3.shape[0])): 
            temp1 = im3[batch_idx,  :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row, normalize=True) 
            torchvision.utils.save_image(temp2,os.path.join(save_path, name3 + '_' + str(save_index) + '_' + str(batch_idx)+ '.tiff'), nrow=num_per_row)

def visualize_feature_maps(feature_maps, save_index, save_path):
    for batch_id in list(range(feature_maps.shape[0])):
        img1 = feature_maps[batch_id,  :, :].squeeze().cpu()
        img2 = torchvision.utils.make_grid(img1.unsqueeze(dim=1), nrow=8,  normalize=True) 
        torchvision.utils.save_image(img2,os.path.join(save_path,'features_' + str(save_index) +  '_' + str(batch_id) + '.tiff'), nrow=8)

def visualize_last_encoder(feature_maps, save_path, num_per_row=8):
    #image = torch.zeros(feature_maps.shape[0],3,feature_maps.shape[1],feature_maps.shape[2])
    #image[:,0,:,:] = feature_maps
    #image = image.permute(2,1,3,0)
    
    f = open(os.path.join(save_path,'last_encoder_avgpool'),"wb")
    pickle.dump(feature_maps,f)
    f.close()

    image = torch.zeros(1,3,feature_maps.shape[0],feature_maps.shape[1])
    image[:,0,:,:] = feature_maps
    img = torchvision.utils.make_grid(image, nrow=num_per_row,  scale_each=True) 
    torchvision.utils.save_image(img,os.path.join(save_path,'last_encoder_avgpool.tiff'), nrow=num_per_row)

# adjusted for 3 labels  
def visualize_difference(target, output, save_index, save_path, num_per_row=4, type_list = ['background','stomach','intestine']):
    output_onehot = output.cpu()
    output = (np.argmax(output_onehot, axis=1)).unsqueeze(dim=1)
    target = target.cpu()
    output[output==0] = torch.max(target)+1
    output, target = output.type(torch.FloatTensor), target.type(torch.FloatTensor)

    shape = target.size()
    shape_onehot = output_onehot.size()
    
    for i in list(range(len(type_list))):
        image = torch.zeros(shape)
        image[target==i] = 1.
        image = image * output
        display_image = torch.zeros(shape_onehot)
        
        for j in list(range(torch.min(target).type(torch.LongTensor)+1,torch.max(target).type(torch.LongTensor)+2)):
            temp = torch.zeros(shape)
            temp[image==j] = 1.
            display_image[:,j-1,:,:,:] = temp.squeeze()

        # stomach in output
        for batch_idx in list(range(target.shape[0])): 
            display_image2 = torchvision.utils.make_grid((display_image[batch_idx,:,:,:,:]).permute(1,0,2,3),nrow=num_per_row) 
            torchvision.utils.save_image(display_image2,os.path.join(save_path,'diff_' + type_list[i] + str(save_index) + '_' + str(batch_idx) + '.tiff'), nrow=num_per_row)	

def calculate_errors(target, output, save_index, save_path, type_list = ['background','stomach','intestine']):
    error = {}
    output_onehot = output.cpu()
    output = (np.argmax(output_onehot, axis=1)).unsqueeze(dim=1)
    target = target.cpu()
    output[output==0] = torch.max(target)+1
    output, target = output.type(torch.FloatTensor), target.type(torch.FloatTensor)
    shape = target.size()

    for i in list(range(len(type_list))):
        image = torch.zeros(shape)
        image[target==i] = 1.
        image = image * output
        for j in list(range(len(type_list))):
            for batch_idx in list(range(target.shape[0])): 
                image_id = image[batch_idx,:,:,:,:]
                if j == 0:
                    error['id'+str(batch_idx)+'_'+type_list[i]+'_'+type_list[j]] = torch.sum(image_id==torch.max(target)+1)
                else:
                    error['id'+str(batch_idx)+'_'+type_list[i]+'_'+type_list[j]] = torch.sum(image_id==j)
     
    save_name = os.path.join(save_path,'patch' + str(save_index) + '_error')
    f = open(save_name,"wb")
    pickle.dump(error,f)
    f.close()

def create_grid(shape):
    x_len = shape[3]
    y_len = shape[4]
    z_len = shape[2]
    # 1-D vectors: a, b, c
    a = torch.linspace(-1.0, 1.0, steps = x_len).repeat(1, 1, 1).permute(2, 0, 1)
    b = torch.linspace(-1.0, 1.0, steps = y_len).repeat(1, 1, 1).permute(0, 2, 1)
    c = torch.linspace(-1.0, 1.0, steps = z_len).repeat(1, 1, 1).permute(0, 1, 2)
    # 3-D matrixs: x_t, y_t, z_t
    x_t = a.repeat(1, y_len, z_len)
    y_t = b.repeat(x_len, 1, z_len)
    z_t = c.repeat(x_len, y_len, 1)
    # create 5-D grid
    grid_4d = torch.cat((y_t.unsqueeze(3).permute(2, 0, 1, 3),
                         x_t.unsqueeze(3).permute(2, 0, 1, 3),
                         z_t.unsqueeze(3).permute(2, 0, 1, 3)), dim=3)
    grid = grid_4d.unsqueeze(0)
    for _ in range(shape[0]-1): 
        grid = torch.cat((grid, grid_4d.unsqueeze(0)), dim=0)
       
    return grid

def log_images(writer, num_iter, name1=None, data1=None, name2=None, data2=None, name3=None, data3=None, name4=None, data4=None, num_per_row=8):
    if name1 is not None:
        for batch_idx in list(range(data1.shape[0])): 
            temp1 = data1[batch_idx, :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True)
            writer.add_image(name1+'_'+str(batch_idx), temp2, num_iter)
    if name2 is not None:
        for batch_idx in list(range(data2.shape[0])): 
            temp1 = data2[batch_idx, :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True)
            writer.add_image(name2+'_'+str(batch_idx), temp2, num_iter)
    if name3 is not None:
        for batch_idx in list(range(data3.shape[0])): 
            temp1 = data3[batch_idx, :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True)
            writer.add_image(name3+'_'+str(batch_idx), temp2, num_iter)
    if name4 is not None:
        for batch_idx in list(range(data4.shape[0])): 
            temp1 = data4[batch_idx, :, :, :, :].squeeze().cpu()
            temp2 = torchvision.utils.make_grid(temp1.unsqueeze(dim=1), nrow=num_per_row,  normalize=True)
            writer.add_image(name4+'_'+str(batch_idx), temp2, num_iter)

def log_scalars(writer, num_iter, name, data):
    writer.add_scalar(name, data, num_iter)

class data_aug:
    def __init__(self):
        self.switcher = {0: self.data_aug_ch0, 1: self.data_aug_ch1, 2: self.data_aug_ch2, 3: self.data_aug_ch3, 
                         4: self.data_aug_ch4, 5: self.data_aug_ch5, 6: self.data_aug_ch6, 7: self.data_aug_ch7}

    def forward(self, data, seed, device):
        self.idx_1 = [i for i in range(data.size(2)-1, -1, -1)]
        self.idx_1 = torch.LongTensor(self.idx_1)
        self.idx_1 = self.idx_1.to(device)
        self.idx_2 = [i for i in range(data.size(3)-1, -1, -1)]
        self.idx_2 = torch.LongTensor(self.idx_2)
        self.idx_2 = self.idx_2.to(device)
        self.idx_3 = [i for i in range(data.size(4)-1, -1, -1)]
        self.idx_3 = torch.LongTensor(self.idx_3)
        self.idx_3 = self.idx_3.to(device)
        
        self.data = data
        self.seed = seed

        func = self.switcher.get(self.seed, "nothing") 
        return func()      
    
    def data_aug_ch0(self):
        inv_data = self.data
        return inv_data
 
    def data_aug_ch1(self):
        inv_data = self.data.index_select(2, self.idx_1)
        return inv_data
    
    def data_aug_ch2(self):
        inv_data = self.data.index_select(3, self.idx_2)
        return inv_data
 
    def data_aug_ch3(self):
        inv_data = self.data.index_select(4, self.idx_3)
        return inv_data
 
    def data_aug_ch4(self):
        inv_data = self.data.index_select(2, self.idx_1)
        inv_data = inv_data.index_select(3, self.idx_2)
        return inv_data

    def data_aug_ch5(self):
        inv_data = self.data.index_select(2, self.idx_1)
        inv_data = inv_data.index_select(4, self.idx_3)
        return inv_data

    def data_aug_ch6(self):
        inv_data = self.data.index_select(3, self.idx_2)
        inv_data = inv_data.index_select(4, self.idx_3)
        return inv_data

    def data_aug_ch7(self):
        inv_data = self.data.index_select(2, self.idx_1)
        inv_data = inv_data.index_select(3, self.idx_2)
        inv_data = inv_data.index_select(4, self.idx_3)
        return inv_data

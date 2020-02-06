import torch
import os
import numpy as np
import matplotlib.pyplot
import utils
import random
from torch.utils.tensorboard import SummaryWriter

torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

class Trainer:
    def __init__(self, config, model, model2, device, train_loader, val_loader, accuracy_criterion, loss_criterion, optimizer):

        self.config = config
        self.max_val_iter = self.config.patience
        self.model = model
        self.model2 = model2
        self.device = device
        self.output_device = self.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accuracy_criterion = accuracy_criterion
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer

        self.train_loss = utils.RunningAverage()
        self.train_accuracy_label1 = utils.RunningAverage()
        self.train_accuracy_label2 = utils.RunningAverage()
        self.print_term = 20
        self.flag_reduce_lr = False

        self.data_aug = utils.data_aug()

        self.num_params_model = self.get_num_learnable_params(self.model)
        print("========> The number of learnable parameters for model1 is {}".format(self.num_params_model))
        self.num_params_model2 = self.get_num_learnable_params(self.model2)
        print("========> The number of learnable parameters for model2 is {}".format(self.num_params_model2))

        self.writer = SummaryWriter(os.path.join(self.config.checkpoint_dir,'logs'))

        # load the saved checkpoint - update model2 parameters (no need to update model parameters since they are fixed)
        if self.config.consume:
            self.state = utils.load_checkpoint(os.path.join(self.config.checkpoint_dir,"last_checkpoint.pytorch"), self.output_device, self.model2, self.optimizer)
            self.num_epoch = self.state['epochs']
            self.num_iter = self.state['iterations']
            self.best_val_accuracy = self.state['best_val_accuracy']
            self.config = self.state['config']
            self.config.consume = True
            self.val_iter = self.state['val_iter']
            self.lr_record = self.state['lr_record']
            self.dict_train_loss = self.state['train_loss']
            self.dict_train_accuracy_label1 = self.state['train_accuracy_label1']
            self.dict_train_accuracy_label2 = self.state['train_accuracy_label2']
            self.dict_val_loss = self.state['val_loss']
            self.dict_val_accuracy_label1 = self.state['val_accuracy_label1']
            self.dict_val_accuracy_label2 = self.state['val_accuracy_label2']
        else:
            self.num_epoch = 0
            self.num_iter = 0
            self.best_val_accuracy = 0.
            self.val_iter = 0
            self.lr_record = self.config.learning_rate
            self.dict_train_loss = {}
            self.dict_train_accuracy_label1 = {}
            self.dict_train_accuracy_label2 = {}
            self.dict_val_loss = {}
            self.dict_val_accuracy_label1 = {}
            self.dict_val_accuracy_label2 = {}

        if self.config.lr_decay_mode == 'accumulated':
            assert self.config.lr_decay_epoch < self.max_val_iter

        print(self.model2)

    def main(self):
        # set the model in the training mode
        self.model2.train()
        print("Training begins for the small model...")
        # start multiple epochs
        for epoch in range(self.config.epochs):
            
            # save the check point states
            if not os.path.exists(self.config.checkpoint_dir):
                os.mkdir(self.config.checkpoint_dir)
            
            self.train_loss.reset()                   
            self.train_accuracy_label1.reset()      
            self.train_accuracy_label2.reset() 

            # record the number of epochs
            self.num_epoch = self.num_epoch + 1

            # start multiple iterations
            for batch_i, data in enumerate(self.train_loader):

                # record the number of iterations
                self.num_iter = self.num_iter + 1

                # save input data
                _, input, target = data
 
                input, target = input.to(self.device), target.to(self.device)
                input, target = input.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)

                # normalize the input image
                input = input/torch.max(input)
                
                input_ds = torch.nn.functional.interpolate(torch.squeeze(input), (self.config.data_size, self.config.data_size), 
                        mode='nearest').unsqueeze(1)
                # target = torch.nn.functional.interpolate(torch.squeeze(target), (self.config.data_size, self.config.data_size), 
                #        mode='nearest').unsqueeze(1)

                # augment the data randomly 1/7
                random_num = random.randint(0, 7)
                input_ds = self.data_aug.forward(input_ds, random_num, self.device)
                target = self.data_aug.forward(target, random_num, self.device)

                # forward pass for model 
                output_ds, output_ds_last_decoder = self.model(input_ds)
                
                # forward pass for model2
                output = self.model2(output_ds_last_decoder)

                # compute loss and accuracy
                loss = self.loss_criterion(output, target)
                accuracy_indi, _ = self.accuracy_criterion(output, target)

                # update the running average of loss and accuracy values
                self.train_loss.update(data=loss, n=self.config.train_batch_size)
                self.train_accuracy_label1.update(data=accuracy_indi[1], n=self.config.train_batch_size)
                self.train_accuracy_label2.update(data=accuracy_indi[2], n=self.config.train_batch_size)

                # log onto the tensorboard for training data
                utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='train_running_loss', data=self.train_loss.avg)
                utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='train_running_accu_label1', data=self.train_accuracy_label1.avg)
                utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='train_running_accu_label2', data=self.train_accuracy_label2.avg)

                # display results for training
                if batch_i % self.print_term == 0:
                    print("========> Training Epoch {}, Iteration {}, Loss {:.02f}, Accuracy for label 1 {:.02f}, Accuracy for label 2 {:.02f}".format(self.num_epoch, self.num_iter, self.train_loss.avg, self.train_accuracy_label1.avg, self.train_accuracy_label2.avg))
                
                # update learning rate
                if self.config.lr_decay_mode == 'fixed':
                    old_lr = self.lr_record
                    self.lr_record = utils.learning_rate_scheduler(self.optimizer, self.num_epoch, initial_lr = self.config.learning_rate, lr_decay_epoch = self.config.lr_decay_epoch, mode = self.config.lr_decay_mode)
                    new_lr = self.lr_record
                    print("The learning rate is adjusted in the fixed mode from {} to {}".format(old_lr, new_lr))

                # backward process
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # evaluate on the validation data
                if self.num_iter % self.config.validate_after_iters == 0:
                    val_accuracy_label1, val_accuracy_label2, val_loss = self.validate()
                    val_accuracy = (val_accuracy_label1 + val_accuracy_label2 )/2
                    if val_accuracy > self.best_val_accuracy:
                        is_best = True
                        self.best_val_accuracy = val_accuracy
                        self.val_iter = 0
                    else:
                        is_best = False
                        self.val_iter = self.val_iter + 1
                    # save model2 parameter (no need to save model parameter w/o update)
                    utils.save_checkpoint(self.num_epoch, self.num_iter, self.model2, self.best_val_accuracy,
                    self.optimizer, self.device, self.config, is_best, self.lr_record, self.dict_train_accuracy_label1, self.dict_train_accuracy_label2, self.dict_val_accuracy_label1, self.dict_val_accuracy_label2, self.dict_train_loss, self.dict_val_loss, self.val_iter, torch.get_rng_state())

                    # log onto the tb for the validation data
                    utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='val_running_loss', data=val_loss)
                    utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='val_running_accu_label1', data=val_accuracy_label1)
                    utils.log_scalars(writer=self.writer, num_iter=self.num_iter, name='val_running_accu_label2', data=val_accuracy_label2)
                    
                    if self.val_iter == self.config.lr_decay_epoch:
                        self.flag_reduce_lr = True
                    if self.val_iter == self.max_val_iter:
                        print('Reach the maximum patience...')
                        break

                if self.val_iter == self.max_val_iter:
                    break
            # save the trend for averaged loss and accuracy for the current epoch
            self.dict_train_loss = utils.save_trends(self.dict_train_loss, self.num_epoch, self.train_loss.avg, os.path.join(self.config.checkpoint_dir, 'train_loss'))
            self.dict_train_accuracy_label1 = utils.save_trends(self.dict_train_accuracy_label1, self.num_epoch, self.train_accuracy_label1.avg, os.path.join(self.config.checkpoint_dir, 'train_accuracy_label1'))
            self.dict_train_accuracy_label2 = utils.save_trends(self.dict_train_accuracy_label2, self.num_epoch, self.train_accuracy_label2.avg, os.path.join(self.config.checkpoint_dir, 'train_accuracy_label2'))
            
            if self.val_iter == self.max_val_iter:
                print("Validation accuracy did not improve for the last {} validation runs. Early stopping..."
                        .format(self.max_val_iter))
                break
            else:
                if self.config.lr_decay_mode == 'accumulated':
                    if  self.flag_reduce_lr:
                        old_lr = self.lr_record
                        self.lr_record = utils.learning_rate_scheduler(self.optimizer, self.num_epoch, initial_lr = self.lr_record, lr_decay_epoch = self.config.lr_decay_epoch, mode = self.config.lr_decay_mode)
                        new_lr = self.lr_record
                        self.flag_reduce_lr = False
                        print("The learning rate is adjusted in the accumulated mode from {} to {}".format(old_lr, new_lr))
                print("Back to training...")
            

    def validate(self):
        val_loss = utils.RunningAverage()
        val_accuracy_label1 = utils.RunningAverage()
        val_accuracy_label2 = utils.RunningAverage()
        print("Validation begins...")
        self.model2.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # save validation data
                _, input, target = data
                input, target = input.to(self.device), target.to(self.device)
                input, target = input.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor)
                
                # normalize the input image
                input = input/torch.max(input)

                input_ds = torch.nn.functional.interpolate(torch.squeeze(input), (self.config.data_size, self.config.data_size), 
                        mode='nearest').unsqueeze(1)
                # target = torch.nn.functional.interpolate(torch.squeeze(target), (self.config.data_size, self.config.data_size), 
                #        mode='nearest').unsqueeze(1)
 
                # augment the data randomly 1/8
                random_num = random.randint(0, 7)
                input_ds = self.data_aug.forward(input_ds, random_num, self.device)
                target = self.data_aug.forward(target, random_num, self.device)

                # forward pass for model
                output_ds, output_ds_last_decoder = self.model(input_ds)
                
                # forward pass for model2
                output = self.model2(output_ds_last_decoder)

                # compute loss and accuracy
                loss = self.loss_criterion(output, target)
                accuracy_indi, _ = self.accuracy_criterion(output, target)

                # update the running average of loss and accuracy values
                val_loss.update(loss, self.config.val_batch_size)
                val_accuracy_label1.update(accuracy_indi[1], self.config.val_batch_size)
                val_accuracy_label2.update(accuracy_indi[2], self.config.val_batch_size)

                # visualize the prediction results
                save_path_visual = os.path.join(self.config.checkpoint_dir,'visual')
                if not os.path.exists(save_path_visual):
                    os.mkdir(save_path_visual)
                utils.visualize_prediction(input, target, output, i, save_path_visual)
                # utils.visualize_difference(target, output, i, save_path_visual)
                
                # save prediction
                save_path_pred = os.path.join(self.config.checkpoint_dir,'pred')
                if not os.path.exists(save_path_pred):
                    os.mkdir(save_path_pred)
                save_name = f'patch{i}_pred.h5'
                utils.save_prediction(save_path_pred, save_name, 'raw', input.cpu(), 'pred', output.cpu(), 'label', target.cpu())

            # display results for training
            prediction = torch.argmax(output, dim=1)
            prediction = torch.unsqueeze(prediction, dim=1)
            prediction = prediction.type(torch.cuda.FloatTensor)
            utils.log_images(writer=self.writer, num_iter=self.num_iter, name1='raw', data1=input, name2='target', data2=target, name3='prediction', data3=prediction, num_per_row=8)
            print("========> Validation Iteration {}, Loss {:.02f}, Accuracy for label 1 {:.02f}, Accuracy for label 2 {:.02f}"
                    .format(i, val_loss.avg, val_accuracy_label1.avg, val_accuracy_label2.avg))
            
            # save trends
            self.dict_val_loss = utils.save_trends(self.dict_val_loss, self.num_epoch, val_loss.avg, os.path.join(self.config.checkpoint_dir, 'val_loss'))
            self.dict_val_accuracy_label1 = utils.save_trends(self.dict_val_accuracy_label1, self.num_epoch, val_accuracy_label1.avg, os.path.join(self.config.checkpoint_dir, 'val_accuracy_label1'))
            self.dict_val_accuracy_label2 = utils.save_trends(self.dict_val_accuracy_label2, self.num_epoch, val_accuracy_label2.avg, os.path.join(self.config.checkpoint_dir, 'val_accuracy_label2'))

        self.model2.train()
        return val_accuracy_label1.avg, val_accuracy_label2.avg, val_loss.avg

    def get_num_learnable_params(self, model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params =  sum([np.prod(p.size()) for p in model_parameters])
        return num_params

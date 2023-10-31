# from comet_ml import Expeiment
import gc
import copy
import torch
import json
from torch import nn
from functools import partial

from pathlib import Path
from torch.optim import Adam
from torchvision import utils

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import torch
# import l1 loss and l2 loss
from torch.nn import L1Loss, MSELoss

from .network import EMA
from .utils import loss_backwards

# trainer class

class simple_trainer_PVC(object):
    def __init__(
        self,
        model,
        data_loader,
        *,
        ema_decay = 0.995,
        image_size = 128,
        time_steps = 1000,
        loss_type = 'l1',
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        load_path = None,
    ):
        super().__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.opt = Adam(model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)

        self.time_steps = time_steps
        self.loss_type = loss_type
        self.loss_fn = L1Loss() if loss_type == 'l1' else MSELoss()
        self.dataloader = data_loader
        self.max_time = torch.tensor(self.time_steps, dtype=torch.float)
        self.max_time = self.max_time.expand(self.batch_size).to(device='cuda')
        self.loss_recorder = []

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, itrs=None):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if itrs is None:
            torch.save(data, str(self.results_folder / f'model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model_{itrs}.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def generate_xt1_xt2(self, data_1, data_2, device):
        # generate t_1 and t_2. t_2 is from 2 to self.time_steps, and t1 is from 1 to t_2 - 1
        t_2_int = torch.randint(3, self.time_steps, (1,))
        t_1_int = torch.randint(1, t_2_int-1, (1,))
        assert t_1_int < t_2_int
        
        # make both t_1 and t_2 of the shape of batch_size
        t_1 = t_1_int.expand(data_1.shape[0])
        t_2 = t_2_int.expand(data_1.shape[0])

        alpha_1 = (t_1_int.float() / self.time_steps) ** (1/3)
        alpha_2 = (t_2_int.float() / self.time_steps) ** (1/3)

        # Explicitly broadcasting alpha_1 and alpha_2
        alpha_1 = alpha_1.view(-1, 1, 1, 1)
        alpha_2 = alpha_2.view(-1, 1, 1, 1)

        data_t1 = (1 - alpha_1) * data_1 + (alpha_1) * data_2
        data_t2 = (1 - alpha_2) * data_1 + (alpha_2) * data_2

        # move to device
        data_t1 = data_t1.to(device)
        data_t2 = data_t2.to(device)
        t_1 = t_1.to(device)
        t_2 = t_2.to(device)

        return data_t1, data_t2, t_1, t_2, t_1_int, t_2_int

    def train(self):
        # experiment = Experiment(api_key="57ArytWuo2X4cdDmgU1jxin77",
        #                         project_name="Cold_Diffusion_Cycle")

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data_1, data_2 = next(self.dataloader)
                
                # for PET, divide the data by its max value
                data_2 = data_2 / data_2.max()

                data_t1, data_t2, t_1, t_2, _, _ = self.generate_xt1_xt2(data_1, data_2, device='cuda')

                data_t2_hat = self.model(data_t1, t_2-t_1)
                loss = self.loss_fn(data_t2_hat, data_t2)

                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
                    self.loss_recorder.append(loss.item())
                u_loss += loss.item()
                backwards(loss / self.gradient_accumulate_every, self.opt)

            acc_loss = acc_loss + (u_loss/self.gradient_accumulate_every)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # experiment.log_current_epoch(self.step)
                milestone = self.step // self.save_and_sample_every
                data_1, data_2 = next(self.dataloader)

                # for PET, divide the data by its max value
                data_2 = data_2 / data_2.max()

                data_t1, data_t2, t_1, t_2, t_1_int, t_2_int = self.generate_xt1_xt2(data_1, data_2, device='cuda')
                # create max_time as a tensor of shape batch_size
                # given that the max_time is self.time_steps as a int
                data_t2_hat = self.model(data_t1, t_2-t_1)
                # data_1 = data_1.to(device='cuda')
                # data_syn_t2 = self.model(data_1, self.max_time)

                imgs_to_plot = [
                    # imgs, title
                    [data_1, 'oriMR'],
                    [data_2, 'pseMR'],
                    # [data_syn_t2, 'synCT'],
                    # include the time step t_1_int in the title
                    [data_t1, f'data_t1_{int(t_1_int)}'],
                    [data_t2, f'data_t2_{int(t_2_int)}'],
                    [data_t2_hat, 'data_t2_hat'],
                ]

                # iteratively plot
                for imgs, title in imgs_to_plot:
                    imgs = imgs.detach().cpu()
                    utils.save_image(imgs, str(self.results_folder / f'{title}-{milestone}.png'), nrow=4)
                    del imgs

                gc.collect()

                acc_loss = acc_loss/(self.save_and_sample_every+1)
                # experiment.log_metric("Training Loss", acc_loss, step=self.step)
                print(f'Mean of last {self.step}: {acc_loss}')
                # save the loss as a json file
                with open(str(self.results_folder / f'loss-{milestone}.json'), 'w') as f:
                    json.dump(self.loss_recorder, f)
                acc_loss=0

                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')

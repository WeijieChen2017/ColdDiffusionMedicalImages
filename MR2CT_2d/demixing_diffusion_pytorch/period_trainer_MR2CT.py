# from comet_ml import Expeiment
import gc
import os
import re
import copy
import torch
from torch import nn
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

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
from skimage.metrics import structural_similarity as ssim

# trainer class

class period_trainer_MR2CT(object):
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
        end_to_end = False,
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
        self.load_path = load_path

        if load_path != None:
            self.load(load_path)

        self.time_steps = time_steps
        self.loss_type = loss_type
        self.loss_fn = L1Loss() if loss_type == 'l1' else MSELoss()
        self.dataloader = data_loader
        self.max_time = torch.tensor(self.time_steps, dtype=torch.float)
        self.max_time = self.max_time.expand(self.batch_size).to(device='cuda')
        self.end_to_end = end_to_end

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
        if self.end_to_end is False:
            t_2_int = torch.randint(3, self.time_steps, (1,))
            t_1_int = torch.randint(1, t_2_int-1, (1,))
            assert t_1_int < t_2_int
        else:
            t_2_int = torch.tensor(self.time_steps, dtype=torch.int)
            t_1_int = torch.tensor(0, dtype=torch.int)

        # make both t_1 and t_2 of the shape of batch_size
        t_1 = t_1_int.expand(data_1.shape[0])
        t_2 = t_2_int.expand(data_1.shape[0])

        alpha_1 = t_1_int.float() / self.time_steps
        alpha_2 = t_2_int.float() / self.time_steps

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
                
                data_t1, data_t2, t_1, t_2, _, _ = self.generate_xt1_xt2(data_1, data_2, device='cuda')

                data_t2_hat = self.model(data_t1, t_2-t_1)
                loss = self.loss_fn(data_t2_hat, data_t2)

                if self.step % 100 == 0:
                    print(f'{self.step}: {loss.item()}')
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
                data_t1, data_t2, t_1, t_2, t_1_int, t_2_int = self.generate_xt1_xt2(data_1, data_2, device='cuda')
                # create max_time as a tensor of shape batch_size
                # given that the max_time is self.time_steps as a int
                data_t2_hat = self.model(data_t1, t_2-t_1)
                # data_1 = data_1.to(device='cuda')
                # data_syn_t2 = self.model(data_1, self.max_time)

                imgs_to_plot = [
                    # imgs, title
                    [data_1, 'MR'],
                    [data_2, 'CT'],
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
                acc_loss=0

                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')

    def eval_jumps(self, n_jumps, time_steps, n_test=-1, save_folder="./results/"):
        self.model.eval()
        self.model = self.model.to(device='cuda')
        # self.ema_model.eval()

        HU_error = []
        dice_bone = []
        PSNR = []
        SSIM = []
        # get the checkpoint name
        numbers = re.findall(r'\d+', self.load_path.split('/')[-1].split('.')[0])

        # Extract the first sequence of digits and convert it to an integer
        # then format it with a 'K' to denote thousands
        num_with_k = f"{int(numbers[0])//1000}K" if numbers else None
        save_path = save_folder + f'/pt{num_with_k}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = save_folder + f'/{n_jumps}_jumps/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        


        # start from 1, end at time_steps, step size = int(time_steps/n_jumps)
        s_step = [int(time_steps/n_jumps) for i in range(1, n_jumps+1)]
        # last step should be time_steps
        s_step[-1] = time_steps - sum(s_step[:-1]) - 1
        print("s_step: ", s_step)

        for batch_idx, data in enumerate(self.dataloader):
            curr_step = 1
            img1, img2 = data
            img1 = img1.to(device='cuda')
            curr_img = img1

            # plot imgs and save
            # left to right: img1, all img2_hat, img2
            # the width of output img is 4*(n_jumps+2)
            n_plot = n_jumps + 2
            plot_width = 4*n_plot
            plt.figure(figsize=(plot_width, 4), dpi=300)

            plt.subplot(1, n_plot, 1)
            plot_1 = img1[0,0,:,:].detach().cpu()
            plot_1 = np.rot90(plot_1)
            plt.imshow(plot_1, cmap='gray')
            plt.title('MR')
            plt.axis('off')

            for i in range(n_jumps):
                t = torch.tensor(s_step[i], dtype=torch.float)
                t = t.expand(1).to(device='cuda')
                curr_img2_hat = self.model(curr_img, t)
                curr_step += s_step[i]
                curr_img2_hat = curr_img2_hat.detach().cpu()

                plt.subplot(1, n_plot, i+2)
                plot_2 = curr_img2_hat[0,0,:,:]
                plot_2 = np.rot90(plot_2)
                plt.imshow(plot_2, cmap='gray')
                plt.title(f'step = {curr_step}')
                plt.axis('off')

                curr_img = curr_img2_hat.to(device='cuda')

            # we need to compute the error between img2_hat[-1] and img2
            # original is 0-3000, we divide it by 4024
            # gt = (img2 * 4024 - 1024) / 4024
            gt = np.squeeze(img2[:, 1, :, :]) * 4024
            gt = gt.detach().cpu().numpy()
            gt = np.clip(gt, -1024, 3000)
            pred = np.squeeze(curr_img2_hat[:, 1, :, :])*4024
            pred = pred.detach().cpu().numpy()
            # print(gt.size(), pred.size())
            # compute the HU error i.e. MAE
            HU_error.append(np.mean(np.abs(gt - pred)))
            # HU_error.append(torch.mean(torch.abs(gt - pred)))
            print(f'batch_idx: {batch_idx}, HU_error: {HU_error[-1]}')

            # compute the dice score
            ground_truth = gt > 500
            prediction = pred > 500
            ground_truth = ground_truth.astype(bool)
            prediction = prediction.astype(bool)
            intersection = np.logical_and(ground_truth, prediction)
            curr_dice = 2. * intersection.sum() / (ground_truth.sum() + prediction.sum())
            dice_bone.append(curr_dice)
            
            # compute the PSNR
            mse = np.mean((gt - pred)**2)
            PSNR.append(20*np.log10(3000/np.sqrt(mse)))

            # compute the SSIM
            SSIM.append(ssim(gt, pred, data_range=4024))

            plt.subplot(1, n_plot, n_plot)
            plot_3 = img2[0,0,:,:]
            plot_3 = np.rot90(plot_3)
            plt.imshow(plot_3, cmap='gray')
            plt.title('CT')
            plt.axis('off')

            plt.savefig(save_path+f'eval_{n_jumps}_jumps_{batch_idx}.png')
            plt.close()

            if batch_idx == n_test:
                break

        # save the metrics
        metrics = {}
        metrics['HU_error'] = HU_error
        metrics['dice_bone'] = dice_bone
        metrics['PSNR'] = PSNR
        metrics['SSIM'] = SSIM
        np.save(save_path+f'metric_{n_jumps}_jumps_{n_test}_test.npy', metrics)
        print("metrics saved to: ", save_path+f'metric_{n_jumps}_jumps_{n_test}_test.npy')


             

            
        
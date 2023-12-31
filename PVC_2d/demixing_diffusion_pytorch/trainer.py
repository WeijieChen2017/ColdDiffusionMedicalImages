# from comet_ml import Expeiment
import copy
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import utils

from sklearn.mixture import GaussianMixture
#from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import torch

from .network import EMA
from .dataset import DatasetPaired_Aug, DatasetPaired
from .utils import cycle, loss_backwards, create_folder

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder1,
        # folder2,
        *,
        ema_decay = 0.995,
        image_size = 128,
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
        dataset = None,
        shuffle=True
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        if dataset == 'train':
            print(dataset, "DA used")
            self.ds1 = DatasetPaired_Aug(folder1, image_size)
            # self.ds2 = DatasetPaired_Aug(folder2, image_size)
        else:
            print(dataset)
            self.ds1 = DatasetPaired(folder1, image_size)
            # self.ds2 = DatasetPaired(folder2, image_size)

        self.dl1 = cycle(data.DataLoader(self.ds1, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=16, drop_last=True))
        # self.dl2 = cycle(data.DataLoader(self.ds2, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=16, drop_last=True))

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.step = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.fp16 = fp16

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


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


    def add_title(self, path, title):

        import cv2
        import numpy as np

        img1 = cv2.imread(path)

        # --- Here I am creating the border---
        black = [0, 0, 0]  # ---Color of the border---
        constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
        height = 20
        violet = np.zeros((height, constant.shape[1], 3), np.uint8)
        violet[:] = (255, 0, 180)

        vcat = cv2.vconcat((violet, constant))

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(vcat, str(title), (violet.shape[1] // 2, height-2), font, 0.5, (0, 0, 0), 1, 0)
        cv2.imwrite(path, vcat)



    def train(self):
        # experiment = Experiment(api_key="57ArytWuo2X4cdDmgU1jxin77",
        #                         project_name="Cold_Diffusion_Cycle")

        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        while self.step < self.train_num_steps:
            u_loss = 0
            for i in range(self.gradient_accumulate_every):
                data_1, data_2 = next(self.dl1)
                data_1.cuda()
                data_2.cuda()
                # data_2 = next(self.dl2).cuda()
                loss = torch.mean(self.model(data_1, data_2))
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
                batches = self.batch_size
                img_1, img_2 = next(self.dl1)
                # img_2.cuda()
                # og_img = next(self.dl2).cuda()
                og_img = img_1.cuda()
                xt_img = img_2.cuda()

                xt, direct_recons, all_images = self.ema_model.module.sample(batch_size=batches, img=og_img)

                og_img = (og_img + 1) * 0.5
                utils.save_image(og_img, str(self.results_folder / f'sample-og-{milestone}.png'), nrow=6)

                all_images = (all_images + 1) * 0.5
                utils.save_image(all_images, str(self.results_folder / f'sample-recon-{milestone}.png'), nrow = 6)

                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'sample-direct_recons-{milestone}.png'), nrow=6)

                # xt = (xt + 1) * 0.5
                utils.save_image(xt_img, str(self.results_folder / f'sample-xt-{milestone}.png'), nrow=6)

                acc_loss = acc_loss/(self.save_and_sample_every+1)
                # experiment.log_metric("Training Loss", acc_loss, step=self.step)
                print(f'Mean of last {self.step}: {acc_loss}')
                acc_loss=0

                self.save()
                if self.step % (self.save_and_sample_every * 100) == 0:
                    self.save(self.step)

            self.step += 1

        print('training completed')

    def test_from_data(self, extra_path, s_times=None):
        batches = self.batch_size
        img_1, img_2 = next(self.dl1)
        # img_2.cuda()
        # og_img = next(self.dl2).cuda()
        og_img = img_1.cuda()
        # og_img = next(self.dl2).cuda()

        X_0s, X_ts = self.ema_model.module.all_sample(batch_size=batches, img=og_img, times=s_times)

        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-x0.png')))

            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images, str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(imageio.imread(str(self.results_folder / f'sample-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{extra_path}-xt.gif'), frames_t)


    def sample_and_save_for_fid(self, noise=0):

        # xt_folder = f'{self.results_folder}_xt'
        # create_folder(xt_folder)

        out_folder = f'{self.results_folder}_out'
        create_folder(out_folder)

        # direct_recons_folder = f'{self.results_folder}_dir_recons'
        # create_folder(direct_recons_folder)

        cnt = 0
        bs = self.batch_size
        for j in range(int(6400/self.batch_size)):
            img_1, img_2 = next(self.dl1)
            # img_2.cuda()
                # og_img = next(self.dl2).cuda()
            og_img = img_1.cuda()
            # og_img = next(self.dl2).cuda()
            print(og_img.shape)

            xt, direct_recons, all_images = self.ema_model.module.gen_sample(batch_size=bs, img=og_img,
                                                                             noise_level=noise)

            for i in range(all_images.shape[0]):
                utils.save_image((all_images[i] + 1) * 0.5,
                                 str(f'{out_folder}/' + f'sample-x0-{cnt}.png'))

                # utils.save_image((xt[i] + 1) * 0.5,
                #                  str(f'{xt_folder}/' + f'sample-x0-{cnt}.png'))
                #
                # utils.save_image((direct_recons[i] + 1) * 0.5,
                #                  str(f'{direct_recons_folder}/' + f'sample-x0-{cnt}.png'))

                cnt += 1

    def paper_showing_diffusion_images_cover_page(self):

        import cv2
        cnt = 0

        #to_show = [2, 4, 8, 16, 32, 64, 128, 192]
        to_show = [2, 4, 16, 64, 128, 256, 384, 448, 480]

        for i in range(5):
            batches = self.batch_size
            og_img_1, og_img_2 = next(self.dl1)
            og_img_1.cuda()
            og_img_2.cuda()
            # og_img_1 = next(self.dl1).cuda()
            # og_img_2 = next(self.dl2).cuda()
            print(og_img_1.shape)

            Forward, Backward, final_all = self.ema_model.module.forward_and_backward(batch_size=batches, img1=og_img_1, img2=og_img_2)

            og_img_1 = (og_img_1 + 1) * 0.5
            og_img_2 = (og_img_2 + 1) * 0.5
            final_all = (final_all + 1) * 0.5



            for k in range(Forward[0].shape[0]):
                print(k)
                l = []

                utils.save_image(og_img_1[k], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)
                start = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')
                l.append(start)

                for j in range(len(Forward)):
                    x_t = Forward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if j in to_show:
                        l.append(x_t)

                for j in range(len(Backward)):
                    x_t = Backward[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'temp.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/temp.png')
                    if (len(Backward) - j) in to_show:
                        l.append(x_t)


                utils.save_image(final_all[k], str(self.results_folder / f'final_{cnt}.png'), nrow=1)
                final = cv2.imread(f'{self.results_folder}/final_{cnt}.png')
                l.append(final)


                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1

            del Forward
            del Backward
            del final_all

    def paper_invert_section_images(self, s_times=None):

        cnt = 0
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for j in range(og_img.shape[0]//3):
                original = og_img[j: j + 1]
                utils.save_image(original, str(self.results_folder / f'original_{cnt}.png'), nrow=3)

                direct_recons = X_0s[0][j: j + 1]
                direct_recons = (direct_recons + 1) * 0.5
                utils.save_image(direct_recons, str(self.results_folder / f'direct_recons_{cnt}.png'), nrow=3)

                sampling_recons = X_0s[-1][j: j + 1]
                sampling_recons = (sampling_recons + 1) * 0.5
                utils.save_image(sampling_recons, str(self.results_folder / f'sampling_recons_{cnt}.png'), nrow=3)

                blurry_image = X_ts[0][j: j + 1]
                blurry_image = (blurry_image + 1) * 0.5
                utils.save_image(blurry_image, str(self.results_folder / f'blurry_image_{cnt}.png'), nrow=3)



                import cv2

                blurry_image = cv2.imread(f'{self.results_folder}/blurry_image_{cnt}.png')
                direct_recons = cv2.imread(f'{self.results_folder}/direct_recons_{cnt}.png')
                sampling_recons = cv2.imread(f'{self.results_folder}/sampling_recons_{cnt}.png')
                original = cv2.imread(f'{self.results_folder}/original_{cnt}.png')

                black = [0, 0, 0]
                blurry_image = cv2.copyMakeBorder(blurry_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                direct_recons = cv2.copyMakeBorder(direct_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                sampling_recons = cv2.copyMakeBorder(sampling_recons, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                original = cv2.copyMakeBorder(original, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([blurry_image, direct_recons, sampling_recons, original])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def paper_showing_diffusion_images(self, s_times=None):

        import cv2
        cnt = 0
        # to_show = [0, 1, 2, 4, 8, 10, 12, 16, 17, 18, 19, 20]
        # to_show = [0, 1, 2, 4, 8, 16, 20, 24, 32, 36, 38, 39, 40]
        # to_show = [0, 1, 2, 4, 8, 16, 24, 32, 40, 44, 46, 48, 49]
        to_show = [0, 2, 4, 8, 16, 32, 64, 80, 88, 92, 96, 98, 99]

        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s, X_ts = self.ema_model.all_sample(batch_size=batches, img=og_img, times=s_times)
            og_img = (og_img + 1) * 0.5

            for k in range(X_ts[0].shape[0]):
                l = []

                for j in range(len(X_ts)):
                    x_t = X_ts[j][k]
                    x_t = (x_t + 1) * 0.5
                    utils.save_image(x_t, str(self.results_folder / f'x_{len(X_ts)-j}_{cnt}.png'), nrow=1)
                    x_t = cv2.imread(f'{self.results_folder}/x_{len(X_ts)-j}_{cnt}.png')
                    if j in to_show:
                        l.append(x_t)


                x_0 = X_0s[-1][k]
                x_0 = (x_0 + 1) * 0.5
                utils.save_image(x_0, str(self.results_folder / f'x_best_{cnt}.png'), nrow=1)
                x_0 = cv2.imread(f'{self.results_folder}/x_best_{cnt}.png')
                l.append(x_0)
                im_h = cv2.hconcat(l)
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt+=1


    def paper_showing_diffusion_images_diff(self, s_times=None):

        import cv2
        for i in range(50):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s_alg2, X_ts_alg2 = self.ema_model.all_sample_both_sample(sampling_routine='x0_step_down', batch_size=batches,
                                                                 img=og_img, times=s_times)
            X_0s_alg1, X_ts_alg1 = self.ema_model.all_sample_both_sample(sampling_routine='default', batch_size=batches,
                                                                 img=og_img, times=s_times)

            og_img = (og_img + 1) * 0.5

            alg2 = []
            alg1 = []

            #to_show = [0, 1, 2, 4, 8, 16, 20, 24, 32, 36, 38, 39, 40]
            to_show = [0, 1, 2, 4, 8, 10, 12, 16, 17, 18, 19, 20]

            for j in range(len(X_ts_alg2)):
                x_t = X_ts_alg2[j][0]
                x_t = (x_t + 1) * 0.5
                utils.save_image(x_t, str(self.results_folder / f'x_alg2_{len(X_ts_alg2)-j}_{i}.png'), nrow=1)
                x_t = cv2.imread(f'{self.results_folder}/x_alg2_{len(X_ts_alg2)-j}_{i}.png')
                if j in to_show:
                    alg2.append(x_t)

                x_t = X_ts_alg1[j][0]
                x_t = (x_t + 1) * 0.5
                utils.save_image(x_t, str(self.results_folder / f'x_alg2_{len(X_ts_alg1) - j}_{i}.png'), nrow=1)
                x_t = cv2.imread(f'{self.results_folder}/x_alg2_{len(X_ts_alg1) - j}_{i}.png')
                if j in to_show:
                    alg1.append(x_t)


            x_0 = X_0s_alg2[-1][0]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'x_best_alg2_{i}.png'), nrow=1)
            x_0 = cv2.imread(f'{self.results_folder}/x_best_alg2_{i}.png')
            alg2.append(x_0)
            im_h = cv2.hconcat(alg2)
            cv2.imwrite(f'{self.results_folder}/all_alg2_{i}.png', im_h)

            x_0 = X_0s_alg1[-1][0]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'x_best_alg1_{i}.png'), nrow=1)
            x_0 = cv2.imread(f'{self.results_folder}/x_best_alg1_{i}.png')
            alg1.append(x_0)
            im_h = cv2.hconcat(alg1)
            cv2.imwrite(f'{self.results_folder}/all_alg1_{i}.png', im_h)


    def paper_showing_sampling_diff_images(self, s_times=None):

        import cv2
        cnt = 0
        for i in range(10):
            batches = self.batch_size
            og_img = next(self.dl).cuda()
            print(og_img.shape)

            X_0s_alg2, _ = self.ema_model.all_sample_both_sample(sampling_routine='x0_step_down', batch_size=batches, img=og_img, times=s_times)
            X_0s_alg1, _ = self.ema_model.all_sample_both_sample(sampling_routine='default', batch_size=batches,
                                                                 img=og_img, times=s_times)

            x0_alg1 = (X_0s_alg1[-1] + 1) * 0.5
            x0_alg2 = (X_0s_alg2[-1] + 1) * 0.5
            og_img = (og_img + 1) * 0.5


            for j in range(og_img.shape[0]):
                utils.save_image(x0_alg1[j], str(self.results_folder / f'x0_alg1_{cnt}.png'), nrow=1)
                utils.save_image(x0_alg2[j], str(self.results_folder / f'x0_alg2_{cnt}.png'), nrow=1)
                utils.save_image(og_img[j], str(self.results_folder / f'og_img_{cnt}.png'), nrow=1)



                alg1 = cv2.imread(f'{self.results_folder}/x0_alg1_{cnt}.png')
                alg2 = cv2.imread(f'{self.results_folder}/x0_alg2_{cnt}.png')
                og = cv2.imread(f'{self.results_folder}/og_img_{cnt}.png')


                black = [255, 255, 255]
                alg1 = cv2.copyMakeBorder(alg1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                alg2 = cv2.copyMakeBorder(alg2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
                og = cv2.copyMakeBorder(og, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)

                im_h = cv2.hconcat([og, alg1, alg2])
                cv2.imwrite(f'{self.results_folder}/all_{cnt}.png', im_h)

                cnt += 1

    def sample_as_a_vector_gmm(self, start=0, end=1000, siz=64, ch=3, clusters=10):

        all_samples = []
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img)
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        all_samples = all_samples.cpu().detach().numpy()

        num_samples = 100

        gm = GaussianMixture(n_components=clusters, random_state=0).fit(all_samples)
        og_x, og_y = gm.sample(n_samples=num_samples)
        og_x = og_x.reshape(num_samples, ch, siz, siz)
        og_x = torch.from_numpy(og_x).cuda()
        og_x = og_x.type(torch.cuda.FloatTensor)
        print(og_x.shape)
        og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')


        X_0s, X_ts = self.ema_model.all_sample(batch_size=1, img=og_img, times=None)

        extra_path = 'vec'
        og_img = (og_img + 1) * 0.5
        utils.save_image(og_img, str(self.results_folder / f'og-{start}-{end}-{siz}-{clusters}-{extra_path}.png'), nrow=6)

        import imageio
        frames_t = []
        frames_0 = []

        for i in range(len(X_0s)):
            print(i)

            x_0 = X_0s[i]
            x_0 = (x_0 + 1) * 0.5
            utils.save_image(x_0, str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png'),
                             nrow=6)
            self.add_title(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png'), str(i))
            frames_0.append(
                imageio.imread(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-x0.png')))

            x_t = X_ts[i]
            all_images = (x_t + 1) * 0.5
            utils.save_image(all_images,
                             str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png'), nrow=6)
            self.add_title(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png'), str(i))
            frames_t.append(
                imageio.imread(str(self.results_folder / f'sample-{start}-{end}-{siz}-{clusters}-{i}-{extra_path}-xt.png')))

        imageio.mimsave(str(self.results_folder / f'Gif-{start}-{end}-{siz}-{clusters}-{extra_path}-x0.gif'), frames_0)
        imageio.mimsave(str(self.results_folder / f'Gif-{start}-{end}-{siz}-{clusters}-{extra_path}-xt.gif'), frames_t)


    def sample_as_a_vector_gmm_and_save(self, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):

        all_samples = []
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img)
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        all_samples = all_samples.cpu().detach().numpy()
        gm = GaussianMixture(n_components=clusters, random_state=0).fit(all_samples)

        all_num = n_sample
        num_samples = 10000
        it = int(all_num/num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}/')

        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            og_x, og_y = gm.sample(n_samples=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)
            og_x = torch.from_numpy(og_x).cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample(batch_size=1, img=og_img, times=None)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}/' + f'sample-x0-{cnt}.png'))
                cnt += 1

            it = it - 1
            print(it)


    def sample_as_a_vector_pytorch_gmm_and_save(self, torch_gmm, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):


        flatten = nn.Flatten()
        dataset = self.ds
        all_samples = None

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img).cuda()
            if idx > start:
                if all_samples is None:
                    all_samples = img
                else:
                    all_samples = torch.cat((all_samples, img), dim=0)

            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break


        #all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        model = torch_gmm(num_components=clusters, trainer_params=dict(gpus=1), covariance_type='full',
                          convergence_tolerance=0.001, batch_size=1000)
        model.fit(all_samples)

        all_num = n_sample
        num_samples = 100
        it = int(all_num / num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}/')
        create_folder(f'{self.results_folder}_gmm_{siz}_{clusters}/')
        create_folder(f'{self.results_folder}_gmm_blur_{siz}_{clusters}/')


        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            #og_x, _ = model.sample(n=num_samples)
            og_x = model.sample(num_datapoints=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)

            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            x0s = X_0s[-1]
            blurs = X_ts[0]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((og_img[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_{siz}_{clusters}/' + f'sample-{cnt}.png'))

                utils.save_image((blurs[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_blur_{siz}_{clusters}/' + f'sample-blur-{cnt}.png'))

                cnt += 1

            it = it - 1
            print(it)

    def sample_as_a_vector_from_blur_pytorch_gmm_and_save(self, torch_gmm, start=0, end=1000, siz=64, ch=3, clusters=10, n_sample=10000):
        flatten = nn.Flatten()
        dataset = self.ds

        print(len(dataset))

        #sample_at = self.ema_model.num_timesteps // 2
        sample_at = self.ema_model.num_timesteps // 2
        all_samples = None

        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0)
            img = self.ema_model.opt(img.cuda(), t=sample_at)
            img = F.interpolate(img, size=siz, mode='bilinear')
            img = flatten(img).cuda()

            if idx > start:
                if all_samples is None:
                    all_samples = img
                else:
                    all_samples = torch.cat((all_samples, img), dim=0)
                #all_samples.append(img[0])

            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        # all_samples = torch.stack(all_samples)
        print(all_samples.shape)

        model = torch_gmm(num_components=clusters, trainer_params=dict(gpus=1), covariance_type='full',
                          convergence_tolerance=0.001, batch_size=1000)
        model.fit(all_samples)



        all_num = n_sample
        num_samples = 100
        it = int(all_num/num_samples)

        create_folder(f'{self.results_folder}_{siz}_{clusters}_{sample_at}/')
        create_folder(f'{self.results_folder}_gmm_{siz}_{clusters}_{sample_at}/')
        create_folder(f'{self.results_folder}_gmm_blur_{siz}_{clusters}_{sample_at}/')

        print(f'{self.results_folder}_{siz}_{clusters}/')

        cnt=0
        while(it):
            og_x = model.sample(num_datapoints=num_samples)
            og_x = og_x.reshape(num_samples, ch, siz, siz)

            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            print(og_x.shape)
            og_img = F.interpolate(og_x, size=self.image_size, mode='bilinear')
            X_0s, X_ts = self.ema_model.all_sample_from_blur(batch_size=og_img.shape[0], img=og_img, start_times=sample_at)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image((x0s[i]+1)*0.5, str(f'{self.results_folder}_{siz}_{clusters}_{sample_at}/' + f'sample-x0-{cnt}.png'))

                utils.save_image((og_img[i] + 1) * 0.5,
                                 str(f'{self.results_folder}_gmm_{siz}_{clusters}_{sample_at}/' + f'sample-{cnt}.png'))

                cnt += 1

            it = it - 1



    def sample_from_data_save(self, start=0, end=1000):

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        create_folder(f'{self.results_folder}/')

        cnt=0
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + 1000]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            x0s = X_0s[-1]
            for i in range(x0s.shape[0]):
                utils.save_image( (x0s[i]+1)*0.5, str(f'{self.results_folder}/' + f'sample-x0-{cnt}.png'))
                cnt += 1

    def fid_distance_decrease_from_manifold(self, fid_func, start=0, end=1000):

        all_samples = []
        dataset = self.ds

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = torch.unsqueeze(img, 0).cuda()
            if idx > start:
                all_samples.append(img[0])
            if idx % 1000 == 0:
                print(idx)
            if end != None:
                if idx == end:
                    print(idx)
                    break

        all_samples = torch.stack(all_samples)
        # create_folder(f'{self.results_folder}/')
        blurred_samples = None
        original_sample = None
        deblurred_samples = None
        direct_deblurred_samples = None

        sanity_check = 1


        cnt=0
        while(cnt < all_samples.shape[0]):
            og_x = all_samples[cnt: cnt + 100]
            og_x = og_x.cuda()
            og_x = og_x.type(torch.cuda.FloatTensor)
            og_img = og_x
            print(og_img.shape)
            X_0s, X_ts = self.ema_model.all_sample(batch_size=og_img.shape[0], img=og_img, times=None)

            og_img = og_img.to('cpu')
            blurry_imgs = X_ts[0].to('cpu')
            deblurry_imgs = X_0s[-1].to('cpu')
            direct_deblurry_imgs = X_0s[0].to('cpu')

            og_img = og_img.repeat(1, 3 // og_img.shape[1], 1, 1)
            blurry_imgs = blurry_imgs.repeat(1, 3 // blurry_imgs.shape[1], 1, 1)
            deblurry_imgs = deblurry_imgs.repeat(1, 3 // deblurry_imgs.shape[1], 1, 1)
            direct_deblurry_imgs = direct_deblurry_imgs.repeat(1, 3 // direct_deblurry_imgs.shape[1], 1, 1)



            og_img = (og_img + 1) * 0.5
            blurry_imgs = (blurry_imgs + 1) * 0.5
            deblurry_imgs = (deblurry_imgs + 1) * 0.5
            direct_deblurry_imgs = (direct_deblurry_imgs + 1) * 0.5

            if cnt == 0:
                print(og_img.shape)
                print(blurry_imgs.shape)
                print(deblurry_imgs.shape)
                print(direct_deblurry_imgs.shape)

                if sanity_check:
                    folder = './sanity_check/'
                    create_folder(folder)

                    san_imgs = og_img[0: 32]
                    utils.save_image(san_imgs,str(folder + f'sample-og.png'), nrow=6)

                    san_imgs = blurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-xt.png'), nrow=6)

                    san_imgs = deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-recons.png'), nrow=6)

                    san_imgs = direct_deblurry_imgs[0: 32]
                    utils.save_image(san_imgs, str(folder + f'sample-direct-recons.png'), nrow=6)


            if blurred_samples is None:
                blurred_samples = blurry_imgs
            else:
                blurred_samples = torch.cat((blurred_samples, blurry_imgs), dim=0)


            if original_sample is None:
                original_sample = og_img
            else:
                original_sample = torch.cat((original_sample, og_img), dim=0)


            if deblurred_samples is None:
                deblurred_samples = deblurry_imgs
            else:
                deblurred_samples = torch.cat((deblurred_samples, deblurry_imgs), dim=0)


            if direct_deblurred_samples is None:
                direct_deblurred_samples = direct_deblurry_imgs
            else:
                direct_deblurred_samples = torch.cat((direct_deblurred_samples, direct_deblurry_imgs), dim=0)

            cnt += og_img.shape[0]

        print(blurred_samples.shape)
        print(original_sample.shape)
        print(deblurred_samples.shape)
        print(direct_deblurred_samples.shape)

        fid_blur = fid_func(samples=[original_sample, blurred_samples])
        rmse_blur = torch.sqrt(torch.mean( (original_sample - blurred_samples)**2 ))
        ssim_blur = ssim(original_sample, blurred_samples, data_range=1, size_average=True)
        # n_og = original_sample.cpu().detach().numpy()
        # n_bs = blurred_samples.cpu().detach().numpy()
        # ssim_blur = ssim(n_og, n_bs, data_range=n_og.max() - n_og.min(), multichannel=True)
        print(f'The FID of blurry images with original image is {fid_blur}')
        print(f'The RMSE of blurry images with original image is {rmse_blur}')
        print(f'The SSIM of blurry images with original image is {ssim_blur}')


        fid_deblur = fid_func(samples=[original_sample, deblurred_samples])
        rmse_deblur = torch.sqrt(torch.mean((original_sample - deblurred_samples) ** 2))
        ssim_deblur = ssim(original_sample, deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of deblurred images with original image is {fid_deblur}')
        print(f'The RMSE of deblurred images with original image is {rmse_deblur}')
        print(f'The SSIM of deblurred images with original image is {ssim_deblur}')

        print(f'Hence the improvement in FID using sampling is {fid_blur - fid_deblur}')

        fid_direct_deblur = fid_func(samples=[original_sample, direct_deblurred_samples])
        rmse_direct_deblur = torch.sqrt(torch.mean((original_sample - direct_deblurred_samples) ** 2))
        ssim_direct_deblur = ssim(original_sample, direct_deblurred_samples, data_range=1, size_average=True)
        print(f'The FID of direct deblurred images with original image is {fid_direct_deblur}')
        print(f'The RMSE of direct deblurred images with original image is {rmse_direct_deblur}')
        print(f'The SSIM of direct deblurred images with original image is {ssim_direct_deblur}')

        print(f'Hence the improvement in FID using direct sampling is {fid_blur - fid_direct_deblur}')


            # x0s = X_0s[-1]
            # for i in range(x0s.shape[0]):
            #     utils.save_image( (x0s[i]+1)*0.5, str(f'{self.results_folder}/' + f'sample-x0-{cnt}.png'))
            #     cnt += 1

    def save_training_data(self):
        dataset = self.ds
        create_folder(f'{self.results_folder}/')

        print(len(dataset))
        for idx in range(len(dataset)):
            img = dataset[idx]
            img = (img + 1) * 0.5
            utils.save_image(img, str(f'{self.results_folder}/' + f'{idx}.png'))
            if idx%1000 == 0:
                print(idx)

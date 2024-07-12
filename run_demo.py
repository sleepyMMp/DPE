
import os
import cv2 
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import collections
from realesrgan import RealESRGANer
from GFPGAN.gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

def load_image1(filename, size):
    img = filename.convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0

def img_preprocessing(img_path, size):
    img = load_image1(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


def video2imgs(videoPath):
    cap = cv2.VideoCapture(videoPath)    
    judge = cap.isOpened()                
    fps = cap.get(cv2.CAP_PROP_FPS)     

    frames = 1                           
    count = 1                           
    img = []
    while judge:
        flag, frame = cap.read()         
        if not flag:
            break
        else:
           img.append(frame) 
    cap.release()

    return img


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        model_path = args.model_path
        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        model_en = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model_en,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
        model_name = 'GFPGANv1.4'
        model_path = os.path.join('./checkpoints', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('realesrgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise ValueError(f'Model {model_name} does not exist.')
        
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)

        print('==> loading data')
        self.save_path = args.output_folder
        os.makedirs(self.save_path, exist_ok=True)


        s_img = video2imgs(args.s_path)
        d_img = video2imgs(args.d_path)

        s = []
        for i in s_img:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            s.append(img_preprocessing(img,256).cuda())

        d = []
        for i in d_img:
            img = Image.fromarray(cv2.cvtColor(i,cv2.COLOR_BGR2RGB))
            d.append(img_preprocessing(img,256).cuda())

        self.s_img = s
        self.d_img = d
        self.run()


    def run(self):
    
        output_dir = self.save_path

        crop_vi = os.path.join(output_dir, 'edit.mp4')
        out_edit = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 's.mp4')
        out_s = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        crop_vi = os.path.join(output_dir, 'd.mp4')
        out_d = cv2.VideoWriter(crop_vi, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256,256))

        print('==> running')
        with torch.no_grad():
            l = min(len(self.d_img),len(self.s_img))
            for i in tqdm(range(l)):
                img_target = self.d_img[i]
                img_source = self.s_img[i]
                output_dict = self.gen(img_source, img_target, args.face)
                fake = output_dict
                fake = fake.cpu().clamp(-1, 1)

                video_numpy = fake[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)

                if self.args.EN:
                    
                    cropped_faces, restored_faces, video_numpy = self.restorer.enhance(
                        video_numpy,
                        has_aligned=False, only_center_face=True, paste_back=True)
                    
                  
                 
                out_edit.write(video_numpy)
                                            
         

                video_numpy = img_source[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)
                out_s.write(video_numpy)

                video_numpy = img_target[:,:3,:,:].clone().cpu().float().detach().numpy()
                video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
                video_numpy = video_numpy.astype(np.uint8)[0]
                video_numpy = cv2.cvtColor(video_numpy, cv2.COLOR_RGB2BGR)
                out_d.write(video_numpy)

            out_edit.release()
            out_s.release()
            out_d.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--s_path", type=str, default='')
    parser.add_argument("--d_path", type=str, default='')
    parser.add_argument("--face", type=str, default='exp')
    parser.add_argument("--model_path", type=str, default='')
    parser.add_argument("--output_folder", type=str, default='')
    parser.add_argument("--EN", action="store_true", help="can enhance the result")
    args = parser.parse_args()

    # demo
    demo = Demo(args)

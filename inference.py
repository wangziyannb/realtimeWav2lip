import argparse
import math
import os
import wave

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
# import platform
# import subprocess
from torch.profiler import profile, record_function, ProfilerActivity
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, ao
from torch.ao.quantization import quantize_dynamic, default_weight_fake_quant, default_weight_only_qconfig, \
    default_fake_quant
from torchvision.models import inception_v3
from tqdm import tqdm
import openvino as ov

import audio
from FID import getFID
# from face_detect import face_rect
from models import Wav2Lip

from batch_face import RetinaFace
from time import time, sleep

import pyaudio
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
# import tkinter as tk
from PIL import Image, ImageTk

from models.conv import Conv2d, nonorm_Conv2d, Conv2dTranspose
from models.wav2lip import QuantizedWav2Lip
from realesrgan import RealESRGANer

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, default="./checkpoints/wav2lip_gan.pth",
                    help='Name of saved checkpoint to load weights from', required=False)


parser.add_argument('--face', type=str, default="Elon_Musk.jpg",
                    help='Filepath of video/image that contains faces to use', required=False)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=False)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                    default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=15., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=8)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--out_height', default=480, type=int,
                    help='Output video height. Best results are obtained at 480 or 720')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                         'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
parser.add_argument(
    '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
parser.add_argument(
    '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
parser.add_argument('-s', '--outscale', type=float, default=3.5, help='The final upsampling scale of the image')
# parser.add_argument('--sr_path', type=str, default="./checkpoints/RealESRGAN_x4plus.pth",
#                     help='Name of saved checkpoint to load super resolution weights from', required=False)
parser.add_argument('--sr_path', type=str, default="./checkpoints/RealESRGAN_x4plus.pth",
                    help='Name of saved checkpoint to load super resolution weights from', required=False)

class Wav2LipInference:

    def __init__(self, args) -> None:

        # self.CHUNK = 1024  # piece of audio data, no of frames per buffer during audio capture, large chunk size reduces computational overhead but may add latency and vise versa
        self.CHUNK = 1024+2048+4096
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1  # no of audio channels, 1 means monaural audio
        self.RATE = 16000  # sample rate of the audio stream, 16000 samples/second
        self.RECORD_SECONDS = 0.5  # time for which we capture the audio
        self.mel_step_size = 16  # mel freq step size
        self.audio_fs = 16000  # Sample rate
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'mps'
        self.args = args

        print('Using {} for inference.'.format(self.device))
        # torch.backends.quantized.engine = 'x86'
        torch.backends.quantized.engine = 'qnnpack'
        self.model = self.load_model()
        self.FIDModel = inception_v3(pretrained=True, transform_input=False)
        self.FIDModel.fc = torch.nn.Identity()

        self.model_prepared = torch.quantization.prepare(self.model)
        self.model_quantized = None
        self.detector = self.load_batch_face_model()
        self.sr_model = self.load_realesrgan()
        self.face_detect_cache_result = None
        self.img_tk = None
        self.total_img_batch = []
        self.total_mel_batch = []
        self.pred = []
        self.pred_q = []

    def load_realesrgan(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        #
        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        # netscale = 2
        # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']

        if self.args.sr_path is not None:
            model_path = self.args.sr_path
        else:
            model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        dni_weight = None
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.args.tile,
            tile_pad=self.args.tile_pad,
            pre_pad=self.args.pre_pad,
            half=False,
            # device=self.device)
            # device='cuda')
            device='mps')
        # if self.args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=self.args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=self.device)
            # device='cuda')
            # device='mps')
        return face_enhancer

    def load_wav2lip_openvino_model(self):
        '''
        func to load open vino model
        for wav2lip
        '''

        print("Calling wav2lip openvino model for inference...")
        core = ov.Core()
        devices = core.available_devices
        print(devices[0])
        model = core.read_model(model=os.path.join("./openvino_model/", "wav2lip_openvino_model.xml"))
        compiled_model = core.compile_model(model=model, device_name=devices[0])
        return compiled_model

    def load_model_weights(self, checkpoint_path):

        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_wav2lip_model(self, checkpoint_path):

        model = Wav2Lip()

        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = self.load_model_weights(checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model.eval()

        face_encoder = model.face_encoder_blocks
        audio_encoder = model.audio_encoder
        face_decoder = model.face_decoder_blocks
        output_block = model.output_block

        for e in face_encoder:
            if isinstance(e, nn.Sequential):
                for idx in range(len(e)):
                    if isinstance(e[idx], Conv2d):
                        if e[idx].residual:
                            torch.quantization.fuse_modules(e, [f'{idx}.conv_block.0',
                                                                f'{idx}.conv_block.1',
                                                                # f'{idx}.act'
                                                                ], inplace=True)
                        else:
                            torch.quantization.fuse_modules(e, [f'{idx}.conv_block.0',
                                                                f'{idx}.conv_block.1',
                                                                f'{idx}.act'
                                                                ], inplace=True)
        for idx in range(len(audio_encoder)):
            if isinstance(audio_encoder[idx], Conv2d):
                if audio_encoder[idx].residual:
                    torch.quantization.fuse_modules(audio_encoder, [f'{idx}.conv_block.0',
                                                                    f'{idx}.conv_block.1',
                                                                    # f'{idx}.act'
                                                                    ], inplace=True)
                else:
                    torch.quantization.fuse_modules(audio_encoder, [f'{idx}.conv_block.0',
                                                                    f'{idx}.conv_block.1',
                                                                    f'{idx}.act'
                                                                    ], inplace=True)
        # face_encoder.qconfig = None
        face_decoder.qconfig = None
        output_block.qconfig = None
        # for m in model.modules():
        #     if isinstance(m, nn.Sequential):
        #         for idx in range(len(m)):
        #             print(m[idx])
        #             # if isinstance(m[idx], Conv2d):
        #                 # if m[idx].residual:
        #                 #     torch.quantization.fuse_modules(m, [f'{idx}.conv_block.0',
        #                 #                                         f'{idx}.conv_block.1',
        #                 #                                         # f'{idx}.act'
        #                 #                                         ], inplace=True)
        #                 # else:
        #                 #     torch.quantization.fuse_modules(m, [f'{idx}.conv_block.0',
        #                 #                                         f'{idx}.conv_block.1',
        #                 #                                         f'{idx}.act'
        #                 #                                         ], inplace=True)
        #             # if isinstance(m[idx], Conv2dTranspose):
        #             #     m[idx].qconfig = ao.quantization.qconfig.QConfig(
        #             #         activation=ao.quantization.default_histogram_observer,
        #             #         weight=ao.quantization.default_weight_observer
        #             #     )
        # # model = QuantizedWav2Lip(model)
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        # model.qconfig = torch.quantization.get_default_qconfig('x86')
        model = model.to(self.device)
        return model

    def load_model(self):

        # if self.device=='cpu':
        #     return self.load_wav2lip_openvino_model()
        # else:
        return self.load_wav2lip_model(self.args.checkpoint_path)

    def load_batch_face_model(self):

        if self.device in ['mps', 'cpu']:
            return RetinaFace(gpu_id=-1, model_path="checkpoints/mobilenet.pth", network="mobilenet")
        else:
            return RetinaFace(gpu_id=0, model_path="checkpoints/mobilenet.pth", network="mobilenet")

    def face_rect(self, images):

        face_batch_size = 64 * 8
        num_batches = math.ceil(len(images) / face_batch_size)
        prev_ret = None
        for i in range(num_batches):
            batch = images[i * face_batch_size: (i + 1) * face_batch_size]
            all_faces = self.detector(batch)  # return faces list of all images
            for faces in all_faces:
                if faces:
                    box, landmarks, score = faces[0]
                    prev_ret = tuple(map(int, box))
                yield prev_ret

    def record_audio_stream(self, stream):

        stime = time()
        print("Recording audio ...")
        frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            frames.append(stream.read(self.CHUNK))  # Append audio data as numpy array

        print("Finished recording for curr time stamp ....")
        print("recording time, ", time() - stime)

        # audio_data = np.concatenate(frames)  # Combine all recorded frames into a single numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio_data

    def get_mel_chunks(self, audio_data):

        # Now you can perform mel chunk extraction directly on audio_data
        # Assuming you have functions audio.load_wav and audio.melspectrogram defined elsewhere in your code
        stime = time()
        # Example:
        wav = audio_data
        mel = audio.melspectrogram(wav)
        # print(mel.shape, time() - stime)

        # convert to mel chunks
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        stime = time()
        mel_chunks = []
        mel_idx_multiplier = 80. / self.args.fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))
        # print(time() - stime)

        return mel_chunks

    def get_smoothened_boxes(self, boxes, T):

        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self, images):

        results = []
        pady1, pady2, padx1, padx2 = self.args.pads

        s = time()

        for image, rect in zip(images, self.face_rect(images)):
            if rect is None:
                print("Face was not detected...")
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        print('face detect time:', time() - s)

        boxes = np.array(results)
        if not self.args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results

    def datagen(self, frames, mels):

        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.args.box[0] == -1:
            if not self.args.static:
                face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect_cache_result  # use cached result #face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):

            idx = 0 if self.args.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.args.img_size, self.args.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.args.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # if there are any other batches
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch


def update_frames(full_frames, stream, inference_pipline):
    stime = time()
    # convert recording to mel chunks
    audio_data = inference_pipline.record_audio_stream(stream)
    mel_chunks = inference_pipline.get_mel_chunks(audio_data)
    print(f"Time to process audio input {time() - stime}")

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = inference_pipline.args.wav2lip_batch_size
    gen = inference_pipline.datagen(full_frames.copy(), mel_chunks.copy())

    pred_quant = None
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(
                                                                        np.ceil(float(len(mel_chunks)) / batch_size)))):

        if inference_pipline.device == 'cpu':
            # start_time = time()
            # img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            # mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            # print(img_batch.shape, mel_batch.shape)
            # with torch.no_grad():
            #     pred = inference_pipline.model(mel_batch, img_batch).cpu().numpy()
            # fin_time = (time() - start_time)
            # print(f"CPU model inference time: {fin_time:.6f} seconds")

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            inference_pipline.total_img_batch.append(img_batch)
            inference_pipline.total_mel_batch.append(mel_batch)
            with torch.no_grad():
                if inference_pipline.model_quantized is None and len(inference_pipline.total_mel_batch) == 1:
                    # def calibrate(model, data_loader):
                    #     model.eval()
                    #     with torch.no_grad():
                    #         for data in data_loader:
                    #             model(data[0], data[1])
                    #
                    # calibration_loader = []
                    # for i in range(len(inference_pipline.total_mel_batch)):
                    #     calibration_loader.append(
                    #         (inference_pipline.total_mel_batch[i], inference_pipline.total_img_batch[i]))
                    # calibrate(inference_pipline.model_prepared, calibration_loader)

                    # model_quantized = torch.quantization.convert(inference_pipline.model_prepared)
                    model_quantized=inference_pipline.model_prepared
                    inference_pipline.model_quantized = model_quantized
                    start_time = time()
                    pred = inference_pipline.model_quantized(mel_batch, img_batch).numpy()
                    fin_time = (time() - start_time)
                    print(f"Quantized model inference time: {fin_time:.6f} seconds")

                    # traced_model = torch.jit.trace(model_quantized, (mel_batch, img_batch))
                    # torch.jit.save(traced_model, "quantized_wave2lip_model.pth")
                    # traced_model = torch.jit.trace(inference_pipline.model, (mel_batch, img_batch))
                    # torch.jit.save(traced_model, "origin_wave2lip_model.pth")
                # with profile(
                #         activities=[
                #             ProfilerActivity.CPU,
                #         ],
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #         record_shapes=True,
                #         profile_memory=True,
                #         with_stack=False
                # ) as prof:
                #     for x in range(1):
                #         # record_function
                #         inference_pipline.model_quantized(mel_batch, img_batch).numpy()
                #
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                #
                # with profile(
                #         activities=[
                #             ProfilerActivity.CPU,
                #         ],
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #         record_shapes=True,
                #         profile_memory=True,
                #         with_stack=False
                # ) as prof:
                #     for x in range(1):
                #         # record_function
                #         inference_pipline.model(mel_batch, img_batch).numpy()

                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                # elif inference_pipline.model_quantized is not None:
                #     # with profile(
                #     #         activities=[
                #     #             ProfilerActivity.CPU,
                #     #         ],
                #     #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #     #         record_shapes=True,
                #     #         profile_memory=True,
                #     #         with_stack=False
                #     # ) as prof:
                #     #     for x in range(1):
                #     #         # record_function
                #     #         inference_pipline.model_quantized(mel_batch, img_batch).numpy()
                #     #
                #     # print(prof.key_averages().table(sort_by="cpu_time_total"))
                #     #
                #     # with profile(
                #     #         activities=[
                #     #             ProfilerActivity.CPU,
                #     #         ],
                #     #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #     #         record_shapes=True,
                #     #         profile_memory=True,
                #     #         with_stack=False
                #     # ) as prof:
                #     #     for x in range(1):
                #     #         # record_function
                #     #         inference_pipline.model(mel_batch, img_batch).numpy()
                #     #
                #     # print(prof.key_averages().table(sort_by="cpu_time_total"))
                #     # start_time = time()
                #     # pred_quant = inference_pipline.model_quantized(mel_batch, img_batch).numpy()
                #     # fin_time = (time() - start_time)
                #     # print(f"Quantized model inference time: {fin_time:.6f} seconds")
                #
                #     start_time = time()
                #     pred = inference_pipline.model(mel_batch, img_batch).numpy()
                #     fin_time = (time() - start_time)
                #     print(f"CPU model inference time: {fin_time:.6f} seconds")
                #
                else:
                    start_time = time()
                    pred = inference_pipline.model(mel_batch, img_batch).numpy()
                    fin_time = (time() - start_time)
                    print(f"CPU wave2lip model inference time: {fin_time:.6f} seconds")
                # with profile(
                #         activities=[
                #             ProfilerActivity.CPU,
                #         ],
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #         record_shapes=True,
                #         profile_memory=True,
                #         with_stack=False
                # ) as prof:
                #     for x in range(1):
                #         # record_function
                #         model_quantized(mel_batch, img_batch).numpy()
                #
                # print(prof.key_averages().table(sort_by="cpu_time_total"))
                #
                # with profile(
                #         activities=[
                #             ProfilerActivity.CPU,
                #         ],
                #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                #         record_shapes=True,
                #         profile_memory=True,
                #         with_stack=False
                # ) as prof:
                #     for x in range(1):
                #         # record_function
                #         inference_pipline.model(mel_batch, img_batch).numpy()
                # #
                # print(prof.key_averages().table(sort_by="cpu_time_total"))

            # pred = inference_pipline.model([mel_batch, img_batch])['output']
            # fin_time = (time() - start_time)
            # print(f"OpenVino model inference time: {fin_time:.6f} seconds")
        else:
            start_time = time()
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(inference_pipline.device)
            with torch.no_grad():
                pred = inference_pipline.model(mel_batch, img_batch).cpu().numpy()
            fin_time = (time() - start_time)
            print(f"GPU wave2lip model inference time: {fin_time:.6f} seconds")

        if pred_quant is not None:
            pred_quant = pred_quant.transpose(0, 2, 3, 1) * 255.

        pred = pred.transpose(0, 2, 3, 1) * 255.
        # if pred_quant is not None:
        #     inference_pipline.pred.append(pred)
        #     inference_pipline.pred_q.append(pred_quant)
        #     if len(inference_pipline.pred_q) == 50:
        #         print(
        #             f"FID score:{getFID( np.concatenate(inference_pipline.pred, axis=0), np.concatenate(inference_pipline.pred_q, axis=0), inference_pipline.FIDModel,
        #                                 cuda=inference_pipline.device != 'cpu')}")
        # else:
        #     print(f"FID score:{getFID(pred, pred, inference_pipline.FIDModel)}")
        if pred_quant is not None:
            pred = pred_quant
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p

            # Convert frame to RGB format
            # frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            start_time = time()
            # a, b, output = inference_pipline.sr_model.enhance(f, has_aligned=False, only_center_face=False,
            #                                                   paste_back=True)
            output = f
            fin_time = (time() - start_time)
            print(f"CPU Real-ESRGAN & GFPGAN model inference time: {fin_time:.6f} seconds")
            # Encode the image to base64
            _, buffer = cv2.imencode('.jpg', output)
            buffer = np.array(buffer)
            buffer = buffer.tobytes()

            return (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')


def main(imagefilepath, flag):
    args = parser.parse_args()
    args.img_size = 96
    args.face = imagefilepath
    inference_pipline = Wav2LipInference(args)

    if os.path.isfile(args.face) and args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (int(args.out_height * aspect_ratio), args.out_height))
            # if args.resize_factor > 1:
            #     frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    p = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            print(f"Device {i}: {dev['name']}")
    stream = p.open(format=inference_pipline.FORMAT,
                    channels=inference_pipline.CHANNELS,
                    rate=inference_pipline.RATE,
                    input=True,
                    frames_per_buffer=inference_pipline.CHUNK)

    inference_pipline.face_detect_cache_result = inference_pipline.face_detect([full_frames[0]])

    while True:
        if not flag:
            stream.stop_stream()
            stream.close()
            p.terminate()
            return b""
        print(f"Model inference flag {flag}")

        start_time = time()
        yield update_frames(full_frames, stream, inference_pipline)
        fin_time = (time() - start_time)
        print(f"Total time for one chunk prediction: {fin_time:.6f} seconds")
        print()

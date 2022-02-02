import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import torchvision.models as models
import librosa
import pandas as pd
from torch._utils import _accumulate
from torch import default_generator, randperm
import random
from tqdm import tqdm
#import pickle
#import shutil
import subprocess
import logging
import argparse

logger = logging.getLogger('train_log')

class SoundDataset(Dataset):
    
    def __init__(self, dataset_path, is_train = True, use_cache = False, cutoff_freq=4000, cache_file="/content/sound_data.cache", save_cache_file="/content/sound_data.cache"):
    #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.file_path = []
        self.labels = []

        self.file_names, self.file_path, self.labels = self._getfile_list(dataset_path)
        self.idx_to_label, self.labels_idx = np.unique(self.labels, return_inverse=True)

        self.is_train = is_train

        self.data = []
        self.cutoff_freq = cutoff_freq

        self.use_cache = use_cache

        if self.use_cache == True:

            print("Loading dataset into memory..")
            for index in tqdm(range(len(self.file_path))):
                self.data.append(self._cache(index))
            
            # if os.path.isfile(cache_file):
            #     with open(cache_file, "rb") as file:
            #         print("Loading cache dataset : {}".format(cache_file))
            #         self.data = pickle.load(file)
            # else:
            #     print("Loading dataset into memory..")
            #     for index in tqdm(range(len(self.file_path))):
            #         self.data.append(self._cache(index))

            #     print("Saving cache to file.. {}".format(cache_file))
            #     with open(cache_file, "wb") as file:
            #         pickle.dump(self.data, file)    
            
        

    def _cache(self, index):    
           # n_fft = 1024
        # hop_length = 256
        # win_length = 1024
        n_fft = 512
        hop_length = 512
        win_length = 512
        n_mels = 224
        sample_rate = 22050
        cutoff_freq = self.cutoff_freq
        audio_length = 22050*5 # 5초
        
        path = self.file_path[index]
        waveform, sample = torchaudio.load(path)
        waveform = torchaudio.transforms.Resample(sample, sample_rate)(waveform)

        if self.is_train:
            self.transform = transform=transforms.Compose([
                    #transforms.Normalize(mean=[0.485], std=[0.229]), 
                    transforms.Resize((224,224)),
                    #transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transform=transforms.Compose([
                    # transforms.Normalize(mean=[0.485], std=[0.229]),
                    transforms.Resize((224,224))
            ])

        waveform = waveform[0,int(sample_rate/5):len(waveform) - int(sample_rate/5)]

        # 리밸런싱을 위해 추가했는데 크게 의미 없는듯?
        # waveform[waveform > 0.0015] = 0.0015
        # waveform[waveform < -0.002] = -0.002

        # 오디오의 길이를 맞춰주기위하여 길이가 짧은 경우 반복하여 붙여줌. 길면 짜름 (길이는 하이퍼파라마터로 조정 가능)
        # if len(waveform) < audio_length:
        #     while(len(waveform) < audio_length):
        #         waveform = torch.cat([waveform, waveform[0: max(len(waveform), audio_length - len(waveform))]], dim=0)
        # else:
        #     waveform = waveform[0: audio_length]

        #waveform = torchaudio.functional.filtering.vad(waveform, sample_rate=sample, noise_reduction_amount= 3)[0]

        spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length= hop_length, win_length=win_length)
        
        # waveform5 = torchaudio.functional.gain(waveform, gain_db=500.0)
        # waveform5 = torchaudio.transforms.MuLawEncoding()(waveform5).float()
        # specgram5 = spec(waveform5)
        # specgram5 = self._normalize(specgram5).unsqueeze(dim=0)
        # specgram5 = self.transform(specgram5).squeeze(dim=0)


        # 음향신호를 시각화 x=시간, y=주파수, z=진폭
        # specgram1 = spec(waveform)
        # specgram1 = self._normalize(specgram1).unsqueeze(dim=0)
        # specgram1 = self.transform(specgram1).squeeze(dim=0)

        # # 고역대 추출
        # waveform_high = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        # specgram2 = spec(waveform_high)
        # specgram2 = self._normalize(specgram2).unsqueeze(dim=0)
        # specgram2 = self.transform(specgram2).squeeze(dim=0)
        
        # # 저역대 추출
        # waveform_low = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        # specgram3 = spec(waveform_low)
        # specgram3 = self._normalize(specgram3).unsqueeze(dim=0)
        # specgram3 = self.transform(specgram3).squeeze(dim=0)

        n_fft = 1024
        hop_length = 512
        win_length = 1024
        n_mels = 224
        sample_rate = 22050
        spec = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length= hop_length, win_length=win_length, n_mels=n_mels)

        #음향신호를 시각화 x=시간, y=주파수, z=진폭
        specgram4 = spec(waveform)
        specgram4 = self._normalize(specgram4).unsqueeze(dim=0)
        specgram4 = self.transform(specgram4).squeeze(dim=0)

        #고역대 추출
        # waveform_high = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        # specgram5 = spec(waveform_high)
        # specgram5 = self._normalize(specgram5).unsqueeze(dim=0)
        # specgram5 = self.transform(specgram5).squeeze(dim=0)
        
        # # 저역대 추출
        # waveform_low = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        # specgram6 = spec(waveform_low)
        # specgram6 = self._normalize(specgram6).unsqueeze(dim=0)
        # specgram6 = self.transform(specgram6).squeeze(dim=0)


        # # 주로 음성신호를 캐치하기위한 스펙트로그램
        # # specgram2 = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length= hop_length, win_length=win_length, n_mels=n_mels)(waveform)
        # # specgram2 = self._normalize(specgram2).unsqueeze(dim=0)
        # # specgram2 = self.transform(specgram2).squeeze(dim=0)

        # waveform2 = waveform.numpy()
        # specgram2 = librosa.feature.mfcc(waveform2, sr=sample, n_mfcc = 64)
        # specgram2 = torch.tensor(specgram2)
        # specgram2 = self._normalize(specgram2).unsqueeze(dim=0)
        # specgram2 = self.transform(specgram2).squeeze(dim=0)
      
        # waveform3 = waveform.numpy()
        # hop_length_duration = float(512)/sample_rate
        # n_fft_duration = float(512)/sample_rate
        # specgram3 = np.abs(librosa.stft(waveform3, n_fft=n_fft_duration, hop_length=hop_length_duration))
        # specgram3 = librosa.amplitude_to_db(specgram3, ref=np.max)
        # specgram3 = torch.tensor(specgram3)
        # specgram3 = self._normalize(specgram3).unsqueeze(dim=0)
        # specgram3 = self.transform(specgram3).squeeze(dim=0)



        specgram = torch.stack([specgram4], dim=0) # ,specgram2, specgram3,specgram4,specgram5

        return specgram

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def _getfile_list(self, path, parent=""):
        file_paths = []
        file_names = []
        labels = []
        for filename in os.listdir(path):
            fullpath = os.path.join(path, filename)
            if os.path.isfile(fullpath):
                if (parent is ""): parent = "None"
                file_paths.append(fullpath)
                file_names.append(filename)
                labels.append(parent)
            else:
                f, p, l = self._getfile_list(fullpath, filename)
                file_names += f
                file_paths += p
                labels += l
        return file_names, file_paths, labels
    
    def _normalize(self, tensor):
        tensor_minusmean = tensor - tensor.mean()
        return tensor_minusmean/tensor_minusmean.abs().max()
    
    
    def __getitem__(self, index):
        data = None
        if self.use_cache:
            data = self.data[index]
        else:
            data = self._cache(index)

        return data, self.file_names[index], self.labels_idx[index]

    def __len__(self):
        return len(self.file_names)
        
def test(model, test_path = "./test/", label = ["1", "0"]):
    ret = []
    model.eval()
    dataset = SoundDataset(test_path, is_train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    with torch.no_grad():
        for batch_idx, (data, filename, target) in tqdm(enumerate(loader)):
            data = data.to(device)
            target = target.to(device).float()
            output = model(data)
            output = torch.sigmoid(output).round()
            ret.append((filename[0], label[output.squeeze(1).squeeze().int()]))
    return ret 

# def splitSubset(dataset, validation_length=[120,100]):
#     leak = []
#     normal = []
#     for idx, label in enumerate(dataset.labels):
#         if label == "leak":
#             leak.append(idx)
#         else:
#             normal.append(idx)
        
#     leak_indices = randperm(sum([len(leak)]), generator=default_generator).tolist()
#     normal_indices = randperm(sum([len(normal)]), generator=default_generator).tolist()
    
#     #train = normal[0:len(normal_indices) - validation_length[0]] + leak[0:len(leak_indices) - validation_length[1]]

#     train = normal[0:len(leak_indices) - validation_length[1]] + leak[0:len(leak_indices) - validation_length[1]]

#     vaildation = normal[len(normal_indices) - validation_length[0]:] + leak[len(leak_indices) - validation_length[1]:]
#     return  [torch.utils.data.Subset(dataset, train), torch.utils.data.Subset(dataset, vaildation)]


# model = models.alexnet()
# num_f = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_f, 1)
# model.features[0] = nn.Conv2d(5, 64, kernel_size=(11, 11), stride=(4, 4), padding=(3, 3))

# model = models.mobilenet_v3_small()
# model.features[0][0] = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# num_f = model.classifier[3].in_features
# model.classifier[3] = nn.Linear(num_f, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', dest='f', type=str)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    result_path = args.f
    pretrained = args.pretrained
    # random_seed =45
    # epochs = 50
    # step_size = 20
    # lr = 0.01
    # pretrained = False
    # weight_decay = 0.9

    train_path = "/content/train"
    test_path = "/content/test"
    cache_file = "/content/audio.cache"
    cache_file_on_drive = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/cache/audio.cache"

    if os.path.isdir(train_path) == False:
        subprocess.run(["unzip", "-qq", "/content/drive/MyDrive/데이터분석/dataset.zip", "-d", "/content"])

    # if os.path.isfile(cache_file_on_drive) and os.path.isfile(cache_file) == False:
    #     shutil.copyfile(cache_file_on_drive, cache_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 모델 설정
    model = models.resnet18(pretrained=pretrained)
    num_f = model.fc.in_features
    model.fc = nn.Linear(num_f, 1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)


    # 모델 학습습 로깅깅 설정
    start_time = time.time()

    #/content/drive/MyDrive/데이터분석/2_수도관누수데이터/model/1636370724/24_0.3122.pt
    model.load_state_dict(torch.load(result_path))
    ret = test(model, test_path)
    df = pd.DataFrame(ret, columns=["AudioName","ClassID"])
    output_path = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/{:0f}.csv".format(start_time)
    df.to_csv(output_path, index = False)

    print("output : {}".format(output_path))
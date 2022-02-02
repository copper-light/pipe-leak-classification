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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class CCNtoLSTM(nn.Module):

    def __init__(self):
        super(CCNtoLSTM, self).__init__()
        self.num_layers = 2
        self.hidden_size =512

        self.lstm = nn.LSTM(input_size=3584, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False)
        self.bn = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
    def forward(self, x):
        x = x.transpose(1,2).transpose(0,1)
        x = x.reshape(x.size(0), x.size(1), x.size(2)*x.size(3))

        x = self.lstm(x)[0]
        x = self.bn(x[-1])
        return x


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()
        
    def forward(self, x):
        return x


class SoundDataset(Dataset):
    
    def __init__(self, dataset_path, is_train = True, cutoff_freq = 3000, use_cache = False, cache_file="/content/sound_data.cache", save_cache_file="/content/sound_data.cache"):
    #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.file_path = []
        self.labels = []

        self.file_names, self.file_path, self.labels = self._getfile_list(dataset_path)
        self.idx_to_label, self.labels_idx = np.unique(self.labels, return_inverse=True)
        self.is_train = is_train
        self.data = []
        self.use_cache = use_cache
        self.cutoff_freq = cutoff_freq
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
        # specgram4 = spec(waveform)
        # specgram4 = self._normalize(specgram4).unsqueeze(dim=0)
        # specgram4 = self.transform(specgram4).squeeze(dim=0)

        #고역대 추출
        # waveform_high = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        # specgram5 = spec(waveform_high)
        # specgram5 = self._normalize(specgram5).unsqueeze(dim=0)
        # specgram5 = self.transform(specgram5).squeeze(dim=0)
        
        # # 저역대 추출
        waveform_low = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample, cutoff_freq=cutoff_freq)
        specgram6 = spec(waveform_low)
        specgram6 = self._normalize(specgram6).unsqueeze(dim=0)
        specgram6 = self.transform(specgram6).squeeze(dim=0)


        # 주로 음성신호를 캐치하기위한 스펙트로그램
        # specgram2 = torchaudio.transforms.MelSpectrogram(n_fft=n_fft, hop_length= hop_length, win_length=win_length, n_mels=n_mels)(waveform)
        # specgram2 = self._normalize(specgram2).unsqueeze(dim=0)
        # specgram2 = self.transform(specgram2).squeeze(dim=0)

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



        specgram = torch.stack([specgram6], dim=0) # ,specgram2, specgram3,specgram4,specgram5

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
        return tensor/tensor_minusmean.abs().max()
    
    
    def __getitem__(self, index):
        data = None
        if self.use_cache:
            data = self.data[index]
        else:
            data = self._cache(index)

        return data, self.file_names[index], self.labels_idx[index]

    def __len__(self):
        return len(self.file_names)

def print_log(text):
    logger.info(text)

def count_collect(output, target):
    ret = 0
    with torch.no_grad():
        ret = (output.round() == target).sum().float()
    return ret


def train(start_time, model, epoch, loader):
    model.train()
    aucc = .0

    y_true = []
    y_pred = []
    total_loss= .0

    for batch_idx, (data, filename, target) in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device).float()
        data = data.requires_grad_() #set requires_grad to True for training
        output = model(data)
        output = output.squeeze(dim=1)
        # target = target.unsqueeze(dim=1)
        loss = criterion(output, target) #the loss functions expects a batchSizex10 input
        loss.backward()
        optimizer.step()

        total_loss += loss
        
        output = torch.sigmoid(output)
        aucc = count_collect(output, target)
        y_true += target.int().tolist()
        y_pred += output.round().int().tolist()

        
        print('\r{:.1f}s, Train Epoch: {} (Step {}/{}) [{}]\tLoss: {:.6f}\tAucc: {:.4f}'.format(
            time.time()-start_time,
            epoch, 
            batch_idx+1, len(loader),
            len(loader.dataset), loss, aucc/len(target)), end="")
        
    f1 = f1_score(y_true, y_pred)
    total_loss/=len(loader)
    
    log = '\r{:.1f}s, Train Epoch: {} [{}]\tLoss: {:.6f}\tf1: {:.4f}'.format(
        time.time()-start_time,
        epoch, 
        len(loader.dataset),
        total_loss,
        f1)
    print_log(log)
            
def val(start_time, model, epoch, loader, std = 0.5):
    model.eval()
    aucc = .0
    loss = .0
    y_true = []
    y_pred = []
    f1 = .0
    leak_loss= .0
    leak_cnt = 0
    with torch.no_grad():
        for batch_idx, (data, filename, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device).float()
            output = model(data)
            output = output.squeeze(dim=1)
            # target = target.unsqueeze(dim=1)
            #loss += criterion(output, target) #the loss functions expects a batchSizex10 input
            
            cur_loss= criterion(output, target) #the loss functions expects a batchSizex10 input
            if target[0] == 0.:
                leak_loss += cur_loss
                leak_cnt += 1
            loss += cur_loss

            output = torch.sigmoid(output)
            aucc += count_collect(output, target)
            y_true += target.int().tolist()
            y_pred += output.round().int().tolist()

        f1 = f1_score(y_true, y_pred)
        loss/=len(loader)
        leak_loss/=leak_cnt

        log = '{:.1f}s, Val Epoch: {} [{}]\tLoss: {:.6f}\tleak_loss: {:.6f}\tAucc: {:.4f}\tf1: {:.4f}'.format(
            time.time()-start_time,
            epoch, 
            len(loader.dataset),
            loss,
            leak_loss,
            aucc/len(loader.dataset),
            f1)
        print_log(log)
        log = confusion_matrix(y_true, y_pred)
        print_log(log)

    return loss, leak_loss, f1
        
def test(model, test_path = "./test/", label = ["1", "0"]):
    ret = []
    model.eval()
    dataset = SoundDataset(test_path, is_train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1)
    with torch.no_grad():
        for batch_idx, (data, filename, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device).float()
            output = model(data)
            output = torch.sigmoid(output).round()
            ret.append((filename[0], label[output.squeeze(1).squeeze().int()]))
    return ret 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def splitSubset(dataset, validation_length=[120,100]):
    leak = []
    normal = []
    for idx, label in enumerate(dataset.labels):
        if label == "leak":
            leak.append(idx)
        else:
            normal.append(idx)
        
    # leak_indices = randperm(sum([len(leak)]), generator=default_generator).tolist()
    # normal_indices = randperm(sum([len(normal)]), generator=default_generator).tolist()

    t_normal = normal[0:len(normal) - validation_length[0]]
    v_normal = normal[len(normal) - validation_length[0]:]

    t_leak = leak[0:len(leak) - validation_length[1]]
    v_leak = leak[len(leak) - validation_length[1]:]


    rate = int(len(normal) / len(leak)) + 3
    normal_size = int(len(normal) / rate) 

    print("trainset - {} normal : {} leak".format(normal_size, len(t_leak)))

    train = []
    offset = 0
    for no in range(rate):
        train_list= None
        if no + 1 == rate:
            train_list= t_normal[(normal_size * no):] + t_leak
        else:
            train_list= t_normal[(normal_size * no):(normal_size * (no+1))] + t_leak
        train.append(torch.utils.data.Subset(dataset, train_list))

    vaildation_list = v_normal + v_leak
    vaildation = torch.utils.data.Subset(dataset, vaildation_list)
    return train, vaildation

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
    parser.add_argument('--seed', dest='random_seed', type=int, default=45)
    parser.add_argument('--lr', dest='lr', type=float, default=0.1)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--step_size', dest='step_size', type=int, default=20)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.01)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=False)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.1)
    parser.add_argument('--cutoff_freq', dest='cutoff_freq', type=int, default=3000)
     
    args = parser.parse_args()
    
    cutoff_freq = args.cutoff_freq
    random_seed = args.random_seed
    epochs = args.epochs
    step_size = args.step_size
    lr = args.lr
    pretrained = args.pretrained
    weight_decay = args.weight_decay
    gamma = args.gamma

    # random_seed =45
    # epochs = 50
    # step_size = 20
    # lr = 0.01
    # pretrained = False
    # weight_decay = 0.9

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    g = torch.Generator()
    g.manual_seed(random_seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    train_path = "/content/train"
    test_path = "/content/test"
    cache_file = "/content/audio.cache"
    cache_file_on_drive = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/cache/audio.cache"

    if os.path.isdir(train_path) == False:
        subprocess.run(["unzip", "-qq", "/content/drive/MyDrive/데이터분석/dataset.zip", "-d", "/content"])

    # if os.path.isfile(cache_file_on_drive) and os.path.isfile(cache_file) == False:
    #     shutil.copyfile(cache_file_on_drive, cache_file)


    # 모델 설정
    model = models.resnet18(pretrained=pretrained)
    num_f = model.fc.in_features
    model.fc = nn.Linear(num_f, 1)
    #model.layer3 = EmptyLayer()
    model.layer4 = EmptyLayer()
    #model.conv1 = nn.Conv2d(2, 64, kernel_size=(15, 15), stride=(4, 4), padding=(7, 7), bias=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.avgpool = CCNtoLSTM()
    model = model.to(device)

    # 데이터셋 로드
    dataset = SoundDataset(train_path, use_cache=True, cutoff_freq = cutoff_freq, cache_file=cache_file, save_cache_file=cache_file_on_drive)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-1000, 1000])

    train_set_list, val_set = splitSubset(dataset, validation_length = [200,300]) # [normal, leak]
    trn_loaders = []

    for sub in train_set_list:
        loader = torch.utils.data.DataLoader(sub, batch_size = 256, shuffle = True, worker_init_fn=seed_worker, generator=g, num_workers=0) # resnet18 best : 256  , generator=g
        trn_loaders.append(loader)

    #trn_loader = torch.utils.data.DataLoader(train_set, batch_size = 256, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 1)


    # 모델 학습습 로깅깅 설정
    start_time = time.time()

    model_save_path = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/model/{:.0f}/".format(start_time)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(filename="{}train.log".format(model_save_path))
    fh.setLevel(logging.INFO)
    logger.addHandler(ch)
    logger.addHandler(fh)

    try:
        print_log("PYTHONHASHSEED {}".format(os.environ['PYTHONHASHSEED']))
    except:
        print_log("PYTHONHASHSEED IS NONE")
    print_log(args)
    
    # 모델델 학습습 시작
    patience = 30
    best_loss = 10.0
    best_f1 = .0
    stop_count = 0

    optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=20, step_size_down=None, mode='exp_range', gamma=0.995)


    criterion = torch.nn.BCEWithLogitsLoss()

    trn_loader_size = len(trn_loaders)
    for epoch in range(1, epochs+1):
        dataset.train()
        
        train(start_time, model, epoch, trn_loaders[epoch % trn_loader_size])
        dataset.eval()
        loss, leak_loss, f1 = val(start_time, model,epoch, val_loader)

        scheduler.step()
        
        if epoch % step_size == 0:
            print("Decay learn rate")
        
        if best_loss > loss:
            model_path = os.path.join(model_save_path, "loss_{}_{:.4f}.pt".format(epoch, loss))
            torch.save(model.state_dict(), model_path)

            print_log("Save model : {}".format(model_path))
            best_loss = loss
            stop_count = 0
        elif best_loss + 0.5 < loss:
            stop_count += 1
        else:
            stop_count = 0

        if best_f1 < f1:
            best_f1 = f1
            model_path = os.path.join(model_save_path, "f1_{}_{:.4f}.pt".format(epoch, best_f1))
            torch.save(model.state_dict(), model_path)
            print_log("Save model : {}".format(model_path))
        
        if stop_count >= patience:
            print_log("EarlyStop")
            model_path = os.path.join(model_save_path, "{}_{:.4f}_last.pt".format(epoch, loss))
            torch.save(model.state_dict(), model_path)
            break
            
    print_log("{:.1f}, Done".format(time.time()-start_time))  
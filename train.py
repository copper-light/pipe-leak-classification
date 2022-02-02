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

class AudioTransforms(object):

    def __init__(self, name, args):
        
        self.transfs = {
            'val': transforms.Compose([
                ToTensorAudio()
            ]),
            'train': transforms.Compose([
                AdditiveNoise(*args['noise']),
                RandomCropLength(*args['crop']),
                ToTensorAudio()
            ])
        }[name]
        
    def apply(self, data):
        audio, sr = data
        # audio -> (time, channel)
        return self.transfs(audio), sr
        
    def __repr__(self):
        return self.transfs.__repr__()


class ProcessChannels(object):

    def __init__(self, mode):
        self.mode = mode

    def _modify_channels(self, audio, mode):
        if mode == 'mono':
            new_audio = audio if audio.ndim == 1 else audio[:,:1]
        elif mode == 'stereo':
            new_audio = np.stack([audio]*2).T if audio.ndim == 1 else audio
        elif mode == 'avg':
            new_audio= audio.mean(axis=1) if audio.ndim > 1 else audio
            new_audio = new_audio[:,None] 
        else:
            new_audio = audio
        return new_audio

    def __call__(self, tensor):
        return self._modify_channels(tensor, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '(mode={})'.format(self.mode)


class ToTensorAudio(object):

    def __call__(self, tensor):
        return torch.from_numpy(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AugmentationTransform(object):

    def __init__(self, prob=None, sig=None, dist_type='uniform'):
        self.sig, self.dist_type = sig, dist_type 
        self.dist = self._get_dist(sig, dist_type)
        self.prob = prob

    def _get_dist(self, sig, dist_type):
        dist = None
        if dist_type == 'normal':
            dist = lambda x: np.random.normal(0, sig, x) 
        elif dist_type == 'uniform':
            dist = lambda x: np.random.uniform(-sig, sig, x)
        elif dist_type == 'half':
            dist = lambda x: np.clip(
                                np.abs(
                                    np.random.normal(0, sig, x)),
                                a_min=0.0,
                                a_max=0.8)
        else:
            raise ValueError('Unimplemented distribution')
        return dist

    def __call__(self, tensor):
        if np.random.rand() <= self.prob:
            return self.transform(tensor)
        return tensor

    def transform(self, tensor):
        raise NotImplementedError

    def __repr__(self):
        param_str = '(prob={}, sig={}, dist_type={})'.format(
                        self.prob, self.sig, self.dist_type)
        return self.__class__.__name__ + param_str
    

class AdditiveNoise(AugmentationTransform):

    def  __init__(self, prob, sig, dist_type='normal'):
        super(AdditiveNoise, self).__init__(prob, sig, dist_type)


    def _noise(self, length):
        return self.dist(length)

    def transform(self, tensor):
        #print("AdditiveNoise", tensor.shape)
        noise = self._noise(tensor.shape[0])[:,None]
        return tensor + noise


class RandomCropLength(AugmentationTransform):

    def __init__(self, prob, sig, dist_type='half'):
        super(RandomCropLength, self).__init__(prob, sig, dist_type)

    def transform(self, tensor):
        #print("RandomCropLength", tensor.shape)
        ind_start, ind_end, perc = self._crop_inds(tensor.shape[1])
        #print(ind_start, ind_end, perc)
        return np.array([self._check_zero(tensor[0,ind_start:ind_end])])

    def _check_zero(self, tensor):
        #print("_check_zero", tensor.shape)
        return tensor + 1e-8 if tensor.sum() == 0 else tensor

    def _crop_inds(self, length):
        d = self.dist(1)[0]
        assert d < 0.9

        perc = 1 - d
        new_length = np.round(length * perc).astype(int)
        max_start = length - new_length + 1
        ind_start = np.random.randint(0, max_start)
        ind_end = ind_start + new_length
        return ind_start, ind_end, perc




class ModifyDuration(object):

    def __init__(self, duration):
        self.duration = duration

    def __call__(self, tensor):
        return self._modify_duration(tensor, self.duration)

    def __repr__(self):
        return self.__class__.__name__ + '(duration={})'.format(self.duration)


    def _modify_duration(self, audio, dur):
        
        if dur < len(audio):
            max_index_start = len(audio) - dur
            index_start = np.random.randint(0,max_index_start)
            index_end = index_start + dur
            new_audio = audio[index_start:index_end]
        else:
            ratio = dur/len(audio)
            full_reps = [audio]*int(ratio)
            padd_reps = [audio[:round(len(audio)*(ratio%1))]]
            new_audio = np.concatenate(full_reps + padd_reps, axis=0)
            
        return new_audio 

class SoundDataset(Dataset):
    
    def __init__(self, dataset_path, is_train = True, use_cache = False, augmentation= True, cache_file="/content/sound_data.cache", save_cache_file="/content/sound_data.cache"):
    #initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.file_path = []
        self.labels = []

        self.file_names, self.file_path, self.labels = self._getfile_list(dataset_path)

        self.idx_to_label, self.labels_idx = np.unique(self.labels, return_inverse=True)

        self.is_train = is_train

        self.data = []

        self.augmentation = augmentation

        if augmentation:
            self.use_cache = False
        else:
            self.use_cache = use_cache
        aug_args = args= {
            "noise"    : [0.3, 0.0005],
            "crop"     : [0.4, 0.15]
        }

        self.audio_trans = AudioTransforms("train", aug_args)    


        # n_fft = 512
        # hop_length = 512
        # win_length = 512
        # n_mels = 128

        n_fft = 448
        win_length = 448
        hop_length = 344
        n_mels = 128
        self.sample_rate = 22050
        self.minimum_length = int(self.sample_rate*3)

        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(n_fft=(n_fft*2), win_length=(win_length*2), hop_length=hop_length, n_mels=n_mels)
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc = n_mels, log_mels=True,
            melkwargs={
                "n_fft" : (n_fft*2), 
                "win_length" : (win_length*2), 
                "hop_length" : hop_length, 
                "n_mels" : n_mels
            }
        )

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
        sample_rate = self.sample_rate
        minimum_length = self.minimum_length
        path = self.file_path[index]
        waveform, sample = torchaudio.load(path)
        # waveform = torchaudio.transforms.Resample(sample, sample_rate)(waveform)

        if self.is_train:
            self.transform = transforms.Compose([
                    #transforms.Normalize(mean=[0.485], std=[0.229]), 
                    
                    transforms.Resize((224,224)),
                    #transforms.RandomHorizontalFlip()
            ])

            if self.augmentation:
                waveform = self.audio_trans.apply((waveform.numpy(), sample_rate))[0].type(torch.FloatTensor)
            #print(waveform.shape)
        else:
            self.transform = transforms.Compose([
                    # transforms.Normalize(mean=[0.485], std=[0.229]),
                    transforms.Resize((224,224)),
            ])
        
        w = waveform[0, int(sample_rate/4):int(len(waveform[0])-(sample_rate/4))]
        if len(w) > minimum_length:
            start = int( (len(w)-minimum_length)/ 2)
            w = w[start:start+minimum_length]
        else:
            while(len(w) < minimum_length):
                w = torch.cat([w[:], w[0:minimum_length-len(w)]])
        
        waveform = w.unsqueeze(dim=0)
        #waveform = w.unsqueeze(dim=0)

        # waveform = torchaudio.transforms.Fade(fade_in_len = fade_size, fade_out_len = fade_size, fade_shape="exponential")(waveform)
        # waveform = waveform[0]

        # waveform[waveform > 0.0015] = 0.0015
        # waveform[waveform < -0.002] = -0.002

        # while(len(waveform) < 22050*5):
        #     waveform = torch.cat([waveform, waveform[0: max(len(waveform), 22050*5 - len(waveform))]], dim=0)

        #waveform = torchaudio.functional.filtering.vad(waveform, sample_rate=sample, noise_reduction_amount= 3)[0]

        #음향신호를 시각화 x=시간, y=주파수, z=진폭
        waveform = torchaudio.functional.gain(waveform, gain_db=200.0) # best 200

        #waveform_low = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample, cutoff_freq=3000)
        #specgram1 = torchaudio.transforms.Spectrogram()(waveform)
        specgram1 = self.spec(waveform)
        specgram1 = self._normalize(specgram1)
        specgram1 = self.transform(specgram1).squeeze(dim=0)
        #specgram1 = self.transform(specgram1).squeeze(dim=0)
        # 음성신호를 캐치하기위한 스펙트로그램

        #waveform_low = torchaudio.functional.lowpass_biquad(waveform, sample_rate=sample, cutoff_freq=3000)

        specgram2 = self.mel_spec(waveform)
        #specgram2 = specgram2[:224,:224]
        #specgram2 = self._mono_to_color(specgram2)
        specgram2 = self._normalize(specgram2)
        specgram2 = self.transform(specgram2).squeeze(dim=0)

        specgram3 = self.mfcc(waveform)
        specgram3 = self._normalize(specgram3)
        specgram3 = self.transform(specgram3).squeeze(dim=0)

        # specgram1 = torchaudio.transforms.MuLawEncoding()(waveform).float()
        # specgram1 = self.spec(specgram1)
        # specgram1 = self._normalize(specgram1)
        # specgram1 = self.transform(specgram1).squeeze(dim=0)

        specgram = torch.stack([specgram1, specgram2, specgram3], dim=0) # ,specgram2, specgram3,specgram4,specgram5

        return specgram #torch.tensor(specgram2[0])

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
            # if self.labels_idx[index]==0:
            #     data = self._cache(index)    
            # else:
            #     data = self.data[index]
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
            loss += cur_loss

            output = torch.sigmoid(output)
            aucc += count_collect(output, target)
            y_true += target.int().tolist()
            y_pred += output.round().int().tolist()

        f1 = f1_score(y_true, y_pred)
        loss/=len(loader)

        log = '{:.1f}s, Val Epoch: {} [{}]\tLoss: {:.6f}\tleak_loss: {:.6f}\tAucc: {:.4f}\tf1: {:.4f}'.format(
            time.time()-start_time,
            epoch, 
            len(loader.dataset),
            loss,
            0.0,
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


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--seed', dest='random_seed', type=int, default=45)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--step_size', dest='step_size', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.4)
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=False)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.1)
    parser.add_argument('--test', dest='test_path', default='', type=str)
    parser.add_argument('--patience', dest='patience', default=40, type=int)
    parser.add_argument('--augmentation', dest='augmentation', type=bool, default=False)

    args = parser.parse_args()

    random_seed = args.random_seed
    epochs = args.epochs
    step_size = args.step_size
    lr = args.lr
    pretrained = args.pretrained
    weight_decay = args.weight_decay
    gamma = args.gamma
    test_model_path = args.test_path
    patience = args.patience
    augmentation = args.augmentation
    batch_size = args.batch_size

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

    torch.backends.cudnn.deterministic = True
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 모델 설정
    model = models.resnet50(pretrained=pretrained)
    num_f = model.fc.in_features
    model.fc = nn.Linear(num_f, 1)
    #model.conv1 = nn.Conv2d(2, 64, kernel_size=(15, 15), stride=(4, 4), padding=(7, 7), bias=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.to(device)

    if len(test_model_path) == 0 :
        # 데이터셋 로드
        dataset = SoundDataset(train_path, use_cache=True, augmentation=augmentation, cache_file=cache_file, save_cache_file=cache_file_on_drive)
        #train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-1000, 1000])

        train_set_list, val_set = splitSubset(dataset, validation_length = [200,300]) # [normal, leak]
        trn_loaders = []

        for sub in train_set_list:
            loader = torch.utils.data.DataLoader(sub, batch_size = batch_size, shuffle = True, worker_init_fn=seed_worker, generator=g, num_workers=0) # resnet18 best : 256  , generator=g
            trn_loaders.append(loader)

        #trn_loader = torch.utils.data.DataLoader(train_set, batch_size = 256, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size)

        model_save_path = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/model/{:.0f}/".format(start_time)

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # 모델 학습습 로깅깅 설정
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
        best_loss = 10.0
        best_f1 = .0
        stop_count = 0

        optimizer = optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
        #optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [120, 240], gamma = gamma, last_epoch=-1) # 60,120,240
        #scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=20, step_size_down=None, mode='exp_range', gamma=0.995)
        #scheduler = optim.ReduceLROnPlateau(optimizer, 'min')
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3)

        cur_lr = lr
        lr_count = 0

        criterion = torch.nn.BCEWithLogitsLoss()

        trn_loader_size = len(trn_loaders)
        for epoch in range(1, epochs+1):
            dataset.train()
            
            train(start_time, model, epoch, trn_loaders[epoch % trn_loader_size])
            dataset.eval()
            loss, leak_loss, f1 = val(start_time, model,epoch, val_loader)

            #scheduler.step()
            
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
    else:
        if os.path.isfile(test_model_path):
            model.load_state_dict(torch.load(test_model_path))
            ret = test(model, test_path)
            df = pd.DataFrame(ret, columns=["AudioName","ClassID"])
            output_path = "/content/drive/MyDrive/데이터분석/2_수도관누수데이터/{:0f}.csv".format(start_time)
            df.to_csv(output_path, index = False)

            print("output : {}".format(output_path))
        else:
            print("not found file : {}".format(test_model_path))

    print_log("{:.1f}, Done".format(time.time()-start_time))  
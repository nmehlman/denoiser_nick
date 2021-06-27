import os
import pandas as pd
import glob
from tqdm import tqdm
import torchaudio
import sys
import numpy as np
import random
from pdb import set_trace as bp
import scipy
from scipy.io.wavfile import write
random.seed(10101)

List_snr = []
name_audio = []
noise_name = []

noise_path = sys.argv[1]
clean_path = sys.argv[2]
noise_list = glob.glob(os.path.join(noise_path, '*.wav'))
clean_list = glob.glob(os.path.join(clean_path, '*.wav'))
snrs = [18,21,24,27,30]
#print(noise_list)
#print(clean_list)
for idx, clean_path in enumerate(clean_list):
    name = clean_path.split('/')[-1]
    snr = random.sample(snrs,1)[0]
    print("SNR = ", snr)
    clean, fs = torchaudio.load(clean_path)
    clean = clean[0,:].numpy()
    clean_db = 10 * np.log10(np.mean(clean ** 2)+1e-4)
    print(clean_db)
    noise_audio = random.choice(noise_list)
    List_snr.append(snr)
    noise_name.append(noise_audio)
    name_audio.append(name)
    noise, fs = torchaudio.load(noise_audio)
    if fs != 16000:
        resample = torchaudio.transforms.Resample(fs, 16000, resampling_method='sinc_interpolation')
        noise = resample(noise[0,:]).numpy()
    else:
        noise = noise[0,:].numpy()
    noise_db = 10 * np.log10(np.mean(noise ** 2)+1e-4)
    noise = np.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noise
    abc = 10 * np.log10(np.mean(noise ** 2)+1e-4)
    print("noisy signal db ", abc)

    if noise.shape[0] > clean.shape[0]:
        start = random.sample(range(noise.shape[0]-clean.shape[0]),1)[0]
        noise = noise[start:start+clean.shape[0]]
    else:
        shortage = clean.shape[0] - noise.shape[0]
        noise = np.pad(noise, (0, shortage), 'wrap')
    noisy_signal = clean + noise
    #noisy_db = 10 * np.log10(np.mean(noisy_signal ** 2)+1e-4)
    #print("noisy signal db ", noisy_db)
    write("/data/home/anirudh/GARD/data/Valentini/noisy_trainset_56spk_wav_resampled/{}".format(name), 16000, noisy_signal)
d = {'noise_SNR': List_snr, 'noise_name': noise_name, 'name_audio': name_audio}
df = pd.DataFrame(data=d)
s = df.to_csv("/data/home/anirudh/GARD/denoiser/Valentini_full_train_snr_data.csv")
print(s)

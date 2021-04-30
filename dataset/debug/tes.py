import dpam

loss_fn = dpam.DPAM()
wav_ref = dpam.load_audio('clean/p287_003.wav')
wav_out = dpam.load_audio('noisy/p287_003.wav')
loss =  loss_fn.forward(wav_ref,wav_out)
print(loss)

import torch
import numpy as np

''' FFT augmentation '''
def FFT2_Amp_MixUp(data_original, data_aug, lamda):
    fft_data_original = torch.fft.fft2(data_original)
    fft_data_aug = torch.fft.fft2(data_aug)
    
    aug_amp = lamda*torch.abs(fft_data_original) + (1-lamda)*torch.abs(fft_data_aug)
    fft_mixup_data = torch.mul(aug_amp, torch.exp(1j*torch.angle(fft_data_original)))
    return torch.real(torch.fft.ifft2(fft_mixup_data))

def Combine_AmplitudeANDPhase(amp, phe):
    return torch.mul(amp, torch.exp(1j*phe))

def FFT_Exchange_Amplitude(domain_data1, domain_data2):
    fft_domain1 = torch.fft.fft2(domain_data1)
    fft_domain2 = torch.fft.fft2(domain_data2)
    lamda1 = torch.rand(1)/2 + 0.5 # [0.5, 1.0]
    lamda2 = torch.rand(1)/2 + 0.5
    lamda1, lamda2 = lamda1.to(domain_data1.device), lamda2.to(domain_data2.device)
    cross_amp_domain1 = lamda1*torch.abs(fft_domain2) + (1-lamda1)*torch.abs(fft_domain1)
    cross_amp_domain2 = lamda2*torch.abs(fft_domain1) + (1-lamda2)*torch.abs(fft_domain2)
    cross_domain1 = Combine_AmplitudeANDPhase(cross_amp_domain1, torch.angle(fft_domain1))
    cross_domain2 = Combine_AmplitudeANDPhase(cross_amp_domain2, torch.angle(fft_domain2))
    return torch.real(torch.fft.ifft2(cross_domain1)), torch.real(torch.fft.ifft2(cross_domain2))

def Batch_FFT2_Amp_MixUp(data_original, data_aug, p=0.5):
    '''
    augmentation between two batch of data
    '''
    apply_p = np.random.rand()
    if apply_p<=p:
        lamda_vector = np.random.rand(data_original.size(0))
        for i in range(data_original.size(0)):
            data_original[i] = FFT2_Amp_MixUp(data_original[i], data_aug[i], lamda_vector[i])
        return data_original
    else:
        return data_original

import wave
import numpy as np
import pylab as pl
from scipy.signal import resample

# read wave file and get parameters.
fw = wave.open('../sounds/Beleswar.wav','rb')
params = fw.getparams()
print(params)
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw.readframes(nframes)
waveData = np.fromstring(strData, dtype=np.int16)
waveData = waveData*1.0/max(abs(waveData))  # normalization
fw.close()
pitch1 = waveData / np.linalg.norm(waveData)

'''
# plot the wave
time = np.arange(0, len(waveData)) * (1.0 / framerate)

index1 = 10000.0 / framerate
index2 = 10512.0 / framerate
index3 = 15000.0 / framerate
index4 = 15512.0 / framerate

pl.subplot(311)
pl.title("pitch")
pl.plot(time, waveData)
pl.plot([index1,index1],[-1,1],'r')
pl.plot([index2,index2],[-1,1],'r')
pl.plot([index3,index3],[-1,1],'g')
pl.plot([index4,index4],[-1,1],'g')
pl.xlabel("time (seconds)")
pl.ylabel("Amplitude")

pl.subplot(312)
pl.plot(np.arange(512),waveData[10000:10512],'r')
pl.plot([59,59],[-1,1],'b')
pl.plot([169,169],[-1,1],'b')
print(1/( (169-59)*1.0/framerate ))
pl.xlabel("index in 1 frame")
pl.ylabel("Amplitude")

pl.subplot(313)
pl.plot(np.arange(512),waveData[15000:15512],'g')
pl.xlabel("index in 1 frame")
pl.ylabel("Amplitude")
pl.savefig("pitch.png")
pl.show()
'''

fw1 = wave.open('../sounds/Jay.wav','rb')
params = fw1.getparams()
print(params)
nchannels, sampwidth, framerate, nframes = params[:4]
strData = fw1.readframes(nframes)
waveData1 = np.fromstring(strData, dtype=np.int16)
waveData1 = waveData1*1.0/max(abs(waveData1))  # normalization
fw1.close()
pitch2 = waveData1 / np.linalg.norm(waveData1)

'''
# plot the wave
time = np.arange(0, len(waveData)) * (1.0 / framerate)

index1 = 10000.0 / framerate
index2 = 10512.0 / framerate
index3 = 15000.0 / framerate
index4 = 15512.0 / framerate

pl.subplot(311)
pl.title("pitch")
pl.plot(time, waveData)
pl.plot([index1,index1],[-1,1],'r')
pl.plot([index2,index2],[-1,1],'r')
pl.plot([index3,index3],[-1,1],'g')
pl.plot([index4,index4],[-1,1],'g')
pl.xlabel("time (seconds)")
pl.ylabel("Amplitude")

pl.subplot(312)
pl.plot(np.arange(512),waveData[10000:10512],'r')
pl.plot([59,59],[-1,1],'b')
pl.plot([169,169],[-1,1],'b')
print(1/( (169-59)*1.0/framerate ))
pl.xlabel("index in 1 frame")
pl.ylabel("Amplitude")

pl.subplot(313)
pl.plot(np.arange(512),waveData[15000:15512],'g')
pl.xlabel("index in 1 frame")
pl.ylabel("Amplitude")
pl.savefig("pitch.png")
pl.show()
'''
# resize the waveforms to have the same length
min_length = min(len(waveData), len(waveData1))
waveData = waveData[:min_length]
waveData1 = waveData1[:min_length]

print(waveData, waveData1)
# compare the two waveforms
diff = waveData - waveData1
percent_similar = 100.0 - (np.count_nonzero(diff) / len(diff)) * 100.0
print("Similarity between the two voices is {}%".format(percent_similar))

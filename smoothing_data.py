import numpy
from numpy import *
from pylab import *
import h5py
import torch

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y

def smooth_demo():
    #open humann h5 file
    h5_file = h5py.File("/home/yu/PycharmProjects/MotionTransfer-master-Yu-comment/test.h5", 'r')
    # position data
    l_shoulder_pos = h5_file['/group1/l_up_pos'][:]
    r_shoulder_pos = h5_file['/group1/r_up_pos'][:]
    l_elbow_pos = h5_file['/group1/l_fr_pos'][:]
    r_elbow_pos = h5_file['/group1/r_fr_pos'][:]
    l_wrist_pos = h5_file['/group1/l_hd_pos'][:]
    r_wrist_pos = h5_file['/group1/r_hd_pos'][:]


    t=linspace(-4,4,100)
    #fetch xn
    xn=numpy.array(torch.tensor(l_wrist_pos).permute(1, 0))[0]



    # l_wrist_pos=torch.tensor([smooth(l_wrist_pos[i], 8, 'hanning') for i in range(3)])
    # l_wrist_pos=l_wrist_pos.permute(1,0)



    ws=31
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    plot(xn)
    for w in windows:
        plot(smooth(xn,8,w))
    l=['original signal']
    l.extend(windows)

    legend(l)
    title("Smoothing a noisy signal")
    show()


if __name__=='__main__':
    smooth_demo()
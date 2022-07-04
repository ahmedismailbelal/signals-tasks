import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

L = 32  # length of signal x[k]
N = 16  # length of signal h[k]
M = 16  # periodicity of periodic convolution


def periodic_summation(x, N):
    "Zero-padding to length N or periodic summation with period N."
    M = len(x)
    rows = int(np.ceil(M/N))

    if (M < int(N*rows)):
        x = np.pad(x, (0, int(N*rows-M)), 'constant')

    x = np.reshape(x, (rows, N))

    return np.sum(x, axis=0)


def periodic_convolve(x, y, P):
    "Periodic convolution of two signals x and y with period P."
    x = periodic_summation(x, P)
    h = periodic_summation(y, P)

    return np.array([np.dot(np.roll(x[::-1], k+1), h) for k in range(P)], float)


# generate signals
x = np.ones(L)
h = sig.triang(N)

# linear convolution
y1 = np.convolve(x, h, 'full')
# periodic convolution
y2 = periodic_convolve(x, h, M)
# linear convolution via periodic convolution
xp = np.append(x, np.zeros(N-1))
hp = np.append(h, np.zeros(L-1))
y3 = periodic_convolve(xp, hp, L+N-1)


def plot_signal(x):
    '''Plots the signals in stem plot.'''
    plt.figure(figsize=(10, 3))
    plt.stem(x)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$y[k]$')
    plt.axis([0, N+L, 0, 1.1*x.max()])

# plot results
plot_signal(x)
plt.title('Signal $x[k]$')
plot_signal(y1)
plt.title('Linear convolution')
plot_signal(y2)
plt.title('Periodic convolution with period M = %d' % M)
plot_signal(y3)
plt.title('Linear convolution by periodic convolution');


L = 16  # length of signal x[k]
N = 16  # length of signal h[k]
M = N+L-1

# generate signals
x = np.ones(L)
h = sig.triang(N)

# linear convolution
y1 = np.convolve(x, h, 'full')
# fast convolution
y2 = np.fft.ifft(np.fft.fft(x, M) * np.fft.fft(h, M))

plt.figure(figsize=(10, 6))
plt.subplot(211)
plt.stem(y1)
plt.xlabel(r'$k$')
plt.ylabel(r'$y[k] = x_L[k] * h_N[k]$')
plt.title('Result of linear convolution')

plt.subplot(212)
plt.stem(y1)
plt.xlabel(r'$k$')
plt.ylabel(r'$y[k] = x_L[k] * h_N[k]$')
plt.title('Result of fast convolution')
plt.tight_layout()
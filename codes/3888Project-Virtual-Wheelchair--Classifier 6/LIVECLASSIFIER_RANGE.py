import matplotlib.pyplot as plt
import numpy as np
import wave as we
from scipy import signal
from scipy.signal import butter, lfilter
import time

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=6):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


with open("training_set_2.csv") as f:
    x = f.read().splitlines()
frequency = 10000
fs = frequency
length = len(x)
indices = np.arange(0,length,1)                  #TAKES A PORTION OF THE DATA
TAKEN = np.take(x, indices)

string = ",".join(TAKEN)                        #PUTS INTO ONE STRING SEPARATED BY A COMMA BUT STILL WITH SPACES
stringnew = string.replace(" ", "")             #REMOVES SPACES
list = stringnew.split(',')                     #SPLITS SUCH THAT IT IS A LIST WITH STRINGS EG ['796.0', '742.0' ...]
arr = np.array(list)
arr = arr.astype(np.float)                      #TURNS INTO AN ARRAY OF FLOATS

filtered = moving_average(arr,1000)            #MOVING AVERAGE DATA
filtlength = len(filtered)

indices2 = np.arange(1,filtlength+1,1)#UPDATE INDICES FOR MOVING AVERAGE
T = indices2 / frequency

fc = 15  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(5, w, 'low')
output= signal.filtfilt(b, a, filtered)    #BUTTER FILTER LOW ELIMINATE NOISE
# output = output_nonorm / np.max(output_nonorm)
# fig = plt.figure(figsize=(50, 30))             #PLOT MOVING AVERAGE
# plt.subplot(211)
# plt.plot(time,filtered)


TWindow = 1.2    # observation interval given in seconds
TMax = np.max(T)  # maximum time recorded
dT = T[1]-T[0] # dT
NWindow = int(round(TWindow/dT))   # number of points in Window
VALUELIST = []
TIMELIST = []
# i = 1
# highthreshold = 535
# lowthreshold = 480

# fig = plt.figure()
# ax1 = fig.add_subplot(111)       #UNCOMMENT THIS AS WELL AS THE AREA IN THE LOOP IF YOU WANT TO SEE A LIVE PLOT
# plt.ion() #Blocks the figure
# fig.show()

for k in range(0,int(round(TMax/TWindow))): #for loop with how many windows fit in time in this case its 240 (#2)

    if k == 0:
        noise_level = np.mean(output[0:NWindow])
        threshold_level = 1.05 * noise_level    #arbitrary value I picked
        thresh=threshold_level - noise_level


    window_index_beginning = 1+NWindow*(k-1)
    window_index_end = NWindow+NWindow*(k-1)
    window_index_range = range(window_index_beginning, window_index_end)        #MAIN WINDOW PARAMETERS
    indices = np.arange(window_index_beginning, window_index_end+1, 1)

    Tlil = indices / frequency                                                  #TIME IN WINDOW
    array = np.take(output,indices)                                             #VALUES IN WINDOW


    maxwindow = np.max(array)
    maxindi = np.where(array == maxwindow)
    maxT = Tlil[maxindi[0]]   #GIVES TIME FOR MAX VALUE
    minwindow = np.min(array)
    window_val_range = maxwindow - minwindow

    if window_val_range > thresh*1.5:
        print("got window",k)
        array2 = array - noise_level
        maxwindow = abs(np.max(array2)) #can find abs value of greatest deviation from noiselevel
        maxindi = np.where(array2 == maxwindow)
        maxT = Tlil[maxindi[0]]   #where abs(max) occurs in unit time
        indexT = Tlil*frequency
        # print(indexT)

        new_window_index_beginning = maxindi[0] - NWindow
        new_window_index_end = maxindi[0] + NWindow
        new_window_index_range = range(window_index_beginning, window_index_end)  # WINDOW PARAMETERS
        new_indices = np.arange(window_index_beginning, window_index_end + 1, 1)

        new_T = new_indices / frequency
        new_array = np.take(output, new_indices)
        print(new_T)  #SHOWS NEW WINDOW IN SECONDS



    # ax1.plot(Tlil, array)
    # # axes = plt.gca()
    # # axes.set_xlim([0, 120])
    # # axes.set_ylim([0.8, 1])
    # ax1.set_xlabel('time [s]')
    # ax1.set_ylabel('signal [a.u.]')
    # fig.canvas.draw()
    # # NEED A VALUE VECTOR? AND A TIME VECTOR
    #
    # time.sleep(2)

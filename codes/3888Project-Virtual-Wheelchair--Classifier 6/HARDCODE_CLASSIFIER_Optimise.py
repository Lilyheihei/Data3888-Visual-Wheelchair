import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
import numpy as np
import wave as we
from scipy import signal
from scipy.signal import butter, lfilter
import time
import pickle
from pynput.keyboard import Key, Controller
from sklearn.metrics import confusion_matrix
with open('pickle_model_knn2.pkl', 'rb') as f:
   classifier = pickle.load(f)

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

window_count = 1

TWindow = 2    # observation interval given in seconds
TMax = np.max(T)  # maximum time recorded
dT = T[1]-T[0] # dT
NWindow = int(round(TWindow/dT))   # number of points in Window
adjustwindow = int(round(TWindow/1.5))  #new_centred window adjusted on each side according to +- (1.5TWindow)

fig = plt.figure()
ax1 = fig.add_subplot(111)       #UNCOMMENT THIS AS WELL AS THE AREA IN THE LOOP IF YOU WANT TO SEE A LIVE PLOT
# ax2 = fig.add_subplot(212)
plt.ion() #Blocks the figure
fig.show()
fig.canvas.draw()

WINDOWLIST = []
TIME_EVENT_LIST = []
CLASSIFY_LIST = []      #USED TO PUT CLASSIFIED DATA (left right etc)

for k in range(0,int(round(TMax/TWindow))): #for loop with how many windows fit in time in this case its 240 (#2)

    if k == 0:
        noise_level = np.mean(output[0:NWindow])
        threshold_level = 1.05 * noise_level    #arbitrary value I picked ie above 540 for a baseline 512
        thresh= threshold_level - noise_level


    window_index_beginning = 1+NWindow*(k-1)                                   #SIMULATED LIVE WINDOW
    window_index_end = NWindow+NWindow*(k-1)
    # window_index_range = range(window_index_beginning, window_index_end)
    indices = np.arange(window_index_beginning, window_index_end+1, 1)

    Tlil = indices / frequency                                                  #TIME IN WINDOW
    array = np.take(output,indices)                                             #VALUES IN WINDOW
    maxwindow = np.max(array)
    maxindi = np.where(array == maxwindow)
    maxT = Tlil[maxindi[0]]   #GIVES TIME FOR MAX VALUE
    minwindow = np.min(array)
    window_val_range = maxwindow - minwindow


    if window_val_range > thresh*1.5 and window_val_range < thresh*4:          #CHECKS IF THE VALUE IS ABOVE THRESHOLD BUT NOT WILDLY ABOVE IT
        window_count += 1
        print("got window",window_count)
        array2 = array - noise_level
        abs_ar2 = abs(array2) # find abs value of greatest deviation from noiselevel in simulated live window
        maxwindow_ar2 = np.max(abs_ar2)
        maxindi = np.where(abs_ar2 == maxwindow_ar2)       #WINDOW ARRAY = 20,000 entries for 2s
        indexT = Tlil*frequency
        index_max = np.int(maxT*frequency)

        new_window_index_beginning = index_max - int(round(NWindow/1.5))      #CENTRES NEW WINDOW AROUND GREATEST DEVIATION
        new_window_index_end = index_max + int(round(NWindow/1.5))
        # new_window_index_range = range(window_index_beginning, window_index_end)
        new_indices = np.arange(new_window_index_beginning, new_window_index_end + 1, 1)

        new_T = new_indices / frequency
        T_length = len(new_T)
        new_array = np.take(output, new_indices)
        new_window_max = np.max(new_array)
        new_window_min = np.min(new_array)

        deleted = False
        #STORE WINDOW into a list and append windows that are overlapped
        WINDOWLIST.append(new_T)
        i = len(WINDOWLIST)
        if i==1:                               #FOR FIRST ENTRY JUST ADD ELEMENT TO LIST
            TIME_EVENT_LIST.append(new_T)

        if i > 1:                              #AFTER THAT NEED TO CHECK FOR OVERLAP WITH PREVIOUS WINDOW
            DELAY = True
            b = WINDOWLIST[i-1]    #CURRENT WINDOW
            a = WINDOWLIST[i-2]    #PREVIOUS WINDOW
            COMMON = not set(a).isdisjoint(b)   #CHECKS LAST ARRAY WITH CURRENT ARRAY IF COMMON TIMES RETURNS: TRUE
            NOTCOMMON = set(a).isdisjoint(b)

            if COMMON:
                append = np.concatenate((a, b), axis=0)
                new_T_common = append
                new_T_beginning = new_T_common[0]
                T_length = len(new_T_common)
                new_T_end = new_T_common[T_length - 1]
                TIME_EVENT_LIST.append(new_T_common)
                len_LIST = len(TIME_EVENT_LIST)
                print ("element deleted")
                TIME_EVENT_LIST.pop(len_LIST-2) #DELETES PREVIOUS WINDOW SO CLASSIFIER ONLY CONSIDERS APPENDED WINDOW
                deleted = True


            elif NOTCOMMON:
                 TIME_EVENT_LIST.append(new_T)

            LIST_LEN = len(TIME_EVENT_LIST)
            # if LIST_LEN ==11:
            #     break
            # print("LIST_LEN")
            # print(LIST_LEN)
            new_T_LIST = TIME_EVENT_LIST[LIST_LEN - 2]    #INDEX STARTS AT 0 so minus current length by 1                                                          #MINUS TWO SO IT DOES THE PREVIOUS WHICH INSERTS A DELAY
                                                          #BUT ALSO DOESN'T DO THE LAST ENTRY.
            print("NEW_T_LIST")
            print(new_T_LIST)
            new_T_beginning_LIST = new_T_LIST[0]    #BEGINING OF REDRAWN LIST IN UNITS S
            index_beginning_LIST = int(new_T_beginning_LIST * 10000)   #BEGINNING OF REDRAWN LIST IN INDICES
            T_length = len(new_T_LIST)
            new_T_end_LIST = new_T_LIST[T_length - 1]                  #END OF REDRAWN LIST IN UNITS (S)
            index_end_LIST = int(new_T_end_LIST * 10000)               #END OF REDRAWN LIST IN INDICES
            new_indices_LIST = np.arange(index_beginning_LIST, index_end_LIST + 1, 1)   #MAKES A NEW INDICE LIST USING THESE AS BOUNDS
            new_array_LIST = np.take(output, new_indices_LIST)         #TAKES OUTPUT FROM ARRAY AT THESE INDICES
            new_window_max_LIST = np.max(new_array_LIST)               #MAX VALUE IN LIST
            new_window_min_LIST = np.min(new_array_LIST)               #MIN VALUE IN LIST

            #APPLY CLASSIFIER HERE AS WELL AS THINGS IT NEEDS
            diff = new_window_max_LIST - new_window_min_LIST           #DIFFERENCE BETWEEN MIN AND MAX
            stand_dev = np.sqrt(np.var(new_array_LIST))
            minindi_LIST = np.where(new_array_LIST == new_window_min_LIST) #GIVES INDICIES IN THE WINDOW ONLY
            maxindi_LIST = np.where(new_array_LIST == new_window_max_LIST)
            diff_timeindi = maxindi_LIST[0] - minindi_LIST[0]              #NUMBER OF INDEX POINTS BETWEEN MIN AND MAX
            print("diff-timeindi",diff_timeindi)
            diff_time = diff_timeindi / 10000                              #CONVERT TO TIME BY MULTIPLYING SAMPLING RATE
            print("diff-time", diff_time)
            result = classifier.predict([[stand_dev, diff_time]])
            print("CLASSIFIER______",result)

            #NOVEL CLASSIFIER
            if abs(diff_time) >= 0.4:

                if maxindi_LIST[0] < minindi_LIST[0]:
                    if deleted:
                        pass
                    else:
                        print("HARDCODE____left")
                        CLASSIFY_LIST.append(1)
                        # keyboard.press(Key.left)
                        # keyboard.release(Key.left)

                elif maxindi_LIST[0] > minindi_LIST[0]:
                    if deleted:
                        pass
                    else:
                        print("HARDCODE____right")
                        CLASSIFY_LIST.append(2)
                        # keyboard.press(Key.right)
                        # keyboard.release(Key.right)
            else:
                if deleted:
                    pass
                else:
                    print("HARDCODE_____BLINK OR DOUBLE BLINK")
                    CLASSIFY_LIST.append(3)

            # ax1.clear() #NEED THE PLOT TO BE ONE ITERATION BEHIND


            ax1.plot(T, output)
            ax1.plot(np.array([new_T_beginning_LIST, new_T_beginning_LIST]),
                       np.array([np.min(output), np.max(output)]), 'k')
            ax1.plot(np.array([new_T_end_LIST, new_T_end_LIST]), np.array([np.min(output), np.max(output)]), 'k')
            ax1.set_xlim([0, 120])
            print("----------------------------------")


    fig.canvas.draw()
y_pred = np.asarray(CLASSIFY_LIST)
print(y_pred)
#1 = left
#2 = right
#3 = blink or double blink

# confusionmat = confusion_matrix(y_true,y_pred)
# print(confusionmat)

show(block=True)
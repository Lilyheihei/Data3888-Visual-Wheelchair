import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


with open("training_set_2.csv") as f:
    x = f.read().splitlines()
frequency = 10000
length = len(x)
indices = np.arange(0,length,1)                  #TAKES A PORTION OF THE DATA
TAKEN = np.take(x, indices)

string = ",".join(TAKEN)                        #PUTS INTO ONE STRING SEPARATED BY A COMMA BUT STILL WITH SPACES
stringnew = string.replace(" ", "")             #REMOVES SPACES
list = stringnew.split(',')                     #SPLITS SUCH THAT IT IS A LIST WITH STRINGS EG ['796.0', '742.0' ...]
arr = np.array(list)
arr = arr.astype(np.float)                      #TURNS INTO AN ARRAY OF FLOATS

filtered = moving_average(arr,1000)            #MOVING AVERAGE DATA
print(filtered)
filtlength = len(filtered)

indices2 = np.arange(1,filtlength+1,1)#UPDATE INDICES FOR MOVING AVERAGE
time = indices2 / frequency
fig = plt.figure(figsize=(50, 30))             #PLOT MOVING AVERAGE
plt.plot(time,filtered)
plt.show()

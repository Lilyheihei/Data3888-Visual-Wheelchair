import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import csv

fig, c1 = plt.subplots()
c2 = c1.twinx()
Fs, data =scipy.io.wavfile.read('andrewright.wav')
dt = 1/Fs
Nq= 1/(2*dt)
Length=len(data)
print(Length)
count = 0

with open('TEST.txt', 'rt') as f:
    csv_reader = csv.reader(f, skipinitialspace=True)

    for line in csv_reader:
        print(line)

for x in range(0, 4):
    print(x, data[x], '\n')

    with open('TEST.txt', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

        for x in range(0,Length):
                count = x
                line = data[x]
                spamwriter.writerow([count , ',', line])


#for row in data:
      #csv_writer.writerow(row)



tx= np.fft.fft(data)
itx = np.fft.ifft(tx)

c1.plot(itx, color="orange")
c1.plot(data)
plt.show()










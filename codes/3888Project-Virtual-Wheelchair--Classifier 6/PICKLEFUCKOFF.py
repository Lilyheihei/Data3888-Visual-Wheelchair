import pickle
import numpy as np
#
with open('pickle_model1.pkl', 'rb') as f:
   data = pickle.load(f)

print(data)
x = np.array([[12,1.8]])
result = data.predict(x)
print(result)


##ENTRY 1 = MEAN
##ENTRY 2 = STANDARD DEVIATION
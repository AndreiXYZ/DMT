import matplotlib.pyplot as plt
import pickle
import numpy

arima = pickle.load( open( "mse_all_arima.pkl", "rb" ) )
svm = pickle.load( open( "mse_svm_per_patient_ascending.pkl", "rb" ) )
base = pickle.load( open( "mse_base.pkl", "rb" ) )
print(len(arima))
print(len(svm))
print(len(base))
plt.plot(arima,'r',label='ARIMA')
del svm[19]
del svm[17]
plt.plot(svm,'b',label='SVR')
del base[19]
del base[17]
plt.plot([i[0] for i in base],'g',label='Baseline')
plt.xlabel('patient')
plt.ylabel('MSE')
plt.legend()
plt.grid()
plt.show()
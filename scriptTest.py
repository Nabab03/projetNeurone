import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = ROOT_DIR + "/res/"

print("res")
RESKNN=RES+'res.txt'

print('KNN')
os.system('echo KNN>'+RESKNN)
os.system('echo pixSize=5    ratioTest=0.8 Pretraitement band=100 weight >>'+RESKNN)
os.system('python projKNN.py 5 0.8 100 weight>>'+RESKNN)

print('RF')
os.system('echo RF>>'+RESKNN)
os.system('echo pixSize=5    ratioTest=0.8 Pretraitement band=100 weight >>'+RESKNN)
os.system('python projRF.py 5 0.8 100 weight>>'+RESKNN)

print('SVM')
os.system('echo SVM>>'+RESKNN)
os.system('echo pixSize=5    ratioTest=0.8 Pretraitement band=100 weight >>'+RESKNN)
os.system('python projSVM.py 5 0.8 100 weight>>'+RESKNN)

print('NB')
os.system('echo NB>>'+RESKNN)
os.system('echo pixSize=5    ratioTest=0.8 Pretraitement band=100 weight >>'+RESKNN)
os.system('python projNB.py 5 0.8 100 weight>>'+RESKNN)

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
RES = ROOT_DIR + "/res/"
print(RES)

print("KNN")
RESKNN=RES+'KNN.txt'
os.system('echo pixSize=3    ratioTest=0.7  PT>'+RESKNN)
os.system('python projKNN.py 3 0.7 pt>>'+RESKNN)

os.system('echo pixSize=3    ratioTest=0.7  None>>'+RESKNN)
os.system('python projKNN.py 3 0.7 none>>'+RESKNN)



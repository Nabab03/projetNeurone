import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
RES = ROOT_DIR + "/res/"
print(RES)

print("RF")

p1 = ['3', '5', 'pixel']
p2 = ['none', '0.5', '0.7', '0.8', '0.9']
p3 = ['50', '100', 'none']
p4 = ['none', 'rdPix', 'block', 'weight']

RESRF=RES+'RF.txt'

os.system(' echo RANDOM FOREST >'+RESRF)
#pixSize=5 ratioTest=0.8 Pretraitement band=100 weight
for i in range(len(p1)) :
    os.system('echo RANDOM FOREST' +p1[i]+' 0.8 100 weight >>'+RESRF)
    com='python projRF.py '+p1[i]+' 0.8 100 weight >>'+RESRF
    os.system(com)
for j in range(len(p2)) :
    os.system('echo RANDOM FOREST 5 '+p2[j]+' 100 weight >>'+RESRF)
    com='python projRF.py  5 '+p2[j]+' 100 weight >>'+RESRF
    os.system(com)
for l in range(len(p3)) :
    os.system('echo RANDOM FOREST 5 0.8 '+p3[l]+' weight >>'+RESRF)
    com='python projRF.py  5 0.8 '+p3[l]+' weight >>'+RESRF
    os.system(com)
for m in range(len(p4)) :
    os.system('echo RANDOM FOREST 5 0.8 100 '+p4[m]+' >>'+RESRF)
    com='python projRF.py 5 0.8 100 '+p4[m]+' >>'+RESRF
    os.system(com)
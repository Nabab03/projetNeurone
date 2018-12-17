import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RES = ROOT_DIR + "/res/"

p2 = ['0.7','0.9', ]
p4 = ['rdPix', 'block', 'weight']

RESRF=RES+'OF2.txt'

os.system(' echo RANDOM FOREST >'+RESRF)
#pixSize=5 ratioTest=0.8 Pretraitement band=100 weight

for j in range(len(p2)) :
    for m in range(len(p4)) :
        os.system('echo RANDOM FOREST 5 '+p2[j]+' 100 '+p4[m]+' >>'+RESRF)
        com='python projRF.py  5 '+p2[j]+' 100 '+p4[m]+' >>'+RESRF
        os.system(com)


import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
RES = ROOT_DIR + "/res/"
print(RES)

print("SVM")

p1 = ['3', '5', 'pixel']
p2 = ['none', '0.5', '0.7', '0.8', '0.9']
p3 = ['none', 'pt']
p4 = ['50', '100', '200']
p5 = ['none', 'rdPix', 'block', 'weight']

RESSVM=RES+'SVM.txt'

os.system('Voici le script de SVM >'+RESSVM)

for i in range(len(p1)) :
    for j in range(len(p2)) :
        for k in range(len(p3)) :
            for l in range(len(p4)) :
                for m in range(len(p5)) :
                    com='python projSVM.py'+p1[i]+' '+p2[j]+' '+p3[k]+' '+p4[l]+' '+p5[m]+'>>'
                    os.system(com+RESSVM)
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
RES = ROOT_DIR + "/res/"
print(RES)

print("NB")

p1 = ['3', '5', 'pixel']
p2 = ['none', '0.5', '0.7', '0.8', '0.9']
p3 = ['none', 'pt']
p4 = ['50', '100', '200']
p5 = ['none', 'rdPix', 'block', 'weight']

RESNB=RES+'test.txt'

os.system('echo NB >'+RESNB)

com='python projNB.py 3 0.5 pt 100 rdPix>>'
os.system(com+RESNB)



# for i in range(len(p1)) :
#     for j in range(len(p2)) :
#         for k in range(len(p3)) :
#             for l in range(len(p4)) :
#                 for m in range(len(p5)) :
#                     com='python projNB.py '+p1[i]+' '+p2[j]+' '+p3[k]+' '+p4[l]+' '+p5[m]+'>>'
                    # os.system(com+RESNB)
# Picture-classification Indian Pines

Projet de classification d'occupation des sols

## Dossiers et Fichiers:

**Dossier datas :**
Les données sont contenus dans `datas/`
<ul>
<li>Indian_pines_corrected.mat</li>
Contient l'image à traiter (145x145x200)
<li>train_data.npy</li>
Contient les labels d'entrainement
<li>test_data.npy</li>
Contient les labels de test
</ul>

**Classifieurs :**
Nous avons mis en place 4 classifieurs :
<ul>
<li>K plus proche voisins (projKNN.py)
</li>
<li>Naïve Bayes (projNB.py)
</li>
<li>Random Forest (projRF.py)
</li>
<li>Machine à support de vecteurs (projSVM.py)</li>
</ul>

**Scripts :**

- *scriptOverfit.py* : extrait les résultats du classifieur le plus efficace (projRF.py) pour chaque méthodes de split des labels (splitFonction) sur des proportions similaires à la distribution originale. Sert à tester si une méthode de split induit de l'overfit.
- *scriptRF.py* : Teste l'impact de chaque paramètre par rapport au jeu de paramètres optimal.

**Librairie :**
Le fichier libIP.py est une librairie contenant un certain nombre de fonction nécessaires à tous les classifieurs
##Paramètres :
Chacun des classifieurs nécessite 4 paramètres :
Dans l'ordre :

    patch :  3/5/pixel  
    splitProp: none/0.5/0.7/0.8/0.9
    pca : none/50/100  
    splitFonction : none/rdPix/block/weight

Le paramètre **patch** détermine quel format de données l'on envoi au classifieur pour chaque pixel.
&nbsp;&nbsp;&nbsp;3 = fenêtre de 3x3 des voisins d'un pixel sur toutes les couches et son label associé. (3,3,200)
&nbsp;&nbsp;&nbsp;5 = fenêtre de 5x5 des voisins d'un pixel sur toutes les couches et son label associé. (5,5,200)
&nbsp;&nbsp;&nbsp; pixel= pixel seul et son label associé (200).

Le paramètre **splitProp** détermine si et quelle méthode est utilisée pour redécouper Y_train et Y_test
&nbsp;&nbsp;&nbsp;none = Split original
&nbsp;&nbsp;&nbsp;0.5 = Split similaire à l'original (~50/50). Utilisé pour tester le potentiel overfit inhérent au nouveau split.
&nbsp;&nbsp;&nbsp;0.7 = Split 70/30
&nbsp;&nbsp;&nbsp;0.8 = Split 80/20
&nbsp;&nbsp;&nbsp;0.9 = Split 90/10

Le paramètre **pca** détermine le nombre de spectres que l'on conservera. Un algorithme de projection orthogonale (pca) est utilisé pour fusionner les couches porteuses de la même information.
&nbsp;&nbsp;&nbsp;none = On conserve les 200 couches
&nbsp;&nbsp;&nbsp;50 = On fusionne la donnée sur 50 couches
&nbsp;&nbsp;&nbsp;100= On fusionne la donnée sur 100 couches

Le paramètre **splitFonction** détermine la fonction de resplit des Labels utilisée.
&nbsp;&nbsp;&nbsp;none = Pas de resplit 
&nbsp;&nbsp;&nbsp;rdPix = on insère aléatoirement splitProp% des labels dans Y_train, le reste dans Y_test
&nbsp;&nbsp;&nbsp;block = on insère aléatoirement splitProp% des blocks existants dans Y_train, le reste dans Y_test
&nbsp;&nbsp;&nbsp;weight = on insère splitProp% de chaque label dans Y_train, le reste dans Y_test

## Installation :
Il est nécessaire de faire un certain nombre d'installations avant de pouvoir exécuter les programmes.
        
    virtualenv tp_eca_2018 -p python3
    pip install numpy matplotlib scikit-learn tensorflow geopandas

## Utilisation :
Avant exécution il est nécessaire de lancer la machine virtuelle python3

    cd tp_eca_2018
    source bin/activate

Il existe deux manières d'exécuter :

 1. Lancer directement un classifieur avec son jeu de paramètre
    `python profRF.py 5 0.8 100 weigh`

 2. Exécuter un des scripts 
    `python scriptOverfit.py`
    Les résultats seront consignés dans     `res/`

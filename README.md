
# review-classification
Classification d'occupation du sol

## Dossiers et Fichiers:

**Dossier data :** tout les datasets, dataset challenge + dataset original + dataset lematisé + dataset lematisé et tagé  

**Dossier dataset :**  données d’apprentissage, chaque sous dossier représente une classe(label)



**Scripts :** 

- *build_package.py* : installe les packages nécessaires à l'exacution des programmes

- *build_dataset.py* : construit le dossier dataset à partir d'un fichier dataset quelconque (ex: *datasetLematisé.csv*) et *label.csv*
- *delete_dataset.py* : détruit la construction d'avant (pratique si on veux tester avec un autre corpus)
- *DataDirtyToLemmatised* : prend *dataset.csv* (dataset original) ---> et renvoie *data/lemmatised_dataset.csv*


## Utilisation :
*Remarque* : j'ai marqué dans le fichier .gitignore les deux deux dossiers Pos et Neg de *dataset/* à cause du nombre de fichiers important dedans, ces deux dossiers vont être ignorés par vos push/pull donc faut construire le dossier d'apprentissage après chaque git pull.

 1. Construire le dossier d’apprentissage avec *scripts/build_dataset.py* (par défault il le construit à partir du dataset orignal dataset.csv) si vous voulez utilisez un autre corpus faut modifier cette ligne  `DATA_REVIEW = DATA +  "/ACHANGER.csv` puis run `python scripts/build_dataset.py`
 2. Maintenant que le dossier d’apprentissage est prêt on peut entraîner des models easy `python NB.py` ...

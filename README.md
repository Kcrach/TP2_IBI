# TP2_IBI
TP d'apprentissage profond par renforcement.

# Modalité d'installation et d'éxécution
Nous avons utilisé Pipenv pour générer un environnement virtuel. Pour installer les dépendances, il faut donc faire la commande suivante "pip install -r requirements.txt". Cepedant, pour vizdoogym, nous avons dût modifier directement le code du fichier vizdoomenv.py de notre environnement virtuel en suivant le pull-request sur le git de Vizdoom. 

Concernant l'éxécution, il y a 2 modes d'éxécution, le mode test et le mode apprentissage. Pour passer de l'un à l'autre il suffit de modifier la valeur du booléen test_mode dans le main.py. Si test_mode est à false, l'agent est en mode apprentissage sinon, il est en mode test. Tous les hyperparamètres se trouvent en haut du fichier main.py.

# Fichier/Dossier et leurs contenus
- Le dossier net contient la sauvegarde des différents réseaux pour chaque environnement. Par exemple "net/CartPole-v1.pt" contient la sauvegarde du réseau de neurone relatif à l'environnement CartPole-v1. Ces sauvegardes sont utilisés pour le mode test.
- AgentRandom.py contient la classe RandomAgent qui correspond à l'agent de la partie 2.
- AgentVizdoom.py contient la classe VizdoomAgent qui correspond à l'agent de la partie 3.
- main.py contient le programme principal, avec la boucle d'apprentissage (ou test), l'assignation des hyperparamètres et la génération des environnements.
- MemoryReplay.py contient la classe Memory qui correspond au buffer Replay.
- Network contient le code des 2 réseaux de neurones utilisés : Network qui est le réseau fully connected et CNN qui est le réseau convolutionnel.
- Requirements.txt et Pipfile sont les fichiers utiles pour pipenv.

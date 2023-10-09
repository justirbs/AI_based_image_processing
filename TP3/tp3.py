# TP3 - Segmentation des images

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import random

# Segmentation d’images par seuillage d’histogramme (Binarisation)

# Calculer l’histogramme de l’image « pepper.bmp » proposée
image_pepper = np.array(Image.open('img/pepper.bmp').convert('L'))

# Calculez l'histogramme de l'image en niveaux de gris
histogram = np.histogram(image_pepper, bins=256, range=(0, 256))[0]

plt.bar(np.arange(256), histogram, width=1.0)
plt.title("Histogramme de l'image en niveaux de gris")
plt.savefig("img/histogramme_pepper.png")

# Implémenter l’algorithme de seuillage proposé et vérifier la convergence de celui-ci
def algo_seuillage(img, proposition=1):
    # on définit le premier seuil aléatoirement
    seuil = random.randint(0, 255)
    # on calcule un nouveaux seuil jusqu'à la convergence
    while True :

        seuil_old = seuil
        moyenne_inf, moyenne_sup, nb_inf, nb_sup = 0, 0, 0, 0

        # on calcule la moyenne des pixels inférieurs et supérieurs au seuil
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] < seuil:
                    moyenne_inf += img[i][j]
                    nb_inf += 1
                else:
                    moyenne_sup += img[i][j]
                    nb_sup += 1
        if nb_inf != 0:
            moyenne_inf = moyenne_inf / nb_inf
        if nb_sup != 0:
            moyenne_sup = moyenne_sup / nb_sup

        # on calcule le nouveau seuil
        seuil = (moyenne_inf + moyenne_sup) / 2

        # condition de convergence
        if seuil == seuil_old:
            break

    # on construit l'image seuillée
    if proposition == 1:
        S1, S2 = 0, 255
    elif proposition == 2:
        S1, S2 = moyenne_inf, moyenne_sup

    img_seuillee = np.zeros((len(img), len(img[0])))

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < seuil:
                img_seuillee[i][j] = S1
            else:
                img_seuillee[i][j] = S2

    return img_seuillee

# Seuiller l’image de deux manières différentes

# proposition 1
img_seuillee_1 = algo_seuillage(image_pepper, 1)
image1 = Image.fromarray(img_seuillee_1.astype('uint8'))
image1.save('img/pepper_seuillee_1.png')

# proposition 2
img_seuillee_2 = algo_seuillage(image_pepper, 2)
image2 = Image.fromarray(img_seuillee_2.astype('uint8'))
image2.save('img/pepper_seuillee_2.png')

# Segmentation d’images grâce à l’algorithme des K-means

# Calculer	l’histogramme	de	l’image	« pepper.bmp »	proposée.

# Implémenter	l’algorithme	de	k-means	proposé	et	vérifier	la	convergence	de	celui-ci.
def k_means(img, K):

    max_iter = 100
    cpt = 0

    # définir les centres initiaux
    centres = []
    for i in range(K):
        centres.append(random.randint(0, 255))
    # ranger les centres dans l'ordre croissant
    centres.sort()

    # tant que les centres changent
    while True:

        cpt += 1

        centres_old = centres.copy()

        # Affecter pour chaque intensité i allant de 0 à 255 de l’histogramme, la classe ayant la moyenne la plus proche de cette intensité	i
        classes = []
        for i in range(K):
            classes.append([])
        for i in range(len(img)):
            for j in range(len(img[0])):
                distance = 255
                classe = 0
                for k in range(K):
                    if abs(img[i][j] - centres[k]) < distance:
                        distance = abs(img[i][j] - centres[k])
                        classe = k
                classes[classe].append(img[i][j])

        # Calculer les nouvelles moyennes de chaque classe
        for i in range(K):
            if len(classes[i]) != 0:
                centres[i] = sum(classes[i]) / len(classes[i])

        # Condition de convergence
        if centres_old == centres or cpt == max_iter:
            break
    
    # on construit l'image seuillée
    img_seuillee = np.zeros((len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            distance = 255
            classe = 0
            for k in range(K):
                if abs(img[i][j] - centres[k]) < distance:
                    distance = abs(img[i][j] - centres[k])
                    classe = k
            img_seuillee[i][j] = centres[classe]

    return img_seuillee 
    

# Segmenter	l’image	en	affectant	à	chaque	pixel,	la	moyenne	de	la	classe	auquel	il	appartient.
img_seuillee = k_means(image_pepper, 4)

image = Image.fromarray(img_seuillee.astype('uint8'))
image.save('img/pepper_seuillee_K4.png')

# Tester	l’image	pour	des	nombres	de	classe	différents.

# k=2
img_seuillee_k2 = k_means(image_pepper, 2)
image_k2 = Image.fromarray(img_seuillee_k2.astype('uint8'))
image_k2.save('img/pepper_seuillee_K2.png')

# k=3
img_seuillee_k3 = k_means(image_pepper, 3)
image_k3 = Image.fromarray(img_seuillee_k3.astype('uint8'))
image_k3.save('img/pepper_seuillee_K3.png')

# k=4
img_seuillee_k4 = k_means(image_pepper, 4)
image_k4 = Image.fromarray(img_seuillee_k4.astype('uint8'))
image_k4.save('img/pepper_seuillee_K4.png')

# k=5
img_seuillee_k5 = k_means(image_pepper, 5)
image_k5 = Image.fromarray(img_seuillee_k5.astype('uint8'))
image_k5.save('img/pepper_seuillee_K5.png')

# k=6
img_seuillee_k6 = k_means(image_pepper, 6)
image_k6 = Image.fromarray(img_seuillee_k6.astype('uint8'))
image_k6.save('img/pepper_seuillee_K6.png')



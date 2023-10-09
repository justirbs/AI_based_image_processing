# Partie 1. Création des image numériques

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import math
from scipy import ndimage 

# Partie 1(a). Création des images binaire et d’intensités

# 1) Créer un triangle blanc inférieur sur un fond noir de taille 10 x10 pixels (Fig. 1).

n = 10

img = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i > j : img[i, j] = 1 

plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')
plt.savefig('img/fig1.png')

# 2) Crée un triangle blanc en haut dans un fond noir de taille NLxNC (avec NL=NC=100) et un triangle blanc en bas dans un fond noir de taille NL x NC, puis les visualiser (Fig. 2a, 2b)

n = 100

img = np.zeros((n, n))

for i in range(int(n/2)):
    for j in range(int(n/2)):
        if i < j : 
            img[i, j] = 1 
            img[i, n-j-1] = 1 

plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')
plt.savefig('img/triangle1.png')

n = 100

img = np.zeros((n, n))

for i in range(int(n/2)):
    for j in range(int(n/2)):
        if i < j : 
            img[n-i-1, j] = 1 
            img[n-i-1, n-j-1] = 1 

plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')
plt.savefig('img/triangle2.png')

# 3) Créer une image binaire alternée de dimension NxN avec N=100 (les pixels dont le numéro est impair ont la valeur 0, pair la valeur 1). Fig.3

n = 100

img = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        img[i, j] = (i+j) % 2

    
plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')

# 4) Créer une image d’intensités (en niveaux de gris) dégradée en ligne d’un rectangle de taille 10 x 8 (Fig.4)

n = 10
m = 8

img = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        img[i, j] = i

plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')

# 5) Créer une image, de dimension NxN (avec N=50, puis N=100), de niveaux de gris
# progressifs en fonction de position de pixel (la valeur de chaque pixel dépend de la position
# de ce pixel dans la matrice de NxN). Fig. 5
# Attention à la normalisation : les niveaux de gris doivent être entre [0,1] ou bien [0, 255]

n = 100

img = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        img[i, j] = i 

plt.axis('off')  # Désactive les axes
plt.imshow(img, cmap='gray')

# Partie 1(b). Image Couleur

# 6) Création d'image couleur RGB

R = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]])
V = np.array([[255, 0, 0], [0, 0, 255], [0, 255, 0]])
B = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]])

plt.axis('off')  # Désactive les axes
plt.imshow(np.dstack((R,V,B)))

# 7) Création des images couleurs indexées

map = np.array([[255, 255, 0], # jaune
                [255, 0, 128], # violet
                [0, 128, 0], # vert foncé
                [255, 128, 0], # orange
                [255, 255, 255], # blanc
                [255, 0, 0], # rouge
                [0, 0, 255], # bleu
                [0, 255, 0], # vert
                ])

I = np.array([[0, 1, 2], 
              [3, 4, 0], 
              [5, 6, 7]])

plt.axis('off')  # Désactive les axes
plt.imshow(map[I])

# Partie 2. Histogramme des niveaux de gris d’images et transformations d’histogramme

# 1) Écrire les algorithmes du CALCUL, de l’ETIREMENT et de l’EGALISATION de
# l’histogramme des niveaux de gris d’une image de taille NLxNC et de 256 niveaux de
# gris entre 0 (noir) et 255 (blanc).

# Fonction pour calculer l'histogramme d'une image en niveaux de gris
def calculer_histogramme(image):
    histogramme = np.zeros(256, dtype=int)
    for pixel in image.flatten():
        histogramme[pixel] += 1
    return histogramme

# Fonction pour effectuer l'étirement de l'histogramme
def etirer_histogramme(image, nouv_min=0, nouv_max=255):
    min_original = np.min(image)
    max_original = np.max(image)
    image_etiree = (image - min_original) * (nouv_max - nouv_min) / (max_original - min_original) + nouv_min
    return np.round(image_etiree).astype(np.uint8)

# Fonction pour effectuer l'égalisation de l'histogramme
def egaliser_histogramme(image):
    histogramme = calculer_histogramme(image)
    cdf = np.cumsum(histogramme)
    cdf_min = np.min(cdf)
    image_egalisee = ((cdf[image] - cdf_min) / (np.prod(image.shape) - cdf_min) * 255).astype(np.uint8)
    return image_egalisee

image = np.array(Image.open('img/fig1.png').convert('L'))

if image is not None:
    plt.axis('off')  # Désactive les axes
    plt.imshow(image, cmap='gray')

    # Calculez l'histogramme
    histogramme = calculer_histogramme(image)

    # Appliquez l'étirement de l'histogramme (par exemple, étiré entre 50 et 200)
    image_etiree = etirer_histogramme(image, nouv_min=50, nouv_max=200)
    plt.imshow(image_etiree, cmap='gray')

    # Appliquez l'égalisation de l'histogramme
    image_egalisee = egaliser_histogramme(image)
    plt.imshow(image_egalisee, cmap='gray')
else:
    print("Erreur lors du chargement de l'image.")

image = np.array(Image.open('img/histo_imageTest_entree_3D.png').convert('L'))
image_to_show = cv2.imread('img/histo_imageTest_entree_3D.png')
image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)

# Calculez l'histogramme de l'image en niveaux de gris
histogram = np.histogram(image, bins=256, range=(0, 256))[0]

# Calculez l'histogramme cumulé
cumulative_histogram = np.cumsum(histogram)

# Étirez l'histogramme
min_pixel_value = np.min(image)
max_pixel_value = np.max(image)
stretched_image = ((image - min_pixel_value) / (max_pixel_value - min_pixel_value) * 255).astype(np.uint8)

# Calculez l'histogramme de l'image étirée
stretched_histogram = np.histogram(stretched_image, bins=256, range=(0, 256))[0]

# Calculez l'histogramme cumulé de l'image étirée
stretched_cumulative_histogram = np.cumsum(stretched_histogram)

# Égalisez l'histogramme
equalized_image = cv2.equalizeHist(image)

# Calculez l'histogramme de l'image égalisée
equalized_histogram = np.histogram(equalized_image, bins=256, range=(0, 256))[0]

# Calculez l'histogramme cumulé de l'image égalisée
equalized_cumulative_histogram = np.cumsum(equalized_histogram)

image = Image.fromarray(stretched_image.astype('uint8'))
image.save('img/stretched_image.png')

image = Image.fromarray(equalized_image.astype('uint8'))
image.save('img/equalized_image.png')


equalized_image_to_show = cv2.imread('img/equalized_image.png')
equalized_image_to_show = cv2.cvtColor(equalized_image_to_show, cv2.COLOR_BGR2RGB)


# Partie 3. Réduction de bruit (Débruitage)

image_originale = cv2.imread('img/lena.jpg', cv2.IMREAD_GRAYSCALE)  # Charger l'image en niveaux de gris

# Ajouter du bruit gaussien
mean = 0
variance = 0.04 * np.max(image_originale)  # Variance pour un bruit gaussien de 4%
bruit_gaussien = np.random.normal(mean, np.sqrt(variance), image_originale.shape).astype(np.uint8)
image_bruitee_gaussienne = cv2.add(image_originale, bruit_gaussien)

# Ajouter du bruit salt & pepper
bruit_salt_pepper = np.random.choice([0, 255], size=image_originale.shape, p=[0.98, 0.02]).astype(np.uint8)
image_bruitee_salt_pepper = cv2.add(image_originale, bruit_salt_pepper)


# Définir le noyau du filtre moyenneur 5x5
kernel = np.ones((5, 5), np.float32) / 25

# Appliquer le filtre moyenneur sur l'image bruitée avec bruit gaussien
image_filtree_gaussienne = cv2.filter2D(image_bruitee_gaussienne, -1, kernel)

# Appliquer le filtre moyenneur sur l'image bruitée avec bruit salt & pepper
image_filtree_salt_pepper = cv2.filter2D(image_bruitee_salt_pepper, -1, kernel)



# Appliquer le filtre médian sur l'image bruitée avec bruit gaussien
image_filtree_median_gaussienne = cv2.medianBlur(image_bruitee_gaussienne, 5)  # Taille du noyau : 5x5

# Appliquer le filtre médian sur l'image bruitée avec bruit salt & pepper
image_filtree_median_salt_pepper = cv2.medianBlur(image_bruitee_salt_pepper, 5)  # Taille du noyau : 5x5
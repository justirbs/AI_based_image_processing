# AI based Image Processing

## Image grayscale histogram and histogram transformations

The following exemple is provided in the file `histogram_transformations.ipynb`:

<div style="text-align: center;">
  <img src="img/1_plot_original.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/1_plot_stretched.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/1_plot_equalized.jpg" style="display: block; margin: auto; width: 90%;">
</div>

The images above show the grayscale histogram of an image and the result of applying two histogram transformations: histogram stretching and histogram equalization.

The first image is the original image, the second image is the image after applying the histogram stretching transformation and the third image is the image after applying the histogram equalization transformation.

## Noise reduction using image filtering

The following exemple is provided in the file `noise_reduction.ipynb`:

<div style="text-align: center;">
  <img src="img/2_plot_median_filter.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/2_plot_noisy_images_lena.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/2_plot_median_filter_lena.jpg" style="display: block; margin: auto; width: 90%;">
</div>

## Detection of edges in images

The following exemple is provided in the file `characteristics_detection.ipynb`:

<div style="text-align: center;">
  <img src="img/3_plot_image_contours_filtre.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/3_plot_image_contours_gradient.jpg" style="display: block; margin: auto; width: 90%;">
  <img src="img/3_plot_image_contours_laplacian.jpg" style="display: block; margin: auto; width: 90%;">
</div>

Images above show the result of applying edge detection filters to an image. The first image is the original image is the result of applying Sobel filter, the second image is the result of applying the gradient filter and the third image is the result of applying the Laplacian filter.

## Image segmentation using clustering

The following exemple is provided in the file `image_segmentation.ipynb`:

<div style="text-align: center;">
  <img src="img/4_binary_thresholding.jpg" style="display: block; margin: auto; width: 90%;">
</div>

<div>
<img src="img/4_k_means.jpg" style="display: block; margin: auto; width: 90%;">
</div>

Images above show the result of applying image segmentation techniques to an image. The first image is the result of applying binary thresholding and the second image is the result of applying the k-means clustering algorithm.
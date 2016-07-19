# clothing-extraction

## High-level Description

When asked to solve this clothing vs. non-clothing segmentation problem, I had two general ideas of different perspectives in my mind. One follows traditional computer vision techniques with an emphasis on image processing to perform the task of binary segmentation. And the other is more of a learning-based approach, parsing the image with a neural network say. Unfortunately, I am not given a large training set with clothing annotations. So I decided to go for the first option.

I started out with k-mean clustering on the RGB value of image pixels. The hope was that the algorithm can discover for itself that there are 4 categories in the image, namely skin, hair, clothing and background, but apparently this is not very likely to happen in practice. I tried to initialize the cluster centers with some hand-picked values to help the clustering algorithm. Improvements were observed but generally the results are very noisy, since we do not impose spatial constraints, such as spatial smoothness. Moreover, since k-means clustering is an unsupervised technique, we have to determine which cluster is the clothing, the face, the hair and the background as a final step. I didn't further pursue this method.

Another obvious choice for foreground/background segmentation (the choice we use) is the famous GrabCut algorithm. This algorithm iteratively minimizes. User is required only to provide foreground/background seeds for the algorithm to begin with. The next question is straightforward: how do we get the seeds? This time, face, hair and background are all considered as the background class. We can detect face, obtain a histogram of the face's hue and saturation channels, and use the histogram to detect skin (such as hands and legs). Face and skin are provided to the GrabCut algorithm as hard background. And we also use 1/6 of the image on both left and right side as hard background. If there is no face in the image, we use a histogram from other images.

## Usage

1. Compile main.cpp (make sure using OpenCV 2.4)
2. Run the executable with arguments "--image_list /path/to/image_list.txt --output_folder /path/to/output/folder"

## Parallelization 

Parallelization can be easily obtained with both the OpenMP API and the MPI API. I have included both.

1. For OpenMP, we have to include omp.h. Function omp_set_num_threads() sets the number of threads to be used on a specific computing cluster.
2. For MPI, we use the MPICH implementation. We can easily parallelize across different computing clusters with MPICH.

For example, if we have 12 computing clusters and use 16 cores per cluster. Theoretically, we can process 12x16 = 192 images simultaneously.

I did not test the parallelization part, because my access to university clusters has been disabled. But using both OpenMO and MPI is what we used to always do for our research projects. :)

## Observations

The GrabCut algorithm depends on good foreground/background seeds for good results. Unfortunately, there are huge variations in different photos. The variations include how much area the model occupies the image. Thus our prior for determining hard background does not always work.

## Possible Improvements

From the results, it is easy to see that we suffer mostly from under-segmentation, meaning that we still have some skin and background. Possible ways of improving include:

1. Deal with background and skin individually rather than putting them into one big category of background.
2. Have a better skin histogram model.

## Time Spent

4 - 5 hours.

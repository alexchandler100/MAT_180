# COLORIZED - Colorization Of Lost Or Ruined ImageZ Employing Diffusion

Clément Weinreich

Brennan Whitfield

Vivek Shome

## Abstract
  
### How will the data be collected?

The data will be collected from a public dataset on Kaggle named [Image Colorization]([https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code](https://www.kaggle.com/datasets/shravankumar9892/image-colorization/code)). This dataset consists of 50 thousand grayscale and colored images rescaled to 224 x 224.  

### What task do you want to accomplish with the data?

We seek to recolor grayscale photographs from the University Archives Photographs at Shields Library. This project could provide the library with a means of viewing a local landmark/area from the perspective of an individual present at the original time.

### What kind of learning algorithm do you propose using to accomplish this task?

We plan to accomplish this task using an image-to-image conditional diffusion model of the form $P(y | x)$, where $x$ is a grayscale image and $y$ is a color image.  

### How will you measure your performance of the task?

To measure the performance of the task, we will use the [Fréchet Inception Distance]([https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)) (FID). The FID metric compares the distribution of generated images with the distribution of a set of real images (ground truth). The FID metric was introduced in 2017 and is the current standard metric for assessing the quality of generative models.

## Instructions (delete this once finished)
Your README.md file (located at the root of your project directory) should completely describe the project, what it accomplishes, how well it performs, and walk through how a new user can use your model on their own data after cloning the repository. This means, by following instructions in your README.md file, I should easily be able to run your algorithm on a new example not in your dataset and see that the model accomplishes what it promised to accomplish (at least up to your claimed performance measure).

## Introduction

outline this file? Merge with abstract?

## Theory

Add all the maths derivations with all steps

## Implementation

How we implemented it

## Results

what it accomplishes, how well it performs

## How to use our model

Walk through how a new user can use your model on their own data after cloning the repository

suggestion: create a notebook where the user can give an image, and the code to preprocess the image, run the model and everything is already included in it?

## Conclusion

feedback, improvements,...

## References


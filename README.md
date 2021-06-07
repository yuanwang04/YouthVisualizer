# Youth Visualizer - Visualizing Convolutional Network on CelebA Dataset

Yuan Wang & Jiajie Shi, 2021-6

![fig_0](./imgs/fig_0.png)

## Abstract



## Running

Clone the repository. Extract the Celeb_A data set or other photos to the directory following the specification. 

To run the program, use `python main.py`. 

Might need to adjust some parameters, file paths to get the optimal result. 

## Dataset

We used the CelebFaces Attributes Dataset (CelebA) dataset in the training and evaluation. "CelebA is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations." [2] 

We trained our network on the label "Young" in order to get some face attribute filters that are related to the youth of a face.

## Model and Training

We used the Darknet64 as our initial model. The model has 5 layers.

The training code is adapted from the tutorials of UW CSE 455 [3]. 

Training statistics:

```
{
	'epoch':
	'schedule':
	'learning_rate': 0.01, 
	'momentum': 0.9, 
	'decay': 0.0005
}
```

## Visualization

#### activated area



#### activate a filter



#### what is youth



## Takeaways



## References

1. Deep Dream:

   Wasilewska, A. W. (n.d.). *Google Deep Dream*. Stonybrook.Edu. Retrieved June 7, 2021, from https://www3.cs.stonybrook.edu/~cse352/T12talk.pdf

2. CelebA Dataset:

   ```
   @inproceedings{liu2015faceattributes,
    title = {Deep Learning Face Attributes in the Wild},
    author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
    booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015} 
   }
   ```

3. Training Code:

   Redmon, J. (n.d.). *pjreddie/uwimg*. GitHub. https://github.com/pjreddie/uwimg. 


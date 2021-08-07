# Dataset

In the `dataset` folder, open two different folders as `train` and `test`. 
For your training data, first open a folder for each class in the train folder, then put the images of each class in its own folder.
Do the same for your test data inside the test folder. 

In a case where you have two classes named `cat` and `dog`, 
the folder structure should be like this:

```
PyImageClassification
└─── dataset
    ├─── train
    │    ├─── cat   
    │    │    ├─── your_cat_image_01.jpg
    │    │    ├─── your_cat_image_02.jpg
    │    │    └─── ...
    │    │    
    │    └──── dog
    │         ├─── your_dog_image_01.jpg
    │         ├─── your_dog_image_02.jpg
    │         └─── ...
    │
    └─── test 
         ├─── cat   
         │    ├─── your_test_cat_image_01.jpg
         │    ├─── your_test_cat_image_02.jpg
         │    └─── ...
         │
         └─── dog
              ├─── your_test_dog_image_01.jpg
              ├─── your_test_dog_image_02.jpg
              └─── ...

```
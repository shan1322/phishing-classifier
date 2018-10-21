<<<<<<< HEAD
# Url-classifiction-using-lexical-features-only
=======
* Lexical features-Features only related to url string 

## Case1-Using Hand Enginered features and Dense Neural Network
* Hand engineered Features
    1. length of URL
    2. Digits in URL
    3. Dots in URL
    
        *features and labels are stored /matpre folder
    ##### To extract lexical features
    
    ```python featureextraction.py```
 * Dense neural networks was used to classification 
    ##### To train dense neural network
    ```python featraindensepy.py```
    ##### To test dense neural network
    ```python featestdensepy.py```

## Case2 - word embeddings
##### To make word embeddings
  ```python embedding.py``` 
  * embeddings and labels are stored /matpre folder
  
##### To train dense neural network
``` python wordlevelcnn.py```

##### To test dense neural network
   ```python cnntest.py```
   
   
[refernce-URLNET](https://arxiv.org/abs/1802.03162)
   
 



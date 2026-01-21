## Motivation 
- Lack of labels led to an idea of using natural language (words) as guiding, and eventually reach the label prediction in natural language 
- images are described in text

## VirTex 
- Convolution to break up the image first, then input text descriptions using transformer to predict the exact caption/words
- predicting exact captions 

## COnVIRT 
- contrastive learning to increase cosine similarity between image and text pairs 
- learns the relationship between pixel and text-level, adds semantic relationship instead of nothing in VirTex which didn't have any cross comparisons
    - doesn't need to predict the exact words

## CLIP 
- break the images and text into batches (through tokenization)
- find the image and text batch that are most similar
- reduce the similarity with every other (n-1)(n) pairs 
- this similarity comparison is alike the self-attention matrix of $Q*K^T$ in transformer model, the key of text looks at the queries to determine which words are the most similar. 
- We see where in the image matches with which words the most. We expect the highest attention on a dog's ear to match with the words "ears are long" if the tokens for these images and texts are not compressed into a single vector 
    - transform into one embedding vector 
- This attention matrix becomes the training objective, to minimize the value between the correct image and other non-correct text pair 
    - **The ground truth is assumed the diagonal**. This is because each diagonal contains the matching text and images 
    - take Symmetric Cross Entropy (acorss row of image embedding) and across column of text embedding 

## Features of CLIP 
- preserved invariance both in text and image processing. Because transformer's self attention works regardless of the position of the pixel and of the data batch order. Invariance is preserved from images automatically 



## Issues 
- high computational cost (large parameters), and the relationships are mainly learned through self attention and in high dimensional spaces, similar vectors are have similar semantic meaning/relationships (same intuition behind Word2Vec's semantic relationship)
- Curse of dimensionality: requires significant amount of data to train because of the parameters, but what may really influence the predictions are the "clusters" of similar meaning 
- how much information does the model actually look at each image and text? is it possible for the model to have correlated relationships such as certain pixel order/values with the word "yellow", "red", etc.  Which led to bias in the training data. It doesn't have enough local and spatial understanding during the training process 
- batch size dependence. the model training performance can still be influenced by the data ordering. Since contrastive loss compares pair-wise with all the other samples in the batch, how relevant/good those samples are inside the batch may make a difference. How many samples inside a batch will also influence training. The higher batch size, the more the model learns to distinguish between each other. It may not maintain the independent and identitically distributed hypothesis behind data shuffling 
- noisy data 
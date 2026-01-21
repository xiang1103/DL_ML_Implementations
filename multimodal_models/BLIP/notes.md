## Bootstrapping Language Image Pre-training Features
- CLIP can't generate things easily, it's designed as an encoder model where the model only learns to get the semantic closeness of pairings (image and text). 
- BLIP allows easier generation of text (namely image captioning) with modifications to the architecture and loss functions 


## Loss functions 
- original CLIP loss 
- image to text matching loss (binary classifcation) 
- language generation cost 

## Features 
- the original CLIP only uses self attention to obtain the important features within each encoder and encoding, so it doesn't understand the relationships with text until the last step of contrastive loss. The contrastive loss is kind of a loss function of self attention/cosine similarity. If the model needs to learn the connection between word and images more, it needs to have cross attention 
- contrastive loss only gives model the ability to distinguish closeness/connections, and not others. So Image-text matching loss is used to directly predict the caption correctness. This process is alike classifying fake images in GAN
- the language genrationn cosst is used for actually generating the text. Cross attention is used in both IMT and LM loss so that the model can better understand the connections and expected output between images and texts 
- adding more loss functions are alike giving the model more things to learn and enhacing certain aspects of the model, such as the model's ability to know which image goes with which text 
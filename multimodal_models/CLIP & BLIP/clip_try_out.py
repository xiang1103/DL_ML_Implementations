# COCO dataset https://cocodataset.org/#explore
from PIL import Image
from torchvision import transforms
import json 
from clip import CLIP
from transformers import AutoTokenizer 
import torch 

''' 
process the image 
then encode image and text 

Encoding image into patches and into one layer (ViT): 
    - break into patches using conv layers
    - flatten the patches into sequential tokens for transformer to process
    - more important image tokens would have higher attention (similar with GradCAM heatmap, but we use self attention)
    - CLS token is added as one D-dimensional vector to the picture 
    - the entire output of vision transformer is the CLS token, which embeds the important attention information


Encoding text: 
    - given input text, we need to tokenize the text first and make sure it fits within the context length for matrix multiplication 
    - then does embedding and the transformer layers with causal attention 
    - because of attention, the last EOS token has seen all the words, which contains all the information
    - so it outputs the final EOS token embedding 
    
'''



def process_image(path):
    img= Image.open(path)   # HxWxC 
    width, height= img.size 
    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),  # CxHxW, normalize into [0,1]
        
    ])
    input_img= transform(img).unsqueeze(0)
    return input_img 

def retrieve_captions(path):
    with open('../training_materials/captions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    captions= data[path]

    return captions[0] 

def pad_tokens(text_tokens, context_length): 
    '''  
    pad the input text_tokens to the length of context 
    '''
    length = context_length- len(text_tokens)
    tokens = text_tokens + [0]*length
    return tokens 
    

def main(): 
    img_path = "training_materials/images/bicycle_woman.jpg"
    
    # 3 x 224 x 224 
    # Batch x Channel x H x W 
    processed_img = process_image("../"+img_path)
    caption = retrieve_captions(img_path)
    
    ''' 
    text tokenization 
    ''' 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    ''' 
    CLIP implementation and make sure forward pass is working 
    '''
    embed_dim = 5   # dimension of the entire ViT ouptut (D_e)
    vision_layer = 1    # ViT layers 
    image_resolution = processed_img.shape[1]
    vision_width = 64   # dimension of the conv layers to embed the images, processed inside the ViT
    vision_patch_size = 16     # conv layer/window size 

    context_length = 12 # total size of inputs + outputs (N), limits our model input size 
    vocab_size = len(tokenizer)
    transformer_width = 6  # width within the transformer for processing the text 
    transformer_heads= 2    # number of multiattention head to go through (# of different QKV pairs) 
    transformer_layers= 1   # number of layer of entire transformer operation 

    model = CLIP(embed_dim, image_resolution, vision_layer, vision_width,vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)


    image_tokens= model.visual(processed_img)


    # 1 x 5 
    #print(f"final output shape: {image_tokens.shape}")
    tokens= tokenizer.encode(caption)
    if (len(tokens)<context_length): 
       tokens= pad_tokens(tokens, context_length)

    #text_tokens= model.encode_text(torch.tensor(tokens).unsqueeze(0))
    
    #print(image_tokens)
    #print(text_tokens)

    #print(image_tokens@text_tokens.T)



if __name__ == "__main__":
    main() 
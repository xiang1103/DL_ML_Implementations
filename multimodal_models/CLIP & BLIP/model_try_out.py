# COCO dataset https://cocodataset.org/#explore
from PIL import Image
from torchvision import transforms
import json 

def process_image(path):
    img= Image.open(path)   # HxWxC 
    width, height= img.size 
    transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),  # CxHxW, normalize into [0,1]
        
    ])
    input_img= transform(img)
    return input_img 

def retrieve_captions(path):
    with open('../training_materials/captions.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    captions = data[]
    

def main(): 
    img_path = "training_materials/images/bicycle_woman.jpg"
    processed_img = process_image("../"+img_path)
    captions = retrieve_captions(img_path)



if __name__ == "__main__":
    main() 
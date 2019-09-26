from torchvision import models,transforms
import torch
from PIL import Image
from telegram.ext import Updater, CommandHandler,MessageHandler,BaseFilter,Filters
import requests
import re
import numpy as np
import os

#os.environ['TORCH_HOME'] = '~//Documents//CodeBase//pretrained_weights'

def load(update,context):    
    update.message.reply_text("Loading the Neural Network")
    classifier = models.resnet101(pretrained=True)
    context.user_data[0] = classifier
    update.message.reply_text("Loading complete")    

def process_image(update,context):
    chat_id = update.message.chat_id
    file_id = update.message.photo[-1].file_id    
    print("Time to classify")    
    input_image = context.bot.getFile(file_id)    
    input_image.download('test_image.jpg')
    
    transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])

    img = Image.open("test_image.jpg")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    classifier = context.user_data[0]
    classifier.eval()
    out = classifier(batch_t)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 

    # Load Imagenet Synsets
    with open('imagenet_synsets.txt', 'r') as f:
        synsets = f.readlines()
    synsets = [x.strip() for x in synsets]
    splits = [line.split(' ') for line in synsets]
    key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

    with open('imagenet_classes.txt', 'r') as f:
        class_id_to_key = f.readlines()
    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Make predictions
    _, indices = torch.sort(out, descending=True)
    for idx in indices[0][:5] :
        class_key = class_id_to_key[idx]
        classname = key_to_classname[class_key]
        print("The image is of {} with confidence {} %".format(classname,percentage[idx].item()))
        update.message.reply_text("The image is of {} with confidence {} %".format(classname,percentage[idx].item()))

def main():    
    updater = Updater('943315344:AAEwI_7FMvQK7NDAcekjFlRM6a4pB6JhIZo',use_context=True)    
    dp = updater.dispatcher    
    dp.add_handler(CommandHandler('load',load,pass_user_data=True))    
    dp.add_handler(MessageHandler(filters=Filters.photo,callback=process_image, pass_user_data=True))    
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
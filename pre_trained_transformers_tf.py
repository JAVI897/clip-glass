import tensorflow as tf
from transformer_models.utility import  get_inference_model, generate_caption, custom_standardization
import json
import re
import time
models_paths = {'EfficientNetB0 pre-trained on imagenet': 
					{512:['transformer_models/EfficientNetB0_imagenet_model_1/tokenizer', 
					      'transformer_models/EfficientNetB0_imagenet_model_1/config_train.json', 
					      'transformer_models/EfficientNetB0_imagenet_model_1/model_weights_coco.h5',
					      'transformer_models/EfficientNetB0_imagenet_model_1/history.json'], 
					1024:['transformer_models/EfficientNetB0_imagenet_model_2/tokenizer', 
					      'transformer_models/EfficientNetB0_imagenet_model_2/config_train.json', 
					      'transformer_models/EfficientNetB0_imagenet_model_2/model_weights_coco.h5',
					      'transformer_models/EfficientNetB0_imagenet_model_2/history.json']
					},
				'VGG16 pre-trained on imagenet':
					{512:['transformer_models/VGG16_imagenet_model_1/tokenizer', 
					      'transformer_models/VGG16_imagenet_model_1/config_train.json', 
					      'transformer_models/VGG16_imagenet_model_1/model_weights_coco.h5',
					      'transformer_models/VGG16_imagenet_model_1/history.json']
					},
				'ResNet pre-trained on imagenet':
					{512:['transformer_models/ResNet_imagenet_model_1/tokenizer',
						  'transformer_models/ResNet_imagenet_model_1/config_train.json', 
					      'transformer_models/ResNet_imagenet_model_1/model_weights_coco.h5',
					      'transformer_models/ResNet_imagenet_model_1/history.json']
					}
				}

def get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path):
	tokenizer = tf.keras.models.load_model(tokernizer_path)
	tokenizer = tokenizer.layers[1]
	# Get model
	model = get_inference_model(get_model_config_path)
	# Load model weights
	model.load_weights(get_model_weights_path)
	with open(get_model_config_path) as json_file:
		model_config = json.load(json_file)
	text_caption = generate_caption(img, model, tokenizer, model_config["SEQ_LENGTH"])
	return text_caption

def get_captions( img_path, EfficientNetB0_1 = True, EfficientNetB0_2 = True, VGG16_1 = True, ResNet_1 = True):

	img = tf.image.decode_jpeg(tf.io.read_file(img_path), channels=3)
	img = tf.image.resize(img, (299, 299))
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.expand_dims(img, axis=0)

	captions = []

	if EfficientNetB0_1 == True:
		top_cnn = 'EfficientNetB0 pre-trained on imagenet'
		embedding_dim = 512
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)
		
	if EfficientNetB0_2 == True:
		top_cnn = 'EfficientNetB0 pre-trained on imagenet'
		embedding_dim = 1024
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)

	if VGG16_1 == True:
		top_cnn = 'VGG16 pre-trained on imagenet'
		embedding_dim = 512
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)
	if ResNet_1 == True:
		top_cnn = 'ResNet pre-trained on imagenet'
		embedding_dim = 512
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)

	return captions

ini_t = time.time()
captions_preds = get_captions( 'target.jpg', EfficientNetB0_1 = True, EfficientNetB0_2 = True, VGG16_1 = True, ResNet_1 = True)
for i in captions_preds:
	print(i)
fin_t = time.time()
print('Tiempo total: ', fin_t - ini_t)
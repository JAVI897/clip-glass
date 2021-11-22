import tensorflow as tf
from transformer_models.utility import  get_inference_model, generate_caption, custom_standardization
import json
import re

#@tf.keras.utils.register_keras_serializable()
class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   def __init__(self, d_model, warmup_steps=4000):
      super(custom_schedule, self).__init__()
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps

   def __call__(self, step):
      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

   def get_config(self):
      config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps
        }
      return config

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

	elif EfficientNetB0_2 == True:
		top_cnn = 'EfficientNetB0 pre-trained on imagenet'
		embedding_dim = 1024
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)

	elif VGG16_1 == True:
		top_cnn = 'VGG16 pre-trained on imagenet'
		embedding_dim = 512
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)
	elif ResNet_1 == True:
		top_cnn = 'ResNet pre-trained on imagenet'
		embedding_dim = 512
		tokernizer_path = models_paths[top_cnn][embedding_dim][0]
		get_model_config_path = models_paths[top_cnn][embedding_dim][1]
		get_model_weights_path = models_paths[top_cnn][embedding_dim][2]
		prediction = get_image_prediction(img, tokernizer_path, get_model_config_path, get_model_weights_path)
		captions.append(prediction)

	return captions

if __name__ == 'main':
	captions_preds = get_captions( 'target.jpg', EfficientNetB0_1 = True, EfficientNetB0_2 = True, VGG16_1 = True, ResNet_1 = True)
	for i in captions_preds:
		print(i)
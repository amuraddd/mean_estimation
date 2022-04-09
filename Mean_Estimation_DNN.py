from utils.preprocessing import preprocess_image #import the preprocessing function from utils
from models.mean_estimation_dnn_model import train_predict #import the train_model function from models

# preprocess images to get pixel values as Y and pixel coordinates as X
X, Y = preprocess_image(data_location='{your_image_path}/*.nii', #change {your_image_path} to the location where your images are saved
                        img_slice=20,
                        num_images_to_use='all',
                        plot_image_slice=False,
                        plot_mean_image_slice=False)

# define model parameters - currently set to default
model_params = {"num_layers": 3,
                "neurons_per_layer": 1000,
                "weight_initializer": "standard_normal",
                "activation_per_layer": "relu",
                "epochs": 500,
                "loss": "mse",
                "l1_regularizer": True,
                "l1_penalty": 1e-7,
                "batch_size": 4,
                "verbose": 0,
                "get_high_resolution_image": True,
                "high_resolution_dimensions": (500, 500),
                "cmap": "gray",
                "save_image_location_and_name": "{your_high_res_image_path}.png"} #change {your_high_res_image_path} to where the images must be saved

# this will only save the recovered images to the location specified in the model_params
train_predict(X, Y, **model_params)
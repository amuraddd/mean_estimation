### Estimation of the Mean Function of the Functional Data via Deep Neural Networks  

### Data
The format of the images used to build the code in this reposiroty is `nii`. If the preprocessing function contained within this repository is used then  
the data must follow the same format, and the location containing all images must be passed to the `preprocess_image` function.

### How to run the code below
To run the code below import the `preprocesing` function from utils and the `train_predict` function from models.  

```python
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

# get model, model_history, and predicted pixel values
model_history, model, y_pred = train_predict(X, Y, **model_params)
```

### Option 1:  
The code can be ran through the `Mean_Estimation_DNN.ipynb` notebook by running the cell containing the code.  
This will return:
    1. Trained model
    2. Model training history
    3. Predicted pixel values

### Option 2:
Run the code as a python process by running the pyhton file `Mean_Estimation_DNN.py` from the terminal
This will only save the recovered images to the location specified in the model_params passed to the `train_predict` function.

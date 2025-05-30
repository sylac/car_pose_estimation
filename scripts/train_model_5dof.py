# filepath: d:\0_WORK\Repository\Other\ai\car_pose_estimation\scripts\train_model_5dof.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params

from car_azimuth_predictor.train_model import train_model
from car_azimuth_predictor.config import load_config
from car_azimuth_predictor.dataset_generation import generate_datasets
from car_azimuth_predictor.utils.training_tools import (
    horizontal_flip_pose_sin_cos_output,
    tf_acc_pi_6_sin_cos_output,
    tf_mean_absolute_angle_error_sin_cos_output,
    tf_median_absolute_angle_error_sin_cos_output,
    tf_rmse_angle_sin_cos_output,
    tf_r2_angle_score_sin_cos_output,
    horizontal_flip_pose_5dof,
    pose_5dof_loss,
    tf_mean_absolute_angle_error_5dof,
    tf_distance_mae_5dof,
    tf_azimuth_mae_5dof,
    tf_elevation_mae_5dof,
    tf_rmse_angle_5dof,
    tf_acc_pi_6_5dof,
    np_get_pose_5dof_from_output,
    augment_image,
    CustomHorizontalFlip,
)


def main(
    approach: int,
    model_name: str,
    epochs: int,
    config_name: str,
    saved_model_path: str,
    initial_model_path: str,
    model_config_name: str,
):
    """
    Main training function for 5DOF car pose estimation
      Args:
        approach: Training approach (1, 2, or 3)
        model_name: Name for the saved model
        epochs: Number of training epochs
        config_name: Configuration file name
        saved_model_path: Path to save the trained model
        initial_model_path: Path to initial model (optional)
        model_config_name: Model configuration name
    """
    print(f"Training 5DOF model with approach {approach}")
    
    # Load configuration
    config = load_config()  # Uses main.yaml which points to dataset_generation_5dof
      # Expected ground truth columns for 5DOF: [azimuth_sin, azimuth_cos, elevation_sin, elevation_cos, distance_normalized]
    gt_cols = ["azimuth_sin", "azimuth_cos", "elevation_sin", "elevation_cos", "distance_normalized"]
    
    # Generate datasets - we'll define pose_flip_fn based on approach
    if approach == 1 or approach == 2:
        pose_flip_fn = horizontal_flip_pose_5dof
    elif approach == 3:
        pose_flip_fn = horizontal_flip_pose_5dof  # Will be handled differently in multi-output
        
    train_dataset, validation_dataset = generate_datasets(
        config=config, 
        gt_cols=gt_cols, 
        pose_flip_fn=pose_flip_fn,        
        batch_size=config.dataset_generation.batch_size,
        augment=True
    )

    if approach == 1:
        # Approach 1: Sin/Cos for azimuth only (original approach, modified for 5DOF)
        def top_model_fn(x):
            """Create the top part of the model for 5DOF output"""
            x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
            x = tf.keras.layers.Dense(100, activation='relu', name="mlp_middle")(x)
            outputs = tf.keras.layers.Dense(5, activation='tanh', name='pose_5dof')(x)
            return outputs

        def model_fn():
            from car_azimuth_predictor.model_generation import generate_model
            model = generate_model(config, top_model_fn)
            return model

        loss_fn = pose_5dof_loss
        metrics = [
            tf_azimuth_mae_5dof,
            tf_elevation_mae_5dof, 
            tf_distance_mae_5dof,
            tf_mean_absolute_angle_error_5dof,
            tf_rmse_angle_5dof,
            tf_acc_pi_6_5dof
        ]
        custom_flip_fn = CustomHorizontalFlip(horizontal_flip_pose_5dof)

    elif approach == 2:
        # Approach 2: Double sigmoid for azimuth, extended for 5DOF
        def top_model_fn(x):
            """Create the top part of the model for 5DOF output"""
            x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
            x = tf.keras.layers.Dense(100, activation='relu', name="mlp_middle")(x)
            outputs = tf.keras.layers.Dense(5, activation='sigmoid', name='pose_5dof')(x)
            return outputs

        def model_fn():
            from car_azimuth_predictor.model_generation import generate_model
            model = generate_model(config, top_model_fn)
            return model

        loss_fn = pose_5dof_loss
        metrics = [
            tf_azimuth_mae_5dof,
            tf_elevation_mae_5dof,
            tf_distance_mae_5dof, 
            tf_mean_absolute_angle_error_5dof,
            tf_rmse_angle_5dof,
            tf_acc_pi_6_5dof
        ]
        custom_flip_fn = CustomHorizontalFlip(horizontal_flip_pose_5dof)

    elif approach == 3:
        # Approach 3: Multi-output model with separate heads for each component
        def top_model_fn(x):
            """Create multi-output top model"""
            x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
            x = tf.keras.layers.Dense(100, activation='relu', name="mlp_middle")(x)
            
            # Create separate output heads
            azimuth_output = tf.keras.layers.Dense(2, activation='tanh', name='azimuth')(x)
            elevation_output = tf.keras.layers.Dense(2, activation='tanh', name='elevation')(x)  
            distance_output = tf.keras.layers.Dense(1, activation='linear', name='distance')(x)
            
            return [azimuth_output, elevation_output, distance_output]

        def model_fn():
            from car_azimuth_predictor.model_generation import generate_model
            model = generate_model(config, top_model_fn)
            return model

        def combined_loss(y_true, y_pred):
            # Split y_true: [azimuth_sin, azimuth_cos, elevation_sin, elevation_cos, distance]
            azimuth_true = y_true[:, 0:2]
            elevation_true = y_true[:, 2:4]
            distance_true = y_true[:, 4:5]
            
            # y_pred is a list: [azimuth_pred, elevation_pred, distance_pred]
            azimuth_pred, elevation_pred, distance_pred = y_pred
            
            azimuth_loss = tf.keras.losses.MSE(azimuth_true, azimuth_pred) * 1.0
            elevation_loss = tf.keras.losses.MSE(elevation_true, elevation_pred) * 1.0  
            distance_loss = tf.keras.losses.MSE(distance_true, distance_pred) * 0.8
            
            return azimuth_loss + elevation_loss + distance_loss

        loss_fn = {
            'azimuth': tf.keras.losses.MSE,
            'elevation': tf.keras.losses.MSE,
            'distance': tf.keras.losses.MSE
        }
        
        metrics = {
            'azimuth': [tf_mean_absolute_angle_error_sin_cos_output, tf_acc_pi_6_sin_cos_output],
            'elevation': [tf_mean_absolute_angle_error_sin_cos_output, tf_acc_pi_6_sin_cos_output],
            'distance': ['mae', 'mse']
        }
        
        def custom_flip_5dof_multioutput(pose):
            """Custom flip for multi-output format"""
            # pose format: [azimuth_sin, azimuth_cos, elevation_sin, elevation_cos, distance]
            flipped = horizontal_flip_pose_5dof(pose)
            return {
                'azimuth': [flipped[0], flipped[1]], 
                'elevation': [flipped[2], flipped[3]],
                'distance': flipped[4]
            }
        
        custom_flip_fn = CustomHorizontalFlip(custom_flip_5dof_multioutput)

    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Create and compile model
    model = model_fn()
    print(f"Model created with {count_params(model.trainable_weights):,} trainable parameters")
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["model_training"]["learning_rate"])
    
    if approach == 3:
        # Multi-output model compilation
        loss_weights = {'azimuth': 1.0, 'elevation': 1.0, 'distance': 0.8}
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics, loss_weights=loss_weights)
    else:
        # Single output model compilation  
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    print("Model compiled successfully")
    print("Model summary:")
    model.summary()

    # Prepare data augmentation
    def augment_fn(img, y_true):
        return augment_image(img, y_true, custom_flip_fn)    # Apply augmentation to training dataset
    if approach == 3:
        # Multi-output: convert single y_true to dict format
        def convert_to_multioutput(img, y_true):
            azimuth = y_true[:, 0:2]
            elevation = y_true[:, 2:4] 
            distance = y_true[:, 4:5]
            return img, {'azimuth': azimuth, 'elevation': elevation, 'distance': distance}
        
        train_dataset = train_dataset.map(convert_to_multioutput)
        validation_dataset = validation_dataset.map(convert_to_multioutput)

    # Setup callbacks
    from car_azimuth_predictor.utils.callbacks import get_callbacks
    callbacks = get_callbacks(
        model_name=model_name,
        saved_model_path=saved_model_path,
        config=config
    )

    # Train model
    print(f"Starting training for {epochs} epochs...")
    
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("Training completed!")
    
    # Save final model
    final_model_path = f"{saved_model_path}/{model_name}_final_5dof.h5"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5DOF car pose estimation model")
    parser.add_argument(
        "--approach", 
        type=int, 
        choices=[1, 2, 3], 
        required=True,
        help="Training approach: 1=sin/cos, 2=double_sigmoid, 3=multi-output"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="car_pose_5dof",
        help="Name for the saved model"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="dataset_generation_5dof",
        help="Configuration file name"
    )
    parser.add_argument(
        "--saved_model_path", 
        type=str, 
        default="./models",
        help="Path to save trained models"
    )
    parser.add_argument(
        "--initial_model_path", 
        type=str, 
        default=None,
        help="Path to initial model weights"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="model_training",
        help="Model configuration name"
    )

    args = parser.parse_args()

    main(
        approach=args.approach,
        model_name=args.model_name,
        epochs=args.epochs,
        config_name=args.config,
        saved_model_path=args.saved_model_path,
        initial_model_path=args.initial_model_path,
        model_config_name=args.model_config,
    )

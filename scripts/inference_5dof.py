#!/usr/bin/env python3
"""
5DOF Car Pose Inference Script

This script performs inference on car images using a trained 5DOF pose estimation model.
It estimates azimuth (horizontal rotation), elevation (vertical rotation), and distance.

Usage:
    python inference_5dof.py --model_path ./models/car_pose_5dof_best.h5 \
                             --images_path ./test_images \
                             --output_path ./results_5dof.json \
                             --approach 3
"""

import argparse
from keras.models import load_model
import os
import sys
from PIL import Image
import tensorflow as tf
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from car_azimuth_predictor.utils.visualization_tools import plot_image_from_path

import tensorflow_hub as hub
from car_azimuth_predictor.utils.training_tools import (
    # Original azimuth functions (approaches 1 & 2)
    horizontal_flip_pose_sin_cos_output,
    tf_acc_pi_6_sin_cos_output,
    tf_mean_absolute_angle_error_sin_cos_output,
    tf_median_absolute_angle_error_sin_cos_output,
    tf_r2_angle_score_sin_cos_output,
    tf_rmse_angle_sin_cos_output,
    angle_double_output_loss,
    horizontal_flip_pose_double_sigmoid,
    tf_mean_absolute_angle_error_double_sigmoid,
    tf_median_absolute_angle_error_double_sigmoid,
    tf_r2_angle_score_double_sigmoid,
    tf_rmse_angle_score_double_sigmoid,
    tf_acc_pi_6_double_sigmoid,
    np_get_angle_from_double_sigmoids,
    np_get_angle_from_sin_cos,    # 5DOF functions (approach 3)
    pose_5dof_loss,
    tf_azimuth_mae_5dof,
    tf_elevation_mae_5dof,
    tf_distance_mae_5dof,
    tf_mean_absolute_angle_error_5dof,
    tf_rmse_angle_5dof,
    tf_acc_pi_6_5dof,
    horizontal_flip_pose_5dof,
    np_get_pose_5dof_from_output,
)


def load_image_to_tensor(image_path: str) -> tf.Tensor:
    """Load image and convert to tensor for model input.

    :param image_path: Path to the image file
    :return: Processed image tensor ready for inference
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, (224, 224))
    # Normalize to [0, 1] range
    image_tensor = image_tensor / 255.0
    return image_tensor


def chunker(seq, size):
    """Split sequence into chunks of specified size."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def create_5dof_visualization(image_path: str, pose_data: dict, save_path: str):
    """Create visualization showing 5DOF pose estimation results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Load and display original image
    image = Image.open(image_path)
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Create pose information display
    axes[1].text(0.1, 0.8, f"Azimuth: {pose_data['azimuth']:.1f}°", 
                 fontsize=14, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.6, f"Elevation: {pose_data['elevation']:.1f}°", 
                 fontsize=14, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.4, f"Distance: {pose_data['distance']:.3f}", 
                 fontsize=14, transform=axes[1].transAxes)
    
    # Create simple 3D representation
    axes[1].text(0.1, 0.2, "Pose Interpretation:", fontsize=12, weight='bold', 
                 transform=axes[1].transAxes)
    
    # Interpret angles for user guidance
    azimuth_desc = ""
    if abs(pose_data['azimuth']) < 15:
        azimuth_desc = "Front view"
    elif 15 <= pose_data['azimuth'] < 75:
        azimuth_desc = "Front-right view"
    elif 75 <= pose_data['azimuth'] < 105:
        azimuth_desc = "Right side view"
    elif 105 <= pose_data['azimuth'] < 165:
        azimuth_desc = "Back-right view"
    elif abs(pose_data['azimuth']) >= 165:
        azimuth_desc = "Back view"
    elif -75 <= pose_data['azimuth'] < -15:
        azimuth_desc = "Front-left view"
    elif -105 <= pose_data['azimuth'] < -75:
        azimuth_desc = "Left side view"
    elif -165 <= pose_data['azimuth'] < -105:
        azimuth_desc = "Back-left view"
    
    elevation_desc = ""
    if pose_data['elevation'] > 10:
        elevation_desc = "Looking down"
    elif pose_data['elevation'] < -10:
        elevation_desc = "Looking up"
    else:
        elevation_desc = "Eye level"
    
    distance_desc = ""
    if pose_data['distance'] < 0.3:
        distance_desc = "Very close"
    elif pose_data['distance'] < 0.6:
        distance_desc = "Close"
    elif pose_data['distance'] < 0.8:
        distance_desc = "Medium distance"
    else:
        distance_desc = "Far away"
    
    axes[1].text(0.1, 0.1, f"• {azimuth_desc}", fontsize=10, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.05, f"• {elevation_desc}", fontsize=10, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.0, f"• {distance_desc}", fontsize=10, transform=axes[1].transAxes)
    
    axes[1].set_title("5DOF Pose Estimation Results")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main(model_path, images_path, approach: str, batch_size=32, output_path=None, 
         visualizations_path=None, units='degrees'):
    """
    Main inference function for 5DOF car pose estimation.
    
    Args:
        model_path: Path to the trained model
        images_path: Directory containing test images
        approach: Model approach ("1", "2", or "3" for 5DOF)
        batch_size: Batch size for inference
        output_path: Path to save JSON results
        visualizations_path: Path to save visualization images
        units: Output units ('degrees' or 'radians')
    """
    
    # Load model with custom objects
    custom_objects = {
        "KerasLayer": hub.KerasLayer,
        # Approach 1 & 2 functions
        "angle_double_output_loss": angle_double_output_loss,
        "tf_mean_absolute_angle_error_double_sigmoid": tf_mean_absolute_angle_error_double_sigmoid,
        "tf_rmse_angle_score_double_sigmoid": tf_rmse_angle_score_double_sigmoid,
        "tf_r2_angle_score_double_sigmoid": tf_r2_angle_score_double_sigmoid,
        "tf_median_absolute_angle_error_double_sigmoid": tf_median_absolute_angle_error_double_sigmoid,
        "tf_acc_pi_6_double_sigmoid": tf_acc_pi_6_double_sigmoid,
        "horizontal_flip_pose_double_sigmoid": horizontal_flip_pose_double_sigmoid,
        "tf_mean_absolute_angle_error_sin_cos_output": tf_mean_absolute_angle_error_sin_cos_output,
        "tf_rmse_angle_sin_cos_output": tf_rmse_angle_sin_cos_output,
        "tf_r2_angle_score_sin_cos_output": tf_r2_angle_score_sin_cos_output,
        "tf_median_absolute_angle_error_sin_cos_output": tf_median_absolute_angle_error_sin_cos_output,
        "tf_acc_pi_6_sin_cos_output": tf_acc_pi_6_sin_cos_output,
        "horizontal_flip_pose_sin_cos_output": horizontal_flip_pose_sin_cos_output,        # 5DOF functions (approach 3)
        "pose_5dof_loss": pose_5dof_loss,
        "tf_azimuth_mae_5dof": tf_azimuth_mae_5dof,
        "tf_elevation_mae_5dof": tf_elevation_mae_5dof,
        "tf_distance_mae_5dof": tf_distance_mae_5dof,
        "tf_mean_absolute_angle_error_5dof": tf_mean_absolute_angle_error_5dof,
        "tf_rmse_angle_5dof": tf_rmse_angle_5dof,
        "tf_acc_pi_6_5dof": tf_acc_pi_6_5dof,
        "horizontal_flip_pose_5dof": horizontal_flip_pose_5dof,
    }
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully!")
    
    # Get list of image files
    files = sorted([f for f in os.listdir(images_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    
    if not files:
        print(f"No image files found in {images_path}")
        return
    
    print(f"Found {len(files)} images for inference")
    
    all_predictions = []
    
    # Process images in batches
    for batch_files in chunker(files, batch_size):
        print(f"Processing batch of {len(batch_files)} images...")
        
        # Load and preprocess images
        images = []
        for image_file in batch_files:
            image_path = os.path.join(images_path, image_file)
            image_tensor = load_image_to_tensor(image_path)
            images.append(image_tensor)
        
        images = tf.stack(images, axis=0)
        
        # Run inference
        predictions = model.predict(images, verbose=0)
        
        # Handle different model output formats
        if approach == "3":  # 5DOF multi-output model
            if isinstance(predictions, list):
                # Multi-output model returns [azimuth_output, elevation_output, distance_output]
                azimuth_output = predictions[0]  # Shape: (batch_size, 2)
                elevation_output = predictions[1]  # Shape: (batch_size, 2) 
                distance_output = predictions[2]  # Shape: (batch_size, 1)
                
                # Combine outputs for pose extraction
                combined_output = np.concatenate([
                    azimuth_output,     # cols 0:2
                    elevation_output,   # cols 2:4  
                    distance_output     # col 4
                ], axis=1)
                
                all_predictions.extend(combined_output)
            else:
                # Single output model
                all_predictions.extend(predictions)
        else:
            # Original azimuth-only models (approaches 1 & 2)
            all_predictions.extend(predictions)
    
    # Convert predictions to pose estimates
    all_predictions = np.array(all_predictions)
    print(f"Prediction shape: {all_predictions.shape}")
    
    if approach == "3":
        # 5DOF pose estimation
        pose_results = np_get_pose_5dof_from_output(all_predictions)
        
        # Convert to specified units
        if units == 'radians':
            pose_results['azimuth'] = pose_results['azimuth'] * np.pi / 180
            pose_results['elevation'] = pose_results['elevation'] * np.pi / 180
        elif units != 'degrees':
            raise ValueError("Unknown units. Use 'degrees' or 'radians'")
        
        # Prepare output
        output = []
        for i, image_file in enumerate(files):
            result = {
                "image": image_file,
                "azimuth": float(pose_results['azimuth'][i]),
                "elevation": float(pose_results['elevation'][i]),
                "distance": float(pose_results['distance'][i])
            }
            output.append(result)
            
        print(f"\nSample results:")
        for i in range(min(3, len(output))):
            result = output[i]
            print(f"  {result['image']}: azimuth={result['azimuth']:.1f}°, "
                  f"elevation={result['elevation']:.1f}°, distance={result['distance']:.3f}")
        
    else:
        # Original azimuth-only estimation (approaches 1 & 2)
        if approach == "1":
            azimuth_converter = np_get_angle_from_sin_cos
        elif approach == "2":
            azimuth_converter = np_get_angle_from_double_sigmoids
        else:
            raise ValueError("Unknown approach")
            
        azimuths = azimuth_converter(all_predictions)
        if units == 'degrees':
            azimuths = azimuths / np.pi * 180
        elif units != 'radians':
            raise ValueError("Unknown units")

        azimuths = azimuths.astype(float).tolist()

        output = []
        for i, (image_file, azimuth) in enumerate(zip(files, azimuths)):
            output.append({"image": image_file, "azimuth": azimuth})
    
    # Save results to JSON
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Generate visualizations
    if visualizations_path is not None:
        import matplotlib.pyplot as plt
        os.makedirs(visualizations_path, exist_ok=True)
        print(f"\nGenerating visualizations in: {visualizations_path}")
        
        for i, result in enumerate(output):
            image_path = os.path.join(images_path, result['image'])
            save_path = os.path.join(visualizations_path, f"result_{result['image']}")
            
            if approach == "3":
                # Create 5DOF visualization
                create_5dof_visualization(image_path, result, save_path)
            else:
                # Original azimuth visualization
                plot_image_from_path(image_path, result['azimuth'], None)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
        
        print(f"Generated {len(output)} visualizations")
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5DOF Car Pose Inference')
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument("--approach", type=str, 
                       help="Approach: 1=Sin&Cos, 2=Directional, 3=5DOF multi-output", 
                       choices=["1", "2", "3"], required=True)
    parser.add_argument('--images_path', type=str, help='Path to images directory', required=True)
    parser.add_argument('--output_path', type=str, help='Path to output JSON file', required=True)
    parser.add_argument('--visualizations_path', type=str, 
                       help='Path to save visualizations (optional)', default=None)
    parser.add_argument('--batch_size', type=int, help='Batch size for inference', default=16)
    parser.add_argument('--units', type=str, help='Output units: degrees or radians', 
                       default='degrees', choices=['radians', 'degrees'])
    
    args = parser.parse_args()
    
    results = main(
        model_path=args.model_path,
        images_path=args.images_path, 
        approach=args.approach,
        batch_size=args.batch_size,
        output_path=args.output_path,
        visualizations_path=args.visualizations_path,
        units=args.units
    )

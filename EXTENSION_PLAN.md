# Detailed Plan: Extending Car Pose Estimation to Include Elevation, Tilt, and Distance

## Current State Analysis

### What We Have:
- **Azimuth estimation**: Horizontal rotation angle (-π to π)
- **EfficientNetB0 backbone**: 1280-dimensional feature extraction
- **Two approaches**: Sin/Cos and Directional discriminators
- **PASCAL3D+ dataset**: Already contains azimuth, elevation, and distance annotations
- **Training pipeline**: Complete with data loading, augmentation, and evaluation

### What We Need to Add:
1. **Elevation estimation**: Vertical viewing angle (-π/2 to π/2)
2. **Tilt estimation**: Camera/object roll angle (-π to π) 
3. **Distance estimation**: Relative distance/scale (continuous value)

## Phase 1: Data Pipeline Extension (1-2 weeks)

### 1.1 Feature Generation Updates
**File**: `car_azimuth_predictor/feature_generation.py`

**Changes Needed**:
- Extract elevation and distance from PASCAL3D+ annotations (already available)
- Add tilt/roll angle processing (may need to derive from existing data or use default values)
- Create normalized representations for all angles
- Update CSV output to include new features

**New Features to Add**:
```python
# Elevation processing (similar to azimuth)
df["elevation_radians"] = df["elevation"] / 180 * np.pi
df["elevation_sin"] = np.sin(df["elevation_radians"])
df["elevation_cos"] = np.cos(df["elevation_radians"])
df["elevation_norm_abs"] = np.abs(df["elevation_radians"]) / (np.pi/2)

# Distance processing (normalize to 0-1 range)
df["distance_normalized"] = (df["distance"] - df["distance"].min()) / (df["distance"].max() - df["distance"].min())

# Tilt processing (if available, otherwise set to 0 for PASCAL3D+)
df["tilt_radians"] = 0  # PASCAL3D+ doesn't have tilt, set to 0 initially
df["tilt_sin"] = np.sin(df["tilt_radians"])
df["tilt_cos"] = np.cos(df["tilt_radians"])
```

### 1.2 Dataset Configuration Updates
**File**: `config/dataset_generation/dataset_generation.yaml`

**Changes**:
```yaml
gt_cols:
  # Azimuth (existing)
  - "azimuth_sin"
  - "azimuth_cos"
  # Elevation (new)
  - "elevation_sin" 
  - "elevation_cos"
  # Tilt (new)
  - "tilt_sin"
  - "tilt_cos"
  # Distance (new)
  - "distance_normalized"
```

### 1.3 Data Augmentation Updates
**File**: `car_azimuth_predictor/utils/training_tools.py`

**New Functions Needed**:
```python
def horizontal_flip_pose_6dof(pose):
    """Flip pose for 6DOF representation"""
    azimuth_sin, azimuth_cos, elevation_sin, elevation_cos, tilt_sin, tilt_cos, distance = pose
    return [
        -azimuth_sin,      # Flip azimuth
        azimuth_cos,       # Keep azimuth cos
        elevation_sin,     # Keep elevation (vertical doesn't change)
        elevation_cos,     
        -tilt_sin,         # Flip tilt
        tilt_cos,          
        distance           # Keep distance
    ]
```

## Phase 2: Model Architecture Extension (2-3 weeks)

### 2.1 Multi-Output Model Architecture
**File**: `scripts/train_model.py`

**New Approach (Approach 3 - 6DOF)**:
```python
if approach == "3":  # 6DOF estimation
    # Azimuth outputs
    azimuth_sin = tf.keras.layers.Dense(1, activation="tanh", name="azimuth_sin")(in_to_dense)
    azimuth_cos = tf.keras.layers.Dense(1, activation="tanh", name="azimuth_cos")(in_to_dense)
    
    # Elevation outputs  
    elevation_sin = tf.keras.layers.Dense(1, activation="tanh", name="elevation_sin")(in_to_dense)
    elevation_cos = tf.keras.layers.Dense(1, activation="tanh", name="elevation_cos")(in_to_dense)
    
    # Tilt outputs
    tilt_sin = tf.keras.layers.Dense(1, activation="tanh", name="tilt_sin")(in_to_dense)
    tilt_cos = tf.keras.layers.Dense(1, activation="tanh", name="tilt_cos")(in_to_dense)
    
    # Distance output
    distance = tf.keras.layers.Dense(1, activation="sigmoid", name="distance")(in_to_dense)
    
    # Concatenate all outputs
    outputs = tf.keras.layers.Concatenate(axis=1, name="pose_6dof")([
        azimuth_sin, azimuth_cos, 
        elevation_sin, elevation_cos,
        tilt_sin, tilt_cos,
        distance
    ])
```

### 2.2 Loss Function Design
**File**: `car_azimuth_predictor/utils/training_tools.py`

**Multi-task Loss Function**:
```python
def pose_6dof_loss(y_true, y_pred):
    """Combined loss for 6DOF pose estimation"""
    # Split predictions
    azimuth_true = y_true[:, 0:2]
    elevation_true = y_true[:, 2:4] 
    tilt_true = y_true[:, 4:6]
    distance_true = y_true[:, 6:7]
    
    azimuth_pred = y_pred[:, 0:2]
    elevation_pred = y_pred[:, 2:4]
    tilt_pred = y_pred[:, 4:6] 
    distance_pred = y_pred[:, 6:7]
    
    # Weighted losses
    azimuth_loss = tf.keras.losses.MSE(azimuth_true, azimuth_pred) * 1.0
    elevation_loss = tf.keras.losses.MSE(elevation_true, elevation_pred) * 1.0
    tilt_loss = tf.keras.losses.MSE(tilt_true, tilt_pred) * 0.5  # Lower weight if less reliable
    distance_loss = tf.keras.losses.MSE(distance_true, distance_pred) * 0.8
    
    return azimuth_loss + elevation_loss + tilt_loss + distance_loss
```

### 2.3 Evaluation Metrics Extension
**File**: `car_azimuth_predictor/utils/training_tools.py`

**New Metrics**:
```python
def tf_mean_absolute_angle_error_6dof(y_true, y_pred):
    """Calculate MAE for all angles"""
    azimuth_mae = tf_mean_absolute_angle_error_sin_cos_output(y_true[:, 0:2], y_pred[:, 0:2])
    elevation_mae = tf_mean_absolute_angle_error_sin_cos_output(y_true[:, 2:4], y_pred[:, 2:4])
    tilt_mae = tf_mean_absolute_angle_error_sin_cos_output(y_true[:, 4:6], y_pred[:, 4:6])
    return (azimuth_mae + elevation_mae + tilt_mae) / 3

def tf_distance_mae(y_true, y_pred):
    """Distance mean absolute error"""
    distance_true = y_true[:, 6:7]
    distance_pred = y_pred[:, 6:7]
    return tf.keras.metrics.mean_absolute_error(distance_true, distance_pred)
```

## Phase 3: Training Pipeline Updates (1 week)

### 3.1 Training Script Updates
**File**: `scripts/train_model.py`

**Key Changes**:
- Add approach "3" for 6DOF estimation
- Update ground truth columns list
- Integrate new loss function and metrics
- Adjust learning rate and batch size for multi-task learning

### 3.2 Validation Updates  
**File**: `scripts/validate_model.py`

**New Validation Logic**:
```python
def validate_6dof_model(model, validation_dataset):
    """Validate 6DOF pose estimation"""
    predictions = model.predict(validation_dataset)
    
    # Extract individual pose components
    azimuth_angles = np_get_angle_from_sin_cos(predictions[:, 0:2])
    elevation_angles = np_get_angle_from_sin_cos(predictions[:, 2:4])
    tilt_angles = np_get_angle_from_sin_cos(predictions[:, 4:6])
    distances = predictions[:, 6]
    
    return {
        'azimuth_mae': calculate_mae(azimuth_angles, true_azimuth),
        'elevation_mae': calculate_mae(elevation_angles, true_elevation),
        'tilt_mae': calculate_mae(tilt_angles, true_tilt),
        'distance_mae': calculate_mae(distances, true_distances)
    }
```

## Phase 4: Inference and Visualization Updates (1 week)

### 4.1 Inference Script Updates
**File**: `scripts/inference.py`

**Enhanced Inference**:
```python
def main_6dof(model_path, images_path, output_path=None):
    """Run 6DOF inference"""
    model = load_model(model_path)
    
    predictions = model.predict(images)
    
    # Extract pose components
    azimuths = np_get_angle_from_sin_cos(predictions[:, 0:2]) * 180 / np.pi
    elevations = np_get_angle_from_sin_cos(predictions[:, 2:4]) * 180 / np.pi  
    tilts = np_get_angle_from_sin_cos(predictions[:, 4:6]) * 180 / np.pi
    distances = predictions[:, 6]
    
    # Output format
    results = []
    for i, (image_path, az, el, tilt, dist) in enumerate(zip(files, azimuths, elevations, tilts, distances)):
        results.append({
            "image": image_path,
            "azimuth": float(az),
            "elevation": float(el), 
            "tilt": float(tilt),
            "distance": float(dist)
        })
```

### 4.2 Visualization Enhancements
**File**: `car_azimuth_predictor/utils/visualization_tools.py`

**3D Pose Visualization**:
```python
def plot_pose_6dof(image_tensor, azimuth, elevation, tilt, distance, ground_truth=None):
    """Visualize complete 6DOF pose"""
    # Create 3D coordinate system visualization
    # Show car orientation with arrows/lines indicating all 3 rotation axes
    # Display distance as scale indicator
    # Optional: overlay ground truth for comparison
```

## Phase 5: Mobile-Specific Adaptations (2-3 weeks)

### 5.1 Model Optimization
**File**: `scripts/convert_to_mobile.py` (new)

**TensorFlow Lite Conversion**:
```python
def convert_to_tflite(model_path, output_path):
    """Convert 6DOF model to TensorFlow Lite"""
    model = tf.keras.models.load_model(model_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Half precision
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

### 5.2 Real-time Processing Pipeline
**File**: `mobile/pose_estimator.py` (new)

**Real-time Inference Class**:
```python
class RealTimePoseEstimator:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
    
    def estimate_pose(self, image_array):
        """Process single frame and return 6DOF pose"""
        # Preprocess image
        # Run inference
        # Return structured pose result
        return {
            'azimuth': float,
            'elevation': float, 
            'tilt': float,
            'distance': float,
            'confidence': float
        }
```

### 5.3 Guidance Logic System
**File**: `mobile/guidance_engine.py` (new)

**Pose Comparison and Guidance**:
```python
class PoseGuidanceEngine:
    def __init__(self, target_poses):
        self.target_poses = target_poses  # Company gold standards
    
    def generate_guidance(self, current_pose, target_type="front_view"):
        """Generate human-readable guidance instructions"""
        target = self.target_poses[target_type]
        
        instructions = []
        
        # Azimuth guidance
        azimuth_diff = target['azimuth'] - current_pose['azimuth']
        if abs(azimuth_diff) > 5:  # 5 degree threshold
            if azimuth_diff > 0:
                instructions.append("Move to the right")
            else:
                instructions.append("Move to the left")
        
        # Elevation guidance  
        elevation_diff = target['elevation'] - current_pose['elevation']
        if abs(elevation_diff) > 5:
            if elevation_diff > 0:
                instructions.append("Hold phone higher")
            else:
                instructions.append("Hold phone lower")
        
        # Distance guidance
        distance_diff = target['distance'] - current_pose['distance']
        if abs(distance_diff) > 0.1:
            if distance_diff > 0:
                instructions.append("Step back")
            else:
                instructions.append("Move closer")
        
        return instructions
```

## Phase 6: Testing and Validation (1-2 weeks)

### 6.1 Unit Tests
**File**: `tests/test_6dof_metrics.py` (new)

### 6.2 Integration Tests
**File**: `tests/test_mobile_pipeline.py` (new)

### 6.3 Performance Benchmarking
- Model accuracy on validation set
- Inference speed on mobile devices
- Memory usage optimization

## Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 1-2 weeks | Extended data pipeline, multi-angle feature extraction |
| Phase 2 | 2-3 weeks | 6DOF model architecture, loss functions, metrics |
| Phase 3 | 1 week | Updated training and validation scripts |
| Phase 4 | 1 week | Enhanced inference and visualization |
| Phase 5 | 2-3 weeks | Mobile optimization, real-time processing, guidance logic |
| Phase 6 | 1-2 weeks | Testing, validation, performance optimization |

**Total Estimated Time: 8-12 weeks**

## Risk Mitigation

### Technical Risks:
1. **Model Complexity**: Multi-task learning may be harder to optimize
   - *Mitigation*: Start with separate models for each task, then combine
2. **Limited Tilt Data**: PASCAL3D+ may not have tilt annotations
   - *Mitigation*: Initially set tilt to 0, add synthetic data later
3. **Mobile Performance**: 6DOF model may be too slow for real-time use
   - *Mitigation*: Model pruning, quantization, and architecture optimization

### Data Risks:
1. **Insufficient Training Data**: Need more diverse poses
   - *Mitigation*: Data augmentation, synthetic data generation
2. **Domain Gap**: PASCAL3D+ vs real mobile camera images
   - *Mitigation*: Fine-tuning on mobile-captured data

## Success Metrics

1. **Accuracy**: MAE < 10° for all angle estimations
2. **Speed**: < 100ms inference time on mobile devices
3. **Guidance Quality**: User study showing improved photo quality
4. **Robustness**: Works across different lighting conditions and car types

This plan provides a comprehensive roadmap to extend the current azimuth-only model to a full 6DOF pose estimation system suitable for mobile guidance applications.

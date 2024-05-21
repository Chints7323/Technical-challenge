from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load a pretrained YOLOv8 model
    torch.cuda.empty_cache()
    model = YOLO("yolov8n.pt")

    # Training parameters
    data_path = "yolo_custom_dataset/custom_data.yaml"
    epochs = 30  
    batch_size = 8
    image_size = 400
    patience = 5
    initial_learning_rate = 0.001  

    # Training the model with specified parameters
    model.train(
        data=data_path, 
        epochs=epochs, 
        batch=batch_size, 
        imgsz=image_size, 
        patience=patience,
        lr0=initial_learning_rate,
        optimizer='Adam',     # Using Adam optimizer
        warmup_epochs=2,      # Warmup to stabilize training
        warmup_momentum=0.8,  # Initial warmup momentum
        warmup_bias_lr=0.1,   # Learning rate for bias during warmup
        box=7.5,              # Box loss gain
        cls=0.5,              # Class loss gain
        dfl=1.5,              # Distribution Focal Loss gain
        label_smoothing=0.1,  # Apply label smoothing
        nbs=64,               # Nominal batch size for normalization
        rect=True,            # Rectangular training
        cache='ram',          # Cache images in RAM for faster training
        device=0              # Use GPU
    )

    # Validate the model performance
    metrics = model.val()

    # Save the model after training
    model.save("yolo_custom_dataset/model.pt")

    print("Training completed and model saved.")

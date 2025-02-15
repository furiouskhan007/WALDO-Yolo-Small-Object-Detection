from ultralytics import YOLO

# Load the model
model = YOLO("WALDO30_yolov8m_640x640.pt")

# Check if the model is loaded correctly
#print(model)
# Run inference
results = model("img.webp", save=True)  # Replace with your image file

# Display results
for r in results:
    r.show()  # Show detections

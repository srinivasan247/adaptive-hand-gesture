import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_gesture_heatmap():
    """
    Generates a professional heatmap representing the distribution of 
    detected hand landmarks across a normalized camera frame.
    """
    print("Generating Gesture Dataset Heatmap...")
    
    # 1. Generate multi-cluster dummy data to simulate real hand landmark distributions
    # In a real scenario, these would be (x, y) coordinates from Mediapipe landmarks
    num_samples = 2000
    
    # Center Cluster (Palm/Wrist)
    palm_x = np.random.normal(320, 40, num_samples // 2)
    palm_y = np.random.normal(240, 50, num_samples // 2)
    
    # Index Finger Cluster (Common movement area)
    finger_x = np.random.normal(350, 60, num_samples // 2)
    finger_y = np.random.normal(150, 70, num_samples // 2)
    
    # Combine data
    x = np.concatenate([palm_x, finger_x])
    y = np.concatenate([palm_y, finger_y])
    
    # Clip to frame boundaries (640x480 typical webcam)
    x = np.clip(x, 0, 640)
    y = np.clip(y, 0, 480)

    # 2. Setup Plot Aesthetics
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Generate Heatmap (Kernel Density Estimate)
    sns.kdeplot(
        x=x, y=y, 
        cmap="rocket", 
        fill=True, 
        thresh=0, 
        levels=100,
        alpha=0.8,
        ax=ax
    )
    
    # Add contour lines for definition
    sns.kdeplot(x=x, y=y, color="white", linewidths=0.5, alpha=0.3, ax=ax)

    # 3. Labeling and Formatting
    plt.title("Spatial Distribution of Gesture Landmarks (Heatmap)", fontsize=16, pad=20, color='white')
    plt.xlabel("Webcam X Coordinate (pixels)", fontsize=12)
    plt.ylabel("Webcam Y Coordinate (pixels)", fontsize=12)
    
    # Invert Y axis to match screen/image coordinates (0,0 is top-left)
    plt.gca().invert_yaxis()
    
    # Add a subtle grid
    plt.grid(True, linestyle='--', alpha=0.1, color='white')
    
    # Annotate key regions
    plt.annotate('Active Interaction Zone', xy=(330, 180), xytext=(450, 100),
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, color='white')

    # Save the output
    output_path = "gesture_heatmap.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='black')
    print(f"Success! Heatmap saved as '{output_path}'")
    
    plt.show()

if __name__ == "__main__":
    # Ensure dependencies are available
    try:
        import seaborn
        import matplotlib
    except ImportError:
        print("\nERROR: Missing dependencies. Please run:")
        print("pip install matplotlib seaborn numpy\n")
    else:
        generate_gesture_heatmap()

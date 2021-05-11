# CarCam
CarCam is a computer vision pipeline for autonomous vehicles, built with PyTorch and OpenCV. CarCam uses machine learning to detect lanes, track road entities (like other vehicles), and assess collision risks with these road entities. 

CarCam relies on input from only one camera, instead of relying on several cameras, multiple sensors, and/or complex systems like LiDAR, making safe self-driving simple, cheap, and easily accessible to all vehicles.

Lanes are marked with blue lines, and road entities are bounded in green or red boxes. A road entity with which a collision is unlikely to occur is bounded in a green box and is deemed 'safe'. A road entity with which a collision may occur is bounded in a red box and is deemed 'risky'. 

# How to Run
1. Download/clone the repository. 
2. Create a Python virtual environment and install the packages listed in the `requirements.txt` file, using `conda`, `pip`, etc. To do this step with `conda`, run this command: `conda create --name <your env name> --file requirements.txt`. 
3. Run this command: `cd config/ && sh ./download_weights.sh` to download the weights of the YOLO object detection model. 
4. Walk through the Jupyter notebook `carcam.ipynb` or run the `carcam.py` file with this command: `python3 carcam.py`. Depending on the specs of your system, this may take a while, so please be patient. Your results will be ready soon! 

# View Results
Download and play the video `output.mp4`. The same video can also be watched on YouTube, here: https://youtu.be/IRfuNHZG8CQ. Enjoy! 

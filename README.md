# XAI-in-Self-Driving-Car
This project implements Explainable AI in self driving cars by the use of post-hoc explanations.

---
 This repository contains Python scripts for training a machine learning model, running simulations, performing explanations using LIME, and conducting object detection.

## Contents

1. **model_training.py**: Python script for training the machine learning model.
2. **drive.py**: Python script to run the trained model in a simulation environment.
3. **limepro.py**: Python script demonstrating LIME (Local Interpretable Model-agnostic Explanations) for explaining model predictions
4. **object_detection.py**: Python script for performing object detection tasks.

## Usage

- **Model Training**: Use `model_training.py` to train your machine learning model. Adjust parameters and datasets as needed within the script.
  

- **Simulation**: Run simulations using `drive.py` to evaluate the trained model in a simulated environment such as Udacity.

- **Explanations**: Explore explanations for model predictions with `limepro.py` using the LIME framework.

- **Object Detection**: Perform object detection tasks using `object_detection.py`.

- **Data Collection**: The data used to train the model are image files obtained by using the record feature available in both airsim and udacity simulators. The data has to preprocessed and augmented.

## Additional Resources

- **Udacity Nanodegree Program**: The simulation environment used is part of the Udacity Nanodegree program. For more details and to download visit [Udacity Nanodegree Program](https://udacity.com/drive)
  and to download-


  [Linux](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip)

  [Mac](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip)

  [Windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip).

- **AirSim Simulator**: For simulation purposes, download the AirSim simulator from [AirSim GitHub Repository](https://github.com/microsoft/AirSim).


---
Both simulation environments are open-source.

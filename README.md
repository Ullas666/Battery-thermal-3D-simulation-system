 3D Physics-Informed Battery Digital Twin

 Overview

This repository contains a Physics-Informed Fourier Neural Operator (FNO) designed to act as a real-time Digital Twin for EV battery packs.

Unlike traditional Finite Element Analysis (FEA) or CFD which take hours to resolve 3D heat equations, this AI model predicts the volumetric Temperature and State of Health (SOH) in milliseconds. It enables "X-Ray" vision into the battery core, allowing R&D engineers to visualize internal short circuits, cooling failures, and propagation events instantly.

 Key Features

   Real-Time Physics Solver: Solves the 3D Heat Diffusion Equation and Arrhenius Degradation dynamics instantly using spectral convolutions.

   Volumetric Health Monitoring: Predicts internal chemical degradation (SOH) based on thermal stress, identifying "silent" failures deep inside the cell stack.

   Dynamic Simulation Mode: Simulates moving faults (e.g., thermal runaway propagation or moving busbar shorts) with interactive 3D particle clouds.

   Smart Visualization: Features a "Peeling Algorithm" that renders internal fault cores by making healthy outer layers transparent.

   Lightweight Deployment: The trained model is a portable .pth file suitable for deployment on Edge AI devices or onboard Battery Management Systems (BMS).

Methodology: Fourier Neural Operator (FNO)
  
This project moves beyond standard CNNs by utilizing FNOs, which learn the solution operator for Partial Differential Equations (PDEs) in the frequency domain.

Input: 3D Tensor representing initial heat distribution.

Spectral Layer: Performs FFT (Fast Fourier Transform) to mix global information.

Physics Enforcement: Trained on analytical solutions to the Heat Equation ($\partial T/\partial t = \alpha \nabla^2 T$).

Output: 3D Volumetric prediction of Temperature and SOH at time $t$.

 Installation & Usage

Prerequisites

Python 3.8+

NVIDIA GPU (Recommended for training)

pip install torch numpy matplotlib pygame PyOpenGL PyOpenGL_accelerate


1. Train the Model

Generate synthetic physics data and train the neural operator:

python surrogate.py


Output: Saves battery_fno_3d_model.pth

2. Run the Diagnostic Engine

Launch the interactive visualization tool:

python battery.py


 Modes of Operation

The diagnostic tool supports three modes:

Manual Mode: Input specific $(X, Y, Z)$ coordinates and heat intensity to probe specific cells.

Gallery Mode: View pre-calculated scenarios like "Core Meltdown" or "Corner Impact."

Dynamic Mode: Define a start and end point to simulate a fault moving through the pack over time (Trajectory Simulation).

 Future Roadmap

 Integration with real-time sensor streams (MQTT/CAN bus).

 Extension to electrochemical states (Lithium-ion concentration).

 Mechanical stress prediction (Swelling).

Developed for Advanced Battery R&D.

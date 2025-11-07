# 🤖 DiffusionHybridControl
**Hybrid Predictive Sampling + Diffusion Policy Control System for Adaptive Robotic Arms**

This repository contains the implementation and experimental results of the research paper:

> **Integrating Predictive Sampling and Diffusion Policy for Adaptive Control in Simulated Robotic Arms**  
> *Author: Arjun R — Department of Robotics and Automation Engineering, Rajalakshmi Engineering College, Chennai, India*

---

## 🧠 Abstract

Traditional robotic control techniques like PID and Model Predictive Control (MPC) provide excellent stability and interpretability but struggle to adapt to dynamic or uncertain conditions.  
In contrast, modern generative learning approaches—particularly **Diffusion Models**—excel in producing smooth, temporally coherent motion but lack explicit physical grounding.  

This project presents a **hybrid adaptive control framework** that unites the deterministic optimization of **Predictive Sampling (PS)** with the generative adaptability of **Diffusion Policy (DP)**.  
Predictive Sampling provides short-horizon foresight through model-based optimization, while the Diffusion Policy refines sampled trajectories through learned denoising, ensuring smooth and stable control signals.

Implemented on a **2R robotic arm simulated in MuJoCo**, this hybrid architecture demonstrates enhanced smoothness, energy efficiency, and stability compared to predictive control alone.  
The results indicate that combining model-based reasoning with data-driven refinement leads to more human-like, adaptive motion in robotic systems.

---

## 🧩 Project Structure

DiffusionHybridControl/
├── env_robotic_arm.py # MuJoCo environment class for the 2R arm
├── hybrid_control.py # Main hybrid PS + DP controller
├── predictive_sampling.py # Predictive Sampling (PS) module
├── diffusion_policy.py # Diffusion Policy (DP) model
├── train_diffusion.py # Script to train the diffusion model
├── generate_data.py # Generate expert trajectories using PS
├── plot_results.py # Visualize joint and trajectory results
├── expert_data.npy # Expert state-action dataset
├── diffusion_policy.pth # Trained diffusion policy weights
├── 2R_robotic_arm.xml # MuJoCo XML model of the 2R arm
├── requirements.txt # Dependencies
├── LICENSE # MIT License


---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- [MuJoCo 3.1](https://mujoco.org/)
- Compatible GPU (optional but recommended)

### Setup
```bash
git clone https://github.com/arjunros/DiffusionHybridControl.git
cd DiffusionHybridControl
pip install -r requirements.txt
```


Running the Simulation
1️⃣ Generate Expert Data
python generate_data.py

2️⃣ Train the Diffusion Policy
python train_diffusion.py

3️⃣ Run the Hybrid Controller
python hybrid_control.py


You’ll see:

A MuJoCo window showing the robotic arm motion

Generated graphs for joint-space dynamics and end-effector trajectory

📊 Results
Performance Metrics
Metric	Predictive Sampling	Hybrid (PS+DP)
Trajectory Smoothness	0.61	0.87
Control Energy (Nm²)	1.00	0.73
Stability (Var(qvel))	0.39	0.21
Adaptability Score	0.54	0.91
Visual Results

Joint Dynamics

Displays the joint angles, velocities, and applied torques across time, highlighting adaptive damping and smooth motion transitions.

End-Effector Trajectory

The hybrid controller generates continuous, curved motion paths indicating stable, adaptive workspace traversal.

🧩 Key Features

✅ Combines Predictive Sampling (model-based foresight) with Diffusion Policy (data-driven refinement)

✅ Generates physically consistent and smooth robot motion

✅ Reduces control noise and energy consumption

✅ Fully implemented in Python + MuJoCo + PyTorch

✅ IEEE-formatted research paper included in /paper/

🧠 Research Summary

This research establishes a foundation for Diffusion-Enhanced Hybrid Control in robotics:

Predictive Sampling performs deterministic short-horizon optimization

Diffusion Policy introduces probabilistic adaptability

Hybridization results in improved smoothness and stability

This combination points toward the next generation of control systems that merge optimization-based intelligence with generative priors—moving closer to adaptive, human-like robot motion.

📘 Citation

If you use this work in your research, please cite it as:

@article{arjun2025hybridcontrol,
  title={Integrating Predictive Sampling and Diffusion Policy for Adaptive Control in Simulated Robotic Arms},
  author={Arjun, R.},
  journal={IEEE Conference Paper (Under Review)},
  year={2025}
}

🧑‍💻 Author

Arjun R
Robotics and Automation Engineer
Rajalakshmi Engineering College, Chennai
📧 Email: itsrarjun@outlook.com

🌐 Medium — @itsrarjun

💼 LinkedIn — Arjun R

🧩 License

This project is released under the MIT License
.

🌟 Acknowledgements

MuJoCo team for providing the simulation environment

The Robotics and Automation Department, Rajalakshmi Engineering College, for continuous support and mentorship

OpenAI’s diffusion model research community for foundational insights

📈 Future Scope

The current hybrid system is implemented on a 2R planar arm, but it can be extended to:

6-DoF manipulators (UR5, Franka Emika Panda)

Quadruped robots for adaptive locomotion

Real-time reinforcement fine-tuning

Integration with LLM-based planning (Nexomation AI)

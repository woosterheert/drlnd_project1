# Deep Reinforcement Learning Nanodegree - Project 1

This repository contains my submission for the first project of the DRL Nanodegree

## Assignment
The challenge is the solve Unity's 'BananaWorld'. This environment consists of blue and yellow banana's randomly 
positioned in a square world. The agent needs to learn to pick up the yellow banana's while staying away from the blue ones.
For each yellow banana collected the agent receives one point. For each blue banana a point is withdrawn.
The agent gets 300 steps to collect as many points as possible. The challenge is solved if the agent manages to collect
on average 13 points over 100 consecutive episodes.

The agent has a 37-dimensional input at its disposal to learn from.
These 37 numbers represent the agent's velocity and a vectorized representation of what the agent sees in front of him.
The agent can choose from four actions: forward, backward, turn left and turn right.

## Solution
This movie shows the resuling agent of my solution:
<video width="640" height="356" controls>
  <source src="solved_agent_480.mov" type="video/mp4">
</video>


## Installation
In order to run the files yourself, please follow these instructions:

- Create a python 3.6 environment
- Install the dependencies in the requirements.txt file in this environment
- Download the necessary Unity Environment from one of the following links and install it in the same directory as the code:
  - Linux: click here
  - Mac OSX: click here
  - Windows (32-bit): click here (64-bit): click here
  
    

#!/bin/bash

# Update the system's package index
sudo apt update
sudo apt upgrade -y

# Install required packages
sudo apt install -y software-properties-common build-essential

# Add the Deadsnakes PPA for newer Python versions
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-distutils

# Install pip for Python 3.11
wget https://bootstrap.pypa.io/get-pip.py
sudo python3.11 get-pip.py

# Install and set up virtual environment
sudo pip3 install virtualenv
virtualenv venv --python=python3.11
source venv/bin/activate

# Install dependencies from the requirements.txt file
pip install -r requirements.txt

# Run the Streamlit app
streamlit run server.py --server.port 8501 --server.address 0.0.0.0

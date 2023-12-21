# Speech-to-Text Machine Learning Models

This repository contains the implementation of machine learning models for converting speech to text. We use a combination of CNN and LSTM architectures in the TensorFlow framework. The models were trained on the LJSpeech dataset.

## Project Structure

The repository is organized into three main folders:

1. **ASR-model:**
   - Contains the Jupyter notebook file (`model.ipynb`) of the trained speech-to-text model.
   - [Link to ASR-model folder](./ASR-model)

2. **Deploy-model-Flask-API:**
   - Contains the Python file (`app.py`) for deploying the model as a Flask API.
   - [Link to Deploy-model-Flask-API folder](./Deploy-model-Flask-API)

3. **Model-experiments:**
   - Contains several Jupyter notebook files documenting model building experiments.
   - [Link to Model-experiments folder](./Model-experiments)

## Usage

### ASR-model

To use the pre-trained model for speech-to-text conversion, follow the instructions in the ASR-model folder.

```bash
cd ASR-model
# Run the Jupyter notebook
jupyter notebook model.ipynb


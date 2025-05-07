# CapGen 📌

## Overview
CapGen is an AI-powered tool that generates trendy and creative captions for your images! Upload a photo, the app will describe it, detect its vibe, and craft engaging captions that can be used for your social media platforms like Pinterest or Instagram.

## Demo!
[![CapGen Demo](https://img.youtube.com/vi/lNRoQ0mZ7Bo/0.jpg)](https://www.youtube.com/watch?v=lNRoQ0mZ7Bo)

## How It Works
The app uses a combination of computer vision and language models:

1. BLIP is used to generate a natural-language description of the uploaded image.

2. CLIP compares the image to a set of predefined "vibes" and selects the closest match.

3. GPT-4o-mini generates three creative and hashtag-friendly captions based on the generated description, optional user description, and generated vibe.

4. Optionally, users can refine captions to their liking by communicating with genAI chat interface, which will regenerate captions to match the user's requirements.

## Features
* Image Upload: Upload images for analysis and captioning.

* Automatic Generative Image Description and Image Vibe Detection.

* AI-Powered Captioning.

* Chat-Based Refinement.

## Requirements
- Python 3, PyTorch
- Gradio
- BLIP, CLIP
- dotenv
- Python Imaging Library
- OpenAI

You can install the necessary dependencies and run the application with the following commands:

```bash
pip install -r requirements.txt
python app.py
```

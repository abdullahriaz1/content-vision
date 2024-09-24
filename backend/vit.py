import os
import cv2
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel, PegasusTokenizer, PegasusForConditionalGeneration
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import numpy as np
from transformers.utils import logging
logging.set_verbosity_error() 
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

def blip():
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
  model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
  
  def generate_caption(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
  
  frames_folder = "frames"

  # Iterate over images in the folder
  captions = []
  for image_file in os.listdir(frames_folder):
      image_path = os.path.join(frames_folder, image_file)
      if os.path.isfile(image_path):
          caption = generate_caption(image_path)
          print(f"{image_file}: {caption}")
          captions.append(caption)
  captions_concat = " ".join(set(" ".join(captions).split()))
  return captions_concat

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    count = 0
    success, image = video.read()
    
    while success:
        if count % interval == 0:
            frame_file = os.path.join(output_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_file, image)
        success, image = video.read()
        count += 1
    
    video.release()

def transcribe_audio(video_path):
    audio_path = "output_audio.wav"
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    transcript = recognizer.recognize_google(audio_data)
    
    os.remove(audio_path)  # Clean up audio file
    return transcript

def analyze_frames(frames_dir):
    model_name = "google/vit-base-patch16-224"
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    frame_features = []
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    
    for frame_file in frame_files:
        image = Image.open(frame_file).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            frame_features.append(outputs.last_hidden_state.cpu().numpy())

    return frame_features

def summarize_text(text, captions):
    try:
        # Load tokenizer and model
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

        # Force usage of CPU
        device = torch.device("cpu")
        model.to(device)

        # Add a prompt to the text (PEGASUS doesn't need a prompt for summarization)
        prompt = "Summarize the following conversation: "
        full_text = f"{prompt} {captions} {text}"

        # Tokenize input text
        inputs = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=100,   # Adjust max_length as needed
            min_length=30,    # Adjust min_length as needed
            length_penalty=1.5,
            num_beams=4,
            early_stopping=True
        )

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        summarizer = pipeline("summarization")

        summary = summarizer(full_text, max_length=100, min_length=0, do_sample=False)
        #print(summary[0]['summary_text'])
        return summary[0]['summary_text']

    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def main(video_path):
    frames_dir = 'frames'
    
    print("Extracting frames from video...")
    extract_frames(video_path, frames_dir)
    
    print("Transcribing audio from video...")
    transcript = transcribe_audio(video_path)
    #transcript = " Miss green I am afraid your case just got a lot more complicated than expected so does this mean I will not get the loan I thought you were the most qualified adviser I didn't say that I will do my best to obtain a loan for you but it might take a little longer"
    print("Transcript:", transcript)

    '''print("Analyzing frames with Vision Transformer...")
    frame_features = analyze_frames(frames_dir)
    print(f"Extracted features for {len(frame_features)} frames.")
    '''
    #captions = blip()
    captions = ""
    print(captions)
    print("Summarizing transcript ...")
    summary = summarize_text(transcript, captions)
    print("Summary:", summary)
    
    # Clean up the frames directory
    for file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, file))
    os.rmdir(frames_dir)

if __name__ == "__main__":
    office_vid = "conversation_sample-1080p-.mp4 (240p).mp4"
    elon_vid = "Elon Musk on the existence of a soul - Lex Fridman Podcast Clips.mp4"
    video_path = elon_vid
    main(video_path)

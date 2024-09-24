import os
import shutil
import cv2
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, PegasusTokenizer, PegasusForConditionalGeneration
import gc

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from the video at a specified frame rate and save them to the output directory.
    """
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
    """
    Extract audio from the video and transcribe it using speech recognition.
    """
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

def load_clip_model():
    """
    Load the CLIP model and processor.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_pegasus_model():
    """
    Load the Pegasus model and tokenizer for text summarization.
    """
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

def analyze_with_clip(images_dir, transcript):
    """
    Analyze images with CLIP and summarize findings.
    """
    model, processor = load_clip_model()
    
    # Load and process images
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.jpg')]
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    texts = [transcript] * len(images)

    # Prepare inputs for CLIP
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    device = torch.device("cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device)
    
    # Get CLIP model outputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    return 


def summarize_text(text):
    """
    Summarize the given text using Pegasus with a specific prompt.
    """
    tokenizer, model = load_pegasus_model()
    device = torch.device("cpu")
    model.to(device)

    prompt = "Summarize the following text: "
    full_text = f"{prompt} {text}"

    inputs = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_length=150,
        min_length=50,
        length_penalty=1.5,
        num_beams=4
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def clear_cpu_memory():
    """
    Clear CPU memory by deleting unused objects and running garbage collection.
    """
    gc.collect()

def main(video_path):
    output_dir = 'frames'
    extract_frames(video_path, output_dir)
    transcript = transcribe_audio(video_path)
    print("Transcript:", transcript)

    # Optionally clear CPU memory
    clear_cpu_memory()

    summary = analyze_with_clip(output_dir, transcript)
    print("Analysis Summary:", summary)

    # Optionally clear CPU memory
    clear_cpu_memory()

    summary_text = summarize_text(transcript)
    print("Summary Text:", summary_text)
    

if __name__ == "__main__":
    video_path = 'conversation_sample-1080p-.mp4 (240p).mp4'  # Replace with your video file path
    main(video_path)

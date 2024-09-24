from moviepy.editor import VideoFileClip
import whisper
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers.utils import logging
import torch
logging.set_verbosity_error() 

def extract_audio_from_video(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as a WAV file.
    """
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        print(f"Audio extracted and saved to {audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def transcribe_audio(audio_path):
    """
    Transcribes an audio file using OpenAI's Whisper model.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

'''def summarize_text_with_prompt(text):
    """
    Summarizes the given text using BART with a specific prompt.
    """
    try:
        # Load tokenizer and model
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        # Force usage of CPU
        device = torch.device("cpu")
        model.to(device)

        # Add a prompt to the text
        prompt = "Who is speaking in the following text: "
        full_text = f"{prompt} {text}"

        # Tokenize input text
        inputs = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=100,
            min_length=30,
            length_penalty=1.5,
            num_beams=4,
            #early_stopping=True
        )

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None
        '''
        
def summarize_text_with_prompt(text):
    """
    Summarizes the given text using PEGASUS with a specific prompt.
    """
    try:
        # Load tokenizer and model
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
        model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

        # Force usage of CPU
        device = torch.device("cpu")
        model.to(device)

        # Add a prompt to the text (PEGASUS doesn't need a prompt for summarization)
        prompt = "Summarize the following conversation: "
        full_text = f"{prompt} {text}"

        # Tokenize input text
        inputs = tokenizer(full_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate summary
        summary_ids = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=400,   # Adjust max_length as needed
            min_length=0,    # Adjust min_length as needed
            length_penalty=1.5,
            num_beams=4,
            early_stopping=True
        )

        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def main():
    video_path = "conversation_sample-1080p-.mp4 (240p).mp4"
    audio_path = "output_audio.wav"
    
    # Step 1: Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    
    # Step 2: Transcribe audio
    transcription = transcribe_audio(audio_path)
    
    if transcription:
        print("Transcription:")
        print(transcription)
        
        # Step 3: Summarize transcription
        summary = summarize_text_with_prompt(transcription)
        
        if summary:
            print("\nSummary:")
            print(summary)
        else:
            print("Summary generation failed.")
    else:
        print("Transcription failed.")

if __name__ == "__main__":
    main()

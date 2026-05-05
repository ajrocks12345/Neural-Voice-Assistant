import pickle
import speech_recognition as sr
import numpy as np

# Ensure classes are available for unpickling
from lstm_numpy import LSTMRNN
from nlp_preprocessor import NLPPreprocessor

def execute_command(intent):
    """Simple stub to execute actions based on intent."""
    print(f"\n---> ACTION: Executing '{intent}' command <---")
    if intent == 'weather':
        print("     Fetching latest weather forecast...")
    elif intent == 'lights':
        print("     Toggling smart home lights...")
    elif intent == 'greeting':
        print("     Hello! How can I help you today?")
    elif intent == 'stop':
        print("     Shutting down. Goodbye!")
    else:
        print("     Unknown action.")

def main():
    print("Loading pre-trained intent model and preprocessor...")
    try:
        with open('intent_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessor.pkl', 'rb') as f:
            prep = pickle.load(f)
    except FileNotFoundError:
        print("Error: Could not find trained model or preprocessor. Run train_intent_model.py first.")
        return

    recognizer = sr.Recognizer()

    print("\n=== Voice Assistant is LIVE ===")
    print("Speak into your microphone. Say 'stop' or 'shut down' to exit.")
    
    with sr.Microphone() as source:
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready! Listening...")

        while True:
            try:
                # Listen to the microphone
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                print("Processing speech...")
                # Transcribe using Google Web Speech API
                text = recognizer.recognize_google(audio).lower()
                print(f"You said: '{text}'")
                
                # Preprocess text to sequence
                seq = prep.text_to_sequence(text)
                onehot_seq = prep.sequence_to_onehot(seq)
                
                # Predict intent using pure NumPy LSTM
                pred_idx = model.predict(onehot_seq)
                intent = prep.idx_to_intent[pred_idx]
                
                print(f"Predicted Intent: [{intent.upper()}]")
                execute_command(intent)
                
                if intent == 'stop':
                    break
                
                print("\nListening...")

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("Sorry, I didn't catch that.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

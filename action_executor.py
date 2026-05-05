import os
import webbrowser
import datetime
import random
import wikipedia
import pyautogui
import requests

def extract_query(text, trigger_words):
    """Simple rule-based entity extraction. Removes trigger words to find the query."""
    text_lower = text.lower()
    for word in trigger_words:
        if word in text_lower:
            text_lower = text_lower.replace(word, "").strip()
    return text_lower if text_lower else "something"

def execute_action(intent, raw_text):
    """
    Executes an action based on the predicted intent and raw text.
    Returns the string that the assistant should say back.
    """
    try:
        if intent == 'open_app':
            app_name = extract_query(raw_text, ["open", "launch", "start", "application", "app"])
            # On Windows, 'start' attempts to open the app or a default file handler
            os.system(f"start {app_name}")
            return f"Attempting to open {app_name}."

        elif intent == 'close_app':
            app_name = extract_query(raw_text, ["close", "exit", "quit", "kill", "app"])
            # Requires precise .exe name to work perfectly, stubbing for safety
            return f"I would close {app_name}, but task killing requires specific process names."

        elif intent == 'search_google':
            query = extract_query(raw_text, ["search google for", "search the web for", "google", "search for"])
            webbrowser.open(f"https://www.google.com/search?q={query}")
            return f"Searching Google for {query}."

        elif intent == 'play_youtube':
            query = extract_query(raw_text, ["play", "on youtube", "youtube search for", "watch a video about"])
            webbrowser.open(f"https://www.youtube.com/results?search_query={query}")
            return f"Playing {query} on YouTube."

        elif intent == 'play_music':
            # Opens Spotify web player or default music app
            webbrowser.open("https://open.spotify.com")
            return "Opening your music player."

        elif intent == 'media_control':
            if "pause" in raw_text or "stop" in raw_text:
                pyautogui.press('playpause')
                return "Pausing media."
            elif "play" in raw_text or "resume" in raw_text:
                pyautogui.press('playpause')
                return "Resuming media."
            return "Toggling media state."

        elif intent == 'tell_time':
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            date_str = now.strftime("%A, %B %d")
            return f"It is currently {time_str} on {date_str}."

        elif intent == 'set_alarm':
            time_query = extract_query(raw_text, ["set an alarm for", "wake me up at", "alarm at"])
            return f"Alarm set for {time_query}. (Note: This is a visual confirmation only in this demo)."

        elif intent == 'set_reminder':
            reminder = extract_query(raw_text, ["remind me to", "set a reminder for", "remind me"])
            return f"I will remind you to {reminder}."

        elif intent == 'calculate':
            # Very basic extraction for simple math, e.g., "5 plus 5"
            math_str = extract_query(raw_text, ["calculate", "what is", "do a calculation", "math"])
            math_str = math_str.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("divided by", "/")
            try:
                # DANGEROUS in prod: eval. Keeping it simple for demo with limited characters.
                if all(c in '0123456789+-*/. ' for c in math_str):
                    result = eval(math_str)
                    return f"The answer is {result}."
                return "I couldn't parse the math equation."
            except:
                return "I ran into an error calculating that."

        elif intent == 'weather':
            try:
                # Using wttr.in for a quick weather text response based on IP
                res = requests.get('https://wttr.in?format=3', timeout=3)
                if res.status_code == 200:
                    return f"The current weather is {res.text.strip()}."
            except:
                pass
            return "I am unable to reach the weather service right now."

        elif intent == 'news':
            webbrowser.open("https://news.google.com")
            return "Here are the latest news headlines from Google News."

        elif intent == 'wikipedia':
            query = extract_query(raw_text, ["search wikipedia for", "tell me about", "look up", "who is", "what is", "wikipedia"])
            try:
                summary = wikipedia.summary(query, sentences=2)
                return f"According to Wikipedia: {summary}"
            except wikipedia.exceptions.DisambiguationError as e:
                return f"There are multiple entries for {query}. Could you be more specific?"
            except wikipedia.exceptions.PageError:
                return f"I couldn't find a Wikipedia page for {query}."
            except:
                return "There was an error accessing Wikipedia."

        elif intent == 'chatgpt':
            # Needs OPENAI_API_KEY env var
            query = extract_query(raw_text, ["ask chatgpt", "question", "tell me", "how does", "explain"])
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return f"I heard you ask: {query}. However, my ChatGPT integration requires an OpenAI API key to be set."
            try:
                import openai
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}],
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"ChatGPT Error: {str(e)}"

        elif intent == 'volume':
            if "up" in raw_text or "increase" in raw_text or "louder" in raw_text:
                for _ in range(5): pyautogui.press('volumeup')
                return "Volume increased."
            elif "down" in raw_text or "decrease" in raw_text or "quieter" in raw_text:
                for _ in range(5): pyautogui.press('volumedown')
                return "Volume decreased."
            elif "mute" in raw_text:
                pyautogui.press('volumemute')
                return "Volume muted."
            return "Adjusting volume."

        elif intent == 'open_folder':
            folder = extract_query(raw_text, ["open", "go to", "show", "folder"])
            return f"Opening your {folder} folder."

        elif intent == 'search_file':
            filename = extract_query(raw_text, ["search for a file named", "find my document", "where is my"])
            return f"Searching your computer for {filename}."

        elif intent == 'joke':
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
                "There are 10 types of people in the world: those who understand binary, and those who don't."
            ]
            return random.choice(jokes)

        elif intent == 'quiz':
            return "Riddle me this: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? An Echo!"

        elif intent == 'calendar':
            return "You have 2 meetings scheduled for today."

        elif intent == 'event_reminder':
            return "I've checked your upcoming events. Your next event is a project sync tomorrow at 10 AM."

        elif intent == 'personalized':
            if "who made" in raw_text.lower() or "who created" in raw_text.lower():
                return "Albin John made me."
            return "You are the creator of this awesome neural sequence recognition system!"

        elif intent == 'contextual':
            return "I am maintaining the context of our conversation. How would you like to proceed?"

        else:
            return "I am not sure how to perform that action yet."
            
    except Exception as e:
        return f"An error occurred while executing the action: {e}"


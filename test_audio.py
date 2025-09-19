import simpleaudio as sa
import time

try:
    wave_obj = sa.WaveObject.from_wave_file("alert.wav")
    print("Wave file loaded successfully.")
    play_obj = wave_obj.play()
    print("Playing sound...")
    play_obj.wait_done()
    print("Sound finished.")
except Exception as e:
    print(f"An error occurred: {e}")
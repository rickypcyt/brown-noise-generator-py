import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from scipy import signal

class BrownNoiseGenerator:
    def __init__(self):
        # Set up the main window
        self.root = tk.Tk()
        self.root.title("Brown Noise Generator")
        # Increase height to accommodate extra presets buttons
        self.root.geometry("300x350")
        
        # Audio settings
        self.sample_rate = 48000
        self.device = 4  # Change this to your desired device ID
        self.is_playing = False  # Start with playback off
        self.volume = 0.5

        # We use a one‐pole leaky integrator to produce brown noise.
        # Its transfer function is: H(z) = (1 - leak) / (1 - leak*z⁻¹)
        # (When leak is very close to 1, this approximates an integrator.)
        # We store the filter state (zi) to ensure continuity between callback blocks.
        self.zi = np.array([0], dtype=np.float32)
        
        # Default bass level slider value.
        self.bass_level = 100  
        self.update_leak_from_bass(self.bass_level)
        
        # Create the GUI elements
        self.create_gui()

    def create_gui(self):
        # --- Volume control ---
        volume_frame = ttk.LabelFrame(self.root, text="Volume", padding="10")
        volume_frame.pack(fill="x", padx=10, pady=5)
        self.volume_slider = ttk.Scale(
            volume_frame,
            from_=0,
            to=100,
            orient="horizontal",
            command=self.update_volume
        )
        self.volume_slider.set(50)  # 50% volume by default
        self.volume_slider.pack(fill="x")
        
        # --- Bass Level (affects noise “color”) ---
        bass_frame = ttk.LabelFrame(self.root, text="Bass Level", padding="10")
        bass_frame.pack(fill="x", padx=10, pady=5)
        self.bass_slider = ttk.Scale(
            bass_frame,
            from_=20,
            to=200,
            orient="horizontal",
            command=self.update_bass
        )
        self.bass_slider.set(100)  # Default is Neutral
        self.bass_slider.pack(fill="x")
        
        # --- Presets ---
        presets_frame = ttk.LabelFrame(self.root, text="Presets", padding="10")
        presets_frame.pack(fill="x", padx=10, pady=5)
        
        # Preset buttons:
        # Bright: Lower bass slider value → less integration, more high–frequency content.
        bright_button = ttk.Button(presets_frame, text="Bright", 
                                   command=lambda: self.set_preset(20))
        bright_button.pack(side="left", expand=True, padx=5, pady=5)
        
        # Neutral: Mid value for a balanced sound.
        neutral_button = ttk.Button(presets_frame, text="Neutral", 
                                    command=lambda: self.set_preset(100))
        neutral_button.pack(side="left", expand=True, padx=5, pady=5)
        
        # Deep: Higher bass slider value → more integration, smoother, deeper noise.
        deep_button = ttk.Button(presets_frame, text="Deep", 
                                 command=lambda: self.set_preset(200))
        deep_button.pack(side="left", expand=True, padx=5, pady=5)
        
        # Random: Sets the bass slider to a random value between 20 and 200.
        random_button = ttk.Button(presets_frame, text="Random", 
                                   command=self.random_preset)
        random_button.pack(side="left", expand=True, padx=5, pady=5)
        
        # --- Play/Stop button ---
        self.play_button = ttk.Button(
            self.root,
            text="Start",
            command=self.toggle_playback
        )
        self.play_button.pack(pady=20)
        
        # --- Status label ---
        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=5)
        
        # Ensure proper cleanup on window close.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def update_volume(self, value):
        self.volume = float(value) / 100.0

    def update_bass(self, value):
        # Convert value to float (it may come as a string from the slider)
        self.bass_level = float(value)
        self.update_leak_from_bass(self.bass_level)
        # (Optional) Reset the filter state if you want the noise to restart:
        # self.zi = np.array([0], dtype=np.float32)
    
    def update_leak_from_bass(self, bass_value):
        """
        Map the bass slider value (20–200) to a leak coefficient.
        When leak is close to 1, the integrator is “tighter” (more integration) 
        and the noise has a deeper, smoother quality.
        
        For example, we map:
          - bass_value=20   → leak=0.95 (bright sound)
          - bass_value=200  → leak=0.999 (deep, smooth noise)
        """
        self.leak = np.interp(bass_value, [20, 200], [0.95, 0.999])
        # Precompute filter coefficients for the one–pole filter:
        self.b_coef = np.array([1 - self.leak], dtype=np.float32)
        self.a_coef = np.array([1, -self.leak], dtype=np.float32)
    
    def set_preset(self, value):
        """Set the bass slider to a preset value and update accordingly."""
        self.bass_slider.set(value)
        self.update_bass(value)
    
    def random_preset(self):
        """Set a random bass value between 20 and 200."""
        value = np.random.uniform(20, 200)
        self.bass_slider.set(value)
        self.update_bass(value)
    
    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        
        # Generate a block of white noise samples (float32)
        white = np.random.randn(frames).astype(np.float32)
        # Pass the white noise through the one–pole leaky integrator.
        # signal.lfilter returns both the filtered output and the updated filter state.
        brown, self.zi = signal.lfilter(self.b_coef, self.a_coef, white, zi=self.zi)
        
        # Apply volume and send the data to the output stream.
        outdata[:] = (brown * self.volume).reshape(-1, 1)
    
    def toggle_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self.play_button.config(text="Stop")
            self.status_label.config(text="Playing")
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                device=self.device,
                blocksize=2048,  # Larger block size can improve stability
                latency='high'
            )
            self.stream.start()
        else:
            self.is_playing = False
            self.play_button.config(text="Start")
            self.status_label.config(text="Stopped")
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
    
    def on_close(self):
        if self.is_playing:
            self.toggle_playback()
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = BrownNoiseGenerator()
    app.run()

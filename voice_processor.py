import numpy as np

class VoiceProcessor:
    """
    Processes audio data using NumPy.
    In a real-world scenario, this would extract MFCCs.
    Here, it generates synthetic sequence data to simulate voice commands.
    """
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_data):
        """
        Simulates extracting features from audio data using NumPy.
        Returns a sequence of feature vectors.
        """
        # In a real app: 
        # 1. Pre-emphasis
        # 2. Framing
        # 3. Windowing (Hamming)
        # 4. FFT
        # 5. Mel Filterbank
        # 6. DCT (MFCCs)
        
        # Simplified simulation for demo:
        # We assume the audio is divided into 20ms frames
        num_frames = len(audio_data) // (self.sample_rate // 50) # 50 frames per sec
        features = []
        for i in range(num_frames):
            # Each frame is a (n_mfcc, 1) vector
            frame_feature = np.random.randn(self.n_mfcc, 1) * 0.1
            features.append(frame_feature)
        return features

    def generate_synthetic_data(self, num_samples=100, sequence_length=20):
        """
        Generates synthetic training data for voice commands.
        Command 0: 'Hello' (increasing intensity)
        Command 1: 'Stop' (decreasing intensity)
        """
        X = []
        Y = []
        
        for _ in range(num_samples):
            # Command 0: Hello
            seq_0 = [np.full((self.n_mfcc, 1), i/sequence_length) + np.random.randn(self.n_mfcc, 1) * 0.05 
                     for i in range(sequence_length)]
            X.append(seq_0)
            Y.append(0)
            
            # Command 1: Stop
            seq_1 = [np.full((self.n_mfcc, 1), (sequence_length-i)/sequence_length) + np.random.randn(self.n_mfcc, 1) * 0.05 
                     for i in range(sequence_length)]
            X.append(seq_1)
            Y.append(1)
            
        return X, Y

    def normalize(self, features):
        """Normalize features using NumPy"""
        feat_array = np.array(features)
        mean = np.mean(feat_array, axis=0)
        std = np.std(feat_array, axis=0) + 1e-8
        norm_features = (feat_array - mean) / std
        return [norm_features[i] for i in range(len(norm_features))]

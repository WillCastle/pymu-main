"""
Audio analysis tools for frequency, amplitude, and other characteristics.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import fft, rfft, rfftfreq
from scipy.io.wavfile import write


class FFTAnalyzer:
    """Frequency domain analysis using Fast Fourier Transform."""

    def __init__(self, sample_rate: int = 44100, window_size: int = 2048):
        """
        Initialise the FFT analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            window_size: Size of FFT window in samples
        """
        self.sample_rate = sample_rate
        self.window_size = window_size

    def compute_spectrum(self, samples: np.ndarray, use_window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency spectrum of a signal.

        Args:
            samples: Input audio samples
            use_window: Whether to apply a Hann window for better frequency resolution

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Apply window function to reduce spectral leakage if requested
        if use_window:
            window = np.hanning(min(len(samples), self.window_size))
            if len(samples) > len(window):
                # For long signals, just use the window size
                samples = samples[: len(window)]
            elif len(samples) < len(window):
                # For short signals, truncate the window
                window = window[: len(samples)]
            samples = samples * window

        # Compute FFT
        n = len(samples)
        spectrum = rfft(samples)
        magnitudes = np.abs(spectrum) * 2 / n  # Scale properly
        frequencies = rfftfreq(n, 1 / self.sample_rate)

        return frequencies, magnitudes

    def compute_spectrogram(
        self, samples: np.ndarray, frame_size: int = None, hop_size: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a spectrogram (time-frequency representation).

        Args:
            samples: Input audio samples
            frame_size: Size of each frame in samples (default: window_size)
            hop_size: Number of samples between successive frames (default: frame_size//4)

        Returns:
            Tuple of (times, frequencies, spectrogram)
        """
        if frame_size is None:
            frame_size = self.window_size
        if hop_size is None:
            hop_size = frame_size // 4

        # Use scipy's spectrogram function
        frequencies, times, spectrogram = signal.spectrogram(
            samples,
            fs=self.sample_rate,
            window="hann",
            nperseg=frame_size,
            noverlap=frame_size - hop_size,
            scaling="spectrum",
        )

        # Convert to dB scale
        spectrogram = 10 * np.log10(spectrogram + 1e-10)

        return times, frequencies, spectrogram

    def get_peak_frequency(self, samples: np.ndarray) -> float:
        """
        Find the peak frequency in a signal.

        Args:
            samples: Input audio samples

        Returns:
            Peak frequency in Hz
        """
        frequencies, magnitudes = self.compute_spectrum(samples)
        if len(frequencies) == 0:
            return 0.0

        peak_index = np.argmax(magnitudes)
        return frequencies[peak_index]

    def get_spectral_centroid(self, samples: np.ndarray) -> float:
        """
        Calculate the spectral centroid (brightness) of a signal.

        Args:
            samples: Input audio samples

        Returns:
            Spectral centroid in Hz
        """
        frequencies, magnitudes = self.compute_spectrum(samples)
        if np.sum(magnitudes) == 0 or len(frequencies) == 0:
            return 0.0

        # Weighted average of frequencies
        return np.sum(frequencies * magnitudes) / np.sum(magnitudes)


class PitchDetector:
    """Detects the fundamental pitch/frequency of audio signals."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the pitch detector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def autocorrelation(self, samples: np.ndarray) -> float:
        """
        Detect pitch using autocorrelation method.

        Args:
            samples: Input audio samples

        Returns:
            Detected fundamental frequency in Hz
        """
        # Ensure we have enough samples
        if len(samples) < 2:
            return 0.0

        # Calculate autocorrelation
        correlation = signal.correlate(samples, samples, mode="full")
        correlation = correlation[len(correlation) // 2 :]

        # Find the first peak after the zero lag
        min_lag = int(self.sample_rate / 2000)  # Min frequency ~2000Hz
        max_lag = int(self.sample_rate / 50)  # Max frequency ~50Hz

        # Trim to range of interest
        if max_lag >= len(correlation):
            max_lag = len(correlation) - 1

        # Find peak in correlation
        peaks, _ = signal.find_peaks(correlation[min_lag:max_lag])

        if len(peaks) > 0:
            first_peak = min_lag + peaks[0]
            return self.sample_rate / first_peak
        else:
            return 0.0

    def zero_crossings(self, samples: np.ndarray) -> float:
        """
        Estimate frequency using zero-crossing rate.

        Args:
            samples: Input audio samples

        Returns:
            Estimated frequency in Hz
        """
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(samples).astype(int))))

        # Calculate frequency: zero crossings per second / 2
        # (divide by 2 because each cycle has 2 crossings)
        if len(samples) > 1:
            duration = len(samples) / self.sample_rate
            return zero_crossings / (2 * duration)
        else:
            return 0.0

    def cepstrum(self, samples: np.ndarray) -> float:
        """
        Estimate pitch using cepstrum analysis.

        Args:
            samples: Input audio samples

        Returns:
            Estimated fundamental frequency in Hz
        """
        # Compute FFT
        spectrum = np.abs(fft(samples))

        # Compute log spectrum
        log_spectrum = np.log(spectrum + 1e-10)

        # Compute cepstrum (IFFT of log spectrum)
        cepstrum = np.abs(fft(log_spectrum))

        # Only use first half
        cepstrum = cepstrum[: len(cepstrum) // 2]

        # Define quefrency range for fundamental frequency
        min_quefrency = int(self.sample_rate / 2000)  # Min frequency ~2000Hz
        max_quefrency = int(self.sample_rate / 50)  # Max frequency ~50Hz

        # Ensure we don't exceed array bounds
        if max_quefrency >= len(cepstrum):
            max_quefrency = len(cepstrum) - 1

        if min_quefrency >= max_quefrency:
            return 0.0

        # Find the peak in the cepstrum in our range
        quefrency_range = cepstrum[min_quefrency:max_quefrency]

        if len(quefrency_range) > 0:
            peak_index = np.argmax(quefrency_range) + min_quefrency
            return self.sample_rate / peak_index
        else:
            return 0.0


class AmplitudeAnalyzer:
    """Analyzes amplitude characteristics of audio signals."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the amplitude analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def rms(self, samples: np.ndarray) -> float:
        """
        Calculate the RMS (Root Mean Square) amplitude of a signal.

        Args:
            samples: Input audio samples

        Returns:
            RMS amplitude value
        """
        if len(samples) == 0:
            return 0.0
        return np.sqrt(np.mean(np.square(samples)))

    def peak(self, samples: np.ndarray) -> float:
        """
        Find the peak amplitude of a signal.

        Args:
            samples: Input audio samples

        Returns:
            Peak amplitude value
        """
        if len(samples) == 0:
            return 0.0
        return np.max(np.abs(samples))

    def crest_factor(self, samples: np.ndarray) -> float:
        """
        Calculate the crest factor (peak to RMS ratio).

        Args:
            samples: Input audio samples

        Returns:
            Crest factor value
        """
        rms_val = self.rms(samples)
        if rms_val == 0:
            return 0.0
        return self.peak(samples) / rms_val

    def true_peak(self, samples: np.ndarray, oversample: int = 4) -> float:
        """
        Calculate the true peak value with oversampling.

        Args:
            samples: Input audio samples
            oversample: Oversampling factor

        Returns:
            True peak value
        """
        if len(samples) == 0:
            return 0.0

        # Upsample for true peak detection
        resampled = signal.resample(samples, len(samples) * oversample)
        return np.max(np.abs(resampled))

    def loudness_ebur128(self, samples: np.ndarray) -> float:
        """
        Estimate LUFS loudness based on EBU R128 standard (simplified).

        Args:
            samples: Input audio samples

        Returns:
            Integrated loudness in LUFS
        """
        if len(samples) == 0:
            return -100.0  # Very quiet

        # This is a simplified version; a full implementation would follow the EBU R128 spec
        # with K-weighting filter, gating, etc.

        # Create a simple K-weighting filter approximation
        b, a = signal.butter(2, [38 / (self.sample_rate / 2), 14000 / (self.sample_rate / 2)], "bandpass")
        k_weighted = signal.lfilter(b, a, samples)

        # Calculate mean square
        ms = np.mean(np.square(k_weighted))

        # Convert to LUFS (simplified)
        return -0.691 + 10 * np.log10(ms + 1e-10)

    def envelope(self, samples: np.ndarray, attack_ms: float = 5, release_ms: float = 50) -> np.ndarray:
        """
        Extract the amplitude envelope of a signal.

        Args:
            samples: Input audio samples
            attack_ms: Envelope attack time in milliseconds
            release_ms: Envelope release time in milliseconds

        Returns:
            Envelope as numpy array
        """
        if len(samples) == 0:
            return np.array([])

        # Calculate time constants
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)

        # Ensure at least 1 sample
        attack_samples = max(1, attack_samples)
        release_samples = max(1, release_samples)

        # Create coefficient multipliers
        attack_coef = np.exp(-1.0 / attack_samples)
        release_coef = np.exp(-1.0 / release_samples)

        # Process envelope
        envelope = np.zeros_like(samples)
        env_value = 0

        for i, sample in enumerate(samples):
            sample_abs = abs(sample)

            if sample_abs > env_value:
                # Attack phase
                env_value = attack_coef * env_value + (1 - attack_coef) * sample_abs
            else:
                # Release phase
                env_value = release_coef * env_value + (1 - release_coef) * sample_abs

            envelope[i] = env_value

        return envelope


class BeatDetector:
    """Analyzes rhythmic patterns and detects beats in audio signals."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the beat detector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def detect_tempo(self, samples: np.ndarray) -> float:
        """
        Estimate the tempo of an audio signal in BPM.

        Args:
            samples: Input audio samples

        Returns:
            Estimated tempo in beats per minute
        """
        if len(samples) < self.sample_rate:
            return 0.0

        # Calculate onset strength signal (energy differential)
        # First, get the envelope
        amplitude_analyzer = AmplitudeAnalyzer(sample_rate=self.sample_rate)
        envelope = amplitude_analyzer.envelope(samples, attack_ms=10, release_ms=50)

        # Calculate the first difference (derivative) of the envelope
        onset_strength = np.diff(envelope)
        onset_strength[onset_strength < 0] = 0  # Keep only increases

        # Downsample for efficiency
        hop_length = 512
        onset_env = onset_strength[::hop_length]

        # Calculate autocorrelation of onset envelope
        correlation = signal.correlate(onset_env, onset_env, mode="full")
        correlation = correlation[len(correlation) // 2 :]

        # Calculate tempogram (use reasonable range for tempos: 40-240 BPM)
        min_bpm = 40
        max_bpm = 240

        # Convert BPM to lag
        sr_downsample = self.sample_rate / hop_length
        min_lag = int(60 * sr_downsample / max_bpm)
        max_lag = int(60 * sr_downsample / min_bpm)

        # Ensure we don't exceed array bounds
        if max_lag >= len(correlation):
            max_lag = len(correlation) - 1

        if min_lag >= max_lag:
            return 0.0

        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(correlation[min_lag:max_lag])

        if len(peaks) == 0:
            return 0.0

        # Find strongest peak
        peak_index = peaks[np.argmax(correlation[min_lag:max_lag][peaks])] + min_lag

        # Convert lag to BPM
        tempo = 60 * sr_downsample / peak_index

        return tempo

    def find_beats(self, samples: np.ndarray) -> np.ndarray:
        """
        Find beat locations in an audio signal.

        Args:
            samples: Input audio samples

        Returns:
            Array of beat positions in samples
        """
        if len(samples) < self.sample_rate:
            return np.array([])

        # Get the tempo first
        tempo = self.detect_tempo(samples)
        if tempo <= 0:
            return np.array([])

        # Calculate beat period in samples
        beat_period = int(60 / tempo * self.sample_rate)

        # Calculate onset strength
        amplitude_analyzer = AmplitudeAnalyzer(sample_rate=self.sample_rate)
        envelope = amplitude_analyzer.envelope(samples, attack_ms=10, release_ms=50)
        onset_strength = np.diff(envelope)
        onset_strength[onset_strength < 0] = 0

        # Pad to match original length
        onset_strength = np.append(onset_strength, [0])

        # Find peaks in onset strength
        peaks, _ = signal.find_peaks(onset_strength, distance=beat_period * 0.5)

        # If no peaks found, fall back to evenly spaced beats
        if len(peaks) == 0:
            # Start with first strong onset or default to 0
            strong_onsets = np.where(onset_strength > np.max(onset_strength) * 0.5)[0]
            start = strong_onsets[0] if len(strong_onsets) > 0 else 0

            # Generate evenly spaced beats
            num_beats = int(len(samples) / beat_period)
            return np.array([start + i * beat_period for i in range(num_beats)])

        return peaks


class OnsetDetector:
    """Detects note onsets in audio signals."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the onset detector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate

    def detect_onsets(self, samples: np.ndarray, threshold: float = 0.2, min_distance_ms: float = 50) -> np.ndarray:
        """
        Detect note onsets in an audio signal.

        Args:
            samples: Input audio samples
            threshold: Detection threshold (0-1)
            min_distance_ms: Minimum time between onsets in milliseconds

        Returns:
            Array of onset positions in samples
        """
        if len(samples) < 1024:
            return np.array([])

        # Calculate minimum distance in samples
        min_distance = int(min_distance_ms * self.sample_rate / 1000)

        # Calculate spectral flux
        window_size = 1024
        hop_size = window_size // 4

        num_frames = 1 + (len(samples) - window_size) // hop_size
        flux = np.zeros(num_frames)

        # Previous spectrum magnitude for difference calculation
        prev_mag = np.ones(window_size // 2 + 1)

        for i in range(num_frames):
            # Get current frame
            frame = samples[i * hop_size : i * hop_size + window_size]

            # Apply window
            frame = frame * np.hanning(len(frame))

            # Compute FFT
            spectrum = np.abs(rfft(frame))

            # Calculate spectral flux (increase in energy)
            diff = spectrum - prev_mag
            diff[diff < 0] = 0  # Keep only increases in energy

            # Sum differences
            flux[i] = np.sum(diff)

            # Update previous magnitude
            prev_mag = spectrum

        # Normalise flux
        if np.max(flux) > 0:
            flux = flux / np.max(flux)

        # Find peaks in flux
        peaks, _ = signal.find_peaks(flux, height=threshold, distance=min_distance // hop_size)

        # Convert peak frame indices to sample positions
        onset_samples = peaks * hop_size

        return onset_samples


class HarmonicAnalyzer:
    """Analyzes harmonic content of audio signals."""

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the harmonic analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.fft_analyzer = FFTAnalyzer(sample_rate=sample_rate)

    def harmonic_to_noise_ratio(self, samples: np.ndarray, fundamental_freq: Optional[float] = None) -> float:
        """
        Calculate the Harmonic-to-Noise Ratio (HNR).

        Args:
            samples: Input audio samples
            fundamental_freq: Fundamental frequency, if known

        Returns:
            Harmonic-to-Noise Ratio in dB
        """
        if len(samples) < 1024:
            return 0.0

        # If fundamental frequency not provided, try to detect it
        if fundamental_freq is None or fundamental_freq <= 0:
            pitch_detector = PitchDetector(sample_rate=self.sample_rate)
            fundamental_freq = pitch_detector.autocorrelation(samples)

        if fundamental_freq <= 0:
            return 0.0

        # Get the spectrum
        frequencies, magnitudes = self.fft_analyzer.compute_spectrum(samples)

        # Calculate harmonic power
        harmonic_power = 0
        noise_power = 0

        # Define narrow bands around expected harmonics
        harmonic_width = fundamental_freq * 0.05  # 5% of fundamental

        for i in range(1, 20):  # Check up to 20 harmonics
            harmonic_freq = fundamental_freq * i

            if harmonic_freq >= self.sample_rate / 2:
                break

            # Find closest bin in frequency array
            idx = np.argmin(np.abs(frequencies - harmonic_freq))

            # Calculate range of bins for this harmonic
            low_idx = np.argmin(np.abs(frequencies - (harmonic_freq - harmonic_width)))
            high_idx = np.argmin(np.abs(frequencies - (harmonic_freq + harmonic_width)))

            # Sum power in the harmonic band
            harmonic_band_power = np.sum(magnitudes[low_idx : high_idx + 1] ** 2)
            harmonic_power += harmonic_band_power

        # Total power
        total_power = np.sum(magnitudes**2)

        # Noise power is what's left after harmonics
        noise_power = total_power - harmonic_power

        # Calculate ratio, avoid division by zero
        if noise_power <= 0:
            return 100.0  # Pure tone

        hnr = 10 * np.log10(harmonic_power / noise_power)
        return hnr

    def find_harmonic_peaks(self, samples: np.ndarray, fundamental_freq: Optional[float] = None) -> Dict[int, float]:
        """
        Find the amplitudes of harmonic peaks.

        Args:
            samples: Input audio samples
            fundamental_freq: Fundamental frequency, if known

        Returns:
            Dictionary mapping harmonic numbers to amplitudes
        """
        if len(samples) < 1024:
            return {}

        # Detect fundamental if not provided
        if fundamental_freq is None or fundamental_freq <= 0:
            pitch_detector = PitchDetector(sample_rate=self.sample_rate)
            fundamental_freq = pitch_detector.autocorrelation(samples)

        if fundamental_freq <= 0:
            return {}

        # Get the spectrum
        frequencies, magnitudes = self.fft_analyzer.compute_spectrum(samples)

        # Find harmonic peaks
        harmonic_peaks = {}

        for i in range(1, 20):  # Check up to 20 harmonics
            harmonic_freq = fundamental_freq * i

            if harmonic_freq >= self.sample_rate / 2:
                break

            # Find closest frequency bin
            idx = np.argmin(np.abs(frequencies - harmonic_freq))

            # Look for peak around this frequency
            window_size = 5  # Look at nearby bins
            start_idx = max(0, idx - window_size)
            end_idx = min(len(frequencies), idx + window_size + 1)

            if start_idx >= end_idx:
                continue

            # Find max in the window
            peak_idx = start_idx + np.argmax(magnitudes[start_idx:end_idx])
            peak_amplitude = magnitudes[peak_idx]

            # Store the result
            harmonic_peaks[i] = peak_amplitude

        return harmonic_peaks

    def inharmonicity(self, samples: np.ndarray, fundamental_freq: Optional[float] = None) -> float:
        """
        Calculate the inharmonicity coefficient.

        Args:
            samples: Input audio samples
            fundamental_freq: Fundamental frequency, if known

        Returns:
            Inharmonicity coefficient (0 = perfectly harmonic)
        """
        # Get harmonic peaks
        peaks = self.find_harmonic_peaks(samples, fundamental_freq)

        if not peaks or 1 not in peaks:
            return 0.0

        if fundamental_freq is None or fundamental_freq <= 0:
            if 1 in peaks:
                # Use the first harmonic as fundamental
                fundamental_freq = peaks[1]
            else:
                return 0.0

        # Calculate inharmonicity (deviation from expected harmonic positions)
        deviations = 0
        count = 0

        for harmonic, amplitude in peaks.items():
            if harmonic > 1:  # Skip fundamental
                # Expected frequency
                expected = harmonic * fundamental_freq

                # Find actual frequency
                frequencies, magnitudes = self.fft_analyzer.compute_spectrum(samples)
                idx = np.argmin(np.abs(frequencies - expected))

                # Look for peak around this frequency
                window_size = int(fundamental_freq * 0.1)  # 10% of fundamental
                start_idx = max(0, idx - window_size)
                end_idx = min(len(frequencies), idx + window_size + 1)

                if start_idx >= end_idx:
                    continue

                # Find max in the window
                peak_idx = start_idx + np.argmax(magnitudes[start_idx:end_idx])
                actual = frequencies[peak_idx]

                # Calculate deviation
                deviation = abs((actual - expected) / expected)
                deviations += deviation
                count += 1

        if count == 0:
            return 0.0

        return deviations / count


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt

    # Generate a test signal (mixture of sine waves)
    DURATION = 3.0  # seconds
    SAMPLE_RATE = 44100
    timebase = np.linspace(0, DURATION, int(DURATION * SAMPLE_RATE), endpoint=False)

    # Create a signal with fundamental at 440Hz and harmonics
    signal_440 = 0.5 * np.sin(2 * np.pi * 440 * timebase)  # Fundamental
    signal_440 += 0.3 * np.sin(2 * np.pi * 880 * timebase)  # 2nd harmonic
    signal_440 += 0.15 * np.sin(2 * np.pi * 1320 * timebase)  # 3rd harmonic

    # Add some "beats" by modulating the amplitude
    beat_env = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * timebase)  # 2 Hz beat
    signal_440 *= beat_env

    # Add some noise
    noise = 0.1 * np.random.normal(0, 1, len(timebase))
    test_signal = signal_440 + noise

    # Create analyzers
    fft_analyzer = FFTAnalyzer(sample_rate=SAMPLE_RATE)
    pitch_detector = PitchDetector(sample_rate=SAMPLE_RATE)
    amplitude_analyzer = AmplitudeAnalyzer(sample_rate=SAMPLE_RATE)
    beat_detector = BeatDetector(sample_rate=SAMPLE_RATE)
    harmonic_analyzer = HarmonicAnalyzer(sample_rate=SAMPLE_RATE)

    # Save test audio to file
    write("test_audio.wav", SAMPLE_RATE, np.int16(test_signal * 32767))
    print("Saved test_audio.wav")

    # Analyze the signal
    print("Analyzing test signal...")

    # FFT Analysis
    frequencies, magnitudes = fft_analyzer.compute_spectrum(test_signal)
    peak_freq = fft_analyzer.get_peak_frequency(test_signal)
    print(f"Peak frequency: {peak_freq:.1f} Hz")

    # Pitch Detection
    pitch_auto = pitch_detector.autocorrelation(test_signal)
    pitch_zc = pitch_detector.zero_crossings(test_signal)
    print(f"Detected pitch (autocorrelation): {pitch_auto:.1f} Hz")
    print(f"Detected pitch (zero crossings): {pitch_zc:.1f} Hz")

    # Amplitude Analysis
    rms_val = amplitude_analyzer.rms(test_signal)
    peak_val = amplitude_analyzer.peak(test_signal)
    crest = amplitude_analyzer.crest_factor(test_signal)
    loudness = amplitude_analyzer.loudness_ebur128(test_signal)
    print(f"RMS amplitude: {rms_val:.4f}")
    print(f"Peak amplitude: {peak_val:.4f}")
    print(f"Crest factor: {crest:.2f}")
    print(f"Estimated loudness: {loudness:.2f} LUFS")

    # Beat Analysis
    tempo = beat_detector.detect_tempo(test_signal)
    beats = beat_detector.find_beats(test_signal)
    print(f"Estimated tempo: {tempo:.1f} BPM")
    print(f"Found {len(beats)} beats")

    # Harmonic Analysis
    hnr = harmonic_analyzer.harmonic_to_noise_ratio(test_signal)
    harmonic_peaks = harmonic_analyzer.find_harmonic_peaks(test_signal)
    inharmonicity = harmonic_analyzer.inharmonicity(test_signal)
    print(f"Harmonic-to-Noise Ratio: {hnr:.2f} dB")
    print(f"Inharmonicity coefficient: {inharmonicity:.6f}")
    print("Harmonic structure:")
    for harmonic, amplitude in sorted(harmonic_peaks.items()):
        if amplitude > 0.01:
            print(f"  Harmonic {harmonic}: {amplitude:.4f}")

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot the waveform
    plt.subplot(4, 1, 1)
    plt.plot(timebase[:5000], test_signal[:5000])
    plt.title("Waveform (first 5000 samples)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot spectrum
    plt.subplot(4, 1, 2)
    plt.plot(frequencies[:5000], magnitudes[:5000])
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # Plot spectrogram
    plt.subplot(4, 1, 3)
    times, freqs, spectrogram = fft_analyzer.compute_spectrogram(test_signal)
    plt.pcolormesh(times, freqs, spectrogram, shading="gouraud")
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")

    # Plot envelope
    plt.subplot(4, 1, 4)
    envelope = amplitude_analyzer.envelope(test_signal)
    plt.plot(timebase, test_signal, alpha=0.5)
    plt.plot(timebase, envelope, linewidth=2)
    plt.title("Amplitude Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

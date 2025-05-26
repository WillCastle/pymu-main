"""
Digital Signal Processing (DSP) processors for audio signals.
"""

from typing import Literal, Optional

import numpy as np
from scipy import signal


class EnvelopeProcessor:
    """Applies amplitude envelopes to audio signals"""

    def __init__(
        self,
        envelope_type: Literal["linear", "exponential", "adsr"] = "adsr",
        attack: float = 0.1,
        decay: float = 0.1,
        sustain: float = 0.7,
        release: float = 0.1,
    ):
        self.envelope_type = envelope_type
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply an amplitude envelope to a waveform.

        Args:
            samples: Input waveform samples.
            envelope_type: Type of envelope to apply.
            attack: Attack time in seconds (for ADSR).
            decay: Decay time in seconds (for ADSR).
            sustain: Sustain level as a proportion of peak amplitude (for ADSR).
            release: Release time in seconds (for ADSR).

        Returns:
            Waveform with envelope applied.
        """
        num_samples = len(samples)

        if self.envelope_type == "linear":
            # Simple linear fade in/out
            fade_samples = int(0.1 * num_samples)
            envelope = np.ones(num_samples)
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        elif self.envelope_type == "exponential":
            # Exponential fade in/out
            fade_samples = int(0.1 * num_samples)
            envelope = np.ones(num_samples)
            envelope[:fade_samples] = np.power(np.linspace(0, 1, fade_samples), 2)
            envelope[-fade_samples:] = np.power(np.linspace(1, 0, fade_samples), 2)

        elif self.envelope_type == "adsr":
            # ADSR envelope
            envelope = np.zeros(num_samples)

            a_samples = int(self.attack * num_samples)
            d_samples = int(self.decay * num_samples)
            r_samples = int(self.release * num_samples)
            s_samples = num_samples - a_samples - d_samples - r_samples

            # Attack phase
            if a_samples > 0:
                envelope[:a_samples] = np.linspace(0, 1, a_samples)

            # Decay phase
            if d_samples > 0:
                envelope[a_samples : a_samples + d_samples] = np.linspace(1, self.sustain, d_samples)

            # Sustain phase
            envelope[a_samples + d_samples : a_samples + d_samples + s_samples] = self.sustain

            # Release phase
            if r_samples > 0:
                envelope[-r_samples:] = np.linspace(self.sustain, 0, r_samples)

        else:
            raise ValueError(f"Unsupported envelope type: {self.envelope_type}")

        return samples * envelope


class Mixer:
    """Mixes multiple waveforms together with optional weighting"""

    @staticmethod
    def mix_waveforms(waveforms: list[np.ndarray], weights: Optional[list[float]] = None) -> np.ndarray:
        """
        Mix multiple waveforms together.

        Args:
            waveforms: List of waveforms to mix.
            weights: List of weights for each waveform (default: equal weights).

        Returns:
            Mixed waveform.
        """
        if not waveforms:
            return np.array([])

        # Ensure all waveforms have the same length
        max_length = max(len(w) for w in waveforms)
        padded_waveforms = []
        for w in waveforms:
            if len(w) < max_length:
                padded = np.pad(w, (0, max_length - len(w)), "constant")
                padded_waveforms.append(padded)
            else:
                padded_waveforms.append(w)

        # Apply weights
        if weights is None:
            weights = [1.0 / len(waveforms)] * len(waveforms)
        elif len(weights) != len(waveforms):
            raise ValueError("Number of weights must match number of waveforms")

        # Mix waveforms
        result = np.zeros(max_length)
        for w, weight in zip(padded_waveforms, weights):
            result += w * weight

        # Normalise to avoid clipping
        max_amplitude = np.max(np.abs(result))
        if max_amplitude > 1.0:
            result /= max_amplitude

        return result


class Filter:
    """Applies frequency filtering to audio signals"""

    def __init__(
        self,
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        cutoff_frequency: float = 1000.0,
        bandwidth: Optional[float] = None,
        order: int = 2,
        sample_rate: int = 44100,
    ):
        """
        Initialise the filter.

        Args:
            filter_type: Type of filter to apply.
            cutoff_frequency: Cutoff frequency in Hz.
            bandwidth: Bandwidth in Hz for bandpass/bandstop filters.
            order: Filter order (higher = steeper cutoff).
            sample_rate: Audio sample rate in Hz.
        """
        self.filter_type = filter_type
        self.cutoff_frequency = cutoff_frequency
        self.bandwidth = bandwidth
        self.order = order
        self.sample_rate = sample_rate
        self._b = None
        self._a = None
        self._update_coefficients()

    def _update_coefficients(self):
        """Update filter coefficients based on current parameters"""

        nyquist = self.sample_rate / 2.0
        cutoff_normalised = self.cutoff_frequency / nyquist

        # Ensure cutoff is in valid range
        cutoff_normalised = min(max(0.001, cutoff_normalised), 0.999)

        if self.filter_type == "lowpass":
            self._b, self._a = signal.butter(self.order, cutoff_normalised, btype="low")
        elif self.filter_type == "highpass":
            self._b, self._a = signal.butter(self.order, cutoff_normalised, btype="high")
        elif self.filter_type in ["bandpass", "bandstop"]:
            if not self.bandwidth:
                self.bandwidth = self.cutoff_frequency / 2.0

            low = (self.cutoff_frequency - self.bandwidth / 2) / nyquist
            high = (self.cutoff_frequency + self.bandwidth / 2) / nyquist

            # Ensure values are in valid range
            low = min(max(0.001, low), 0.999)
            high = min(max(low + 0.001, high), 0.999)

            btype = "band" if self.filter_type == "bandpass" else "bandstop"
            self._b, self._a = signal.butter(self.order, [low, high], btype=btype)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply the filter to an audio signal.

        Args:
            samples: Input audio samples.

        Returns:
            Filtered audio samples.
        """

        # Update coefficients if they haven't been calculated yet
        if self._b is None or self._a is None:
            self._update_coefficients()

        # Apply filter
        return signal.lfilter(self._b, self._a, samples)


class Compressor:
    """Dynamic range compressor for audio signals"""

    def __init__(
        self,
        threshold: float = -20.0,  # dB
        ratio: float = 4.0,  # compression ratio
        attack: float = 0.01,  # seconds
        release: float = 0.1,  # seconds
        makeup_gain: float = 0.0,  # dB
        sample_rate: int = 44100,
    ):
        """
        Initialise the compressor.

        Args:
            threshold: Threshold level in dB below which compression begins.
            ratio: Compression ratio (higher = more compression).
            attack: Attack time in seconds.
            release: Release time in seconds.
            makeup_gain: Gain to apply after compression, in dB.
            sample_rate: Audio sample rate in Hz.
        """
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release
        self.makeup_gain = makeup_gain
        self.sample_rate = sample_rate

        # Initialise internal state
        self._envelope = 0.0
        self._attack_coef = np.exp(-1.0 / (sample_rate * attack))
        self._release_coef = np.exp(-1.0 / (sample_rate * release))

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression to an audio signal.

        Args:
            samples: Input audio samples.

        Returns:
            Compressed audio samples.
        """
        # Convert threshold from dB to linear
        threshold_linear = 10.0 ** (self.threshold / 20.0)

        # Convert makeup gain from dB to linear
        makeup_linear = 10.0 ** (self.makeup_gain / 20.0)

        output = np.zeros_like(samples)

        # Process each sample
        for i, input_sample in enumerate(samples):
            # Get current sample's absolute value
            input_level = abs(input_sample)

            # Envelope follower (peak detector)
            if input_level > self._envelope:
                self._envelope = self._attack_coef * self._envelope + (1.0 - self._attack_coef) * input_level
            else:
                self._envelope = self._release_coef * self._envelope + (1.0 - self._release_coef) * input_level

            # Calculate gain reduction
            if self._envelope > threshold_linear:
                # Above threshold: apply compression
                gain_reduction = threshold_linear + (self._envelope - threshold_linear) / self.ratio
                gain = gain_reduction / self._envelope
            else:
                # Below threshold: no compression
                gain = 1.0

            # Apply gain and makeup gain to sample
            output[i] = input_sample * gain * makeup_linear

        return output


class Delay:
    """Creates echo and delay effects"""

    def __init__(
        self,
        delay_time: float = 0.5,  # seconds
        feedback: float = 0.5,  # 0-1 range
        wet_level: float = 0.5,  # 0-1 range
        dry_level: float = 0.5,  # 0-1 range
        sample_rate: int = 44100,
    ):
        """
        Initialise the delay effect.

        Args:
            delay_time: Delay time in seconds.
            feedback: Amount of feedback (0-1), higher values create more repeating echoes.
            wet_level: Level of processed signal in the output (0-1).
            dry_level: Level of unprocessed signal in the output (0-1).
            sample_rate: Audio sample rate in Hz.
        """
        self.delay_time = delay_time
        self.feedback = min(max(0.0, feedback), 0.99)  # Limit feedback to avoid runaway
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.sample_rate = sample_rate

        # Create delay buffer
        self._buffer_size = int(delay_time * sample_rate)
        self._buffer = np.zeros(self._buffer_size)
        self._buffer_index = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply delay effect to an audio signal.

        Args:
            samples: Input audio samples.

        Returns:
            Processed audio samples with delay effect.
        """
        output = np.zeros_like(samples)

        for i, input_sample in enumerate(samples):
            # Read from delay buffer
            delayed_sample = self._buffer[self._buffer_index]

            # Mix dry and wet signal
            output[i] = self.dry_level * input_sample + self.wet_level * delayed_sample

            # Write to delay buffer with feedback
            self._buffer[self._buffer_index] = input_sample + self.feedback * delayed_sample

            # Update buffer index
            self._buffer_index = (self._buffer_index + 1) % self._buffer_size

        return output


class Reverb:
    """Simple reverb effect using multiple delay lines"""

    def __init__(
        self,
        room_size: float = 0.5,  # 0-1 range
        damping: float = 0.5,  # 0-1 range
        wet_level: float = 0.35,  # 0-1 range
        dry_level: float = 0.65,  # 0-1 range
        width: float = 1.0,  # stereo width
        sample_rate: int = 44100,
    ):
        """
        Initialise a simple reverb effect.

        Args:
            room_size: Size of the simulated room (0-1), affects reverb time.
            damping: High frequency damping amount (0-1).
            wet_level: Level of processed signal in the output (0-1).
            dry_level: Level of unprocessed signal in the output (0-1).
            width: Stereo width of the reverb (mono input only).
            sample_rate: Audio sample rate in Hz.
        """
        self.room_size = room_size
        self.damping = damping
        self.wet_level = wet_level
        self.dry_level = dry_level
        self.width = width
        self.sample_rate = sample_rate

        # Create multiple delay lines with different times for a more natural effect
        self._num_delay_lines = 8
        self._delays = []

        # Create delay lines with different times and feedback values
        base_delay = 0.03  # 30 ms
        for i in range(self._num_delay_lines):
            # Vary delay times slightly for each line
            delay_time = base_delay * (0.5 + i * 0.2)
            # Feedback based on room size but varies slightly per delay line
            feedback = self.room_size * 0.5 * (0.9 + 0.1 * (i % 3))

            delay = Delay(
                delay_time=delay_time, feedback=feedback, wet_level=1.0, dry_level=0.0, sample_rate=sample_rate
            )
            self._delays.append(delay)

        # Initialise internal filter for damping
        self._damping_filter = Filter(
            filter_type="lowpass", cutoff_frequency=5000 * (1.0 - self.damping * 0.7), sample_rate=sample_rate
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply reverb effect to an audio signal.

        Args:
            samples: Input audio samples.

        Returns:
            Processed audio samples with reverb effect.
        """
        # Start with silence
        wet_output = np.zeros_like(samples)

        # Process through each delay line
        for delay in self._delays:
            # Each delay line contributes to output
            delayed = delay.process(samples)
            wet_output += delayed / self._num_delay_lines

        # Apply damping filter
        wet_output = self._damping_filter.process(wet_output)

        # Mix dry and wet signals
        return self.dry_level * samples + self.wet_level * wet_output


class Distortion:
    """Applies various distortion effects to audio signals"""

    def __init__(
        self,
        drive: float = 1.0,  # distortion amount (1.0 = none)
        distortion_type: Literal["soft_clip", "hard_clip", "fuzz", "bit_crush"] = "soft_clip",
        output_gain: float = 0.5,
        bit_depth: int = 8,  # for bit_crush type
    ):
        """
        Initialise the distortion effect.

        Args:
            drive: Distortion drive amount (higher = more distortion).
            distortion_type: Type of distortion to apply.
            output_gain: Output level after distortion.
            bit_depth: Bit depth for bit crusher distortion.
        """
        self.drive = max(1.0, drive)  # Ensure drive is 1.0 or higher
        self.distortion_type = distortion_type
        self.output_gain = output_gain
        self.bit_depth = bit_depth

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply distortion effect to an audio signal.

        Args:
            samples: Input audio samples.

        Returns:
            Processed audio samples with distortion effect.
        """
        # Apply input gain (drive)
        driven_samples = samples * self.drive

        if self.distortion_type == "soft_clip":
            # Soft clipping (tanh)
            processed = np.tanh(driven_samples)

        elif self.distortion_type == "hard_clip":
            # Hard clipping
            processed = np.clip(driven_samples, -1.0, 1.0)

        elif self.distortion_type == "fuzz":
            # Fuzz distortion (asymmetrical clipping)
            processed = np.sign(driven_samples) * (1 - np.exp(-abs(driven_samples)))

        elif self.distortion_type == "bit_crush":
            # Bit crusher effect
            max_val = 2**self.bit_depth
            processed = np.round(driven_samples * max_val / 2) / (max_val / 2)

        else:
            raise ValueError(f"Unsupported distortion type: {self.distortion_type}")

        # Apply output gain
        return processed * self.output_gain

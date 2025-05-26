"""
Sound synthesis module for creating audio waveforms.

This module provides functions to generate various audio waveforms such as
sine, square, triangle, and sawtooth waves with configurable parameters.
"""

from typing import Literal, Optional

import numpy as np

from pymu.digital_signal_processing.processors import Mixer

WaveformType = Literal["sine", "square", "triangle", "sawtooth", "noise"]


class Oscillator:
    def __init__(
        self, frequency: float, waveform_type: WaveformType = "sine", amplitude: float = 0.5, sample_rate: int = 44100
    ):
        """
        Initialise the oscillator with frequency, waveform type, and other properties.

        Args:
            frequency: Frequency of the oscillator in Hz.
            waveform_type: Type of waveform to generate.
            amplitude: Amplitude of the waveform (0.0 to 1.0).
            sample_rate: Number of samples per second.
        """
        self.frequency = frequency
        self.waveform_type = waveform_type
        self.amplitude = amplitude
        self.sample_rate = sample_rate
        self.phase = 0.0

    def generate(self, duration: float) -> np.ndarray:
        """Generate samples for the specified duration"""
        return WaveformGenerator.generate_waveform(
            frequency=self.frequency,
            duration=duration,
            amplitude=self.amplitude,
            sample_rate=self.sample_rate,
            waveform=self.waveform_type,
            phase=self.phase,
        )

    def set_frequency(self, frequency: float) -> None:
        """Set the oscillator's frequency"""
        self.frequency = frequency

    def set_amplitude(self, amplitude: float) -> None:
        """Set the oscillator's amplitude"""
        self.amplitude = amplitude

    def set_phase(self, phase: float) -> None:
        """Set the oscillator's phase"""
        self.phase = phase


class WaveformGenerator:
    @staticmethod
    def generate_waveform(
        frequency: float,
        duration: float,
        amplitude: float = 0.5,
        sample_rate: int = 44100,
        waveform: WaveformType = "sine",
        phase: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a waveform with the specified parameters.

        Args:
            frequency: Frequency of the waveform in Hz.
            duration: Duration of the sound in seconds.
            amplitude: Amplitude of the waveform (default: 0.5).
            sample_rate: Number of samples per second (default: 44100).
            waveform: Type of waveform to generate (default: "sine").
            phase: Initial phase of the waveform in radians (default: 0.0).

        Returns:
            Numpy array containing the waveform samples.
        """
        timebase = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        if waveform == "sine":
            samples = amplitude * np.sin(2 * np.pi * frequency * timebase + phase)
        elif waveform == "square":
            raw_wave = np.sin(2 * np.pi * frequency * timebase + phase)
            # Force values to be exactly +1 or -1 with no zeros
            samples = amplitude * np.where(raw_wave >= 0, 1.0, -1.0)
        elif waveform == "triangle":
            samples = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * timebase + phase))
        elif waveform == "sawtooth":
            samples = amplitude * (2 * (timebase * frequency + phase / (2 * np.pi)) % 2 - 1)
        elif waveform == "noise":
            samples = amplitude * (2 * np.random.random(len(timebase)) - 1)
        else:
            raise ValueError(f"Unsupported waveform type: {waveform}")

        return samples


class FrequencyModulatorSynthesiser:
    @staticmethod
    def frequency_modulation(
        carrier_freq: float,
        modulator_freq: float,
        modulation_index: float,
        duration: float,
        amplitude: float = 1.0,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        """
        Generate a frequency modulated waveform.

        Args:
            carrier_freq: Carrier frequency in Hz.
            modulator_freq: Modulator frequency in Hz.
            modulation_index: Modulation index (depth of modulation).
            duration: Duration in seconds.
            amplitude: Amplitude of the waveform.
            sample_rate: Number of samples per second.

        Returns:
            FM waveform as a numpy array.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t + modulation_index * modulator)
        return amplitude * carrier


class HarmonicSeriesSynthesiser:
    """Synthesiser that generates sounds using harmonic series"""

    def __init__(self):
        self.fundamental_freq = 440.0
        self.num_harmonics = 5
        self.amplitudes = [1.0, 0.5, 0.25, 0.125, 0.0625]

    def generate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """Generate a harmonic series waveform"""
        return create_harmonic_series(
            fundamental_freq=self.fundamental_freq,
            num_harmonics=self.num_harmonics,
            amplitudes=self.amplitudes,
            duration=duration,
            sample_rate=sample_rate,
        )


def create_harmonic_series(
    fundamental_freq: float,
    num_harmonics: int,
    amplitudes: Optional[list[float]] = None,
    falloff: Literal["linear", "inverse", "inverse_square", "exponential"] = "inverse",
    falloff_factor: float = 1.0,
    duration: float = 1.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Create a waveform with a harmonic series.

    Args:
        fundamental_freq: Fundamental frequency in Hz.
        num_harmonics: Number of harmonics to include.
        amplitudes: Explicit list of amplitudes for each harmonic (overrides falloff).
        falloff: Type of amplitude falloff for harmonics:
            - "linear": Amplitudes decrease linearly
            - "inverse": Classic 1/n falloff (default)
            - "inverse_square": 1/n² falloff
            - "exponential": Exponential falloff
        falloff_factor: Adjusts the steepness of the falloff.
        duration: Duration in seconds.
        amplitude: Overall amplitude scaling.
        sample_rate: Number of samples per second.

    Returns:
        Waveform with harmonic series.
    """
    # Determine harmonic amplitudes
    if amplitudes is not None:
        # Use provided amplitudes, extending or truncating as needed
        if len(amplitudes) < num_harmonics:
            # Extend with zeros if provided list is too short
            harmonic_weights = amplitudes + [0.0] * (num_harmonics - len(amplitudes))
        else:
            # Truncate if provided list is too long
            harmonic_weights = amplitudes[:num_harmonics]
    else:
        # Calculate amplitudes based on falloff pattern
        harmonic_weights = []
        for i in range(1, num_harmonics + 1):
            if falloff == "linear":
                # Linear falloff: 1 - (i-1)*factor/num_harmonics
                weight = max(0, 1.0 - (i - 1) * falloff_factor / num_harmonics)
            elif falloff == "inverse":
                # Classic 1/n falloff
                weight = 1.0 / (i**falloff_factor)
            elif falloff == "inverse_square":
                # 1/n² falloff
                weight = 1.0 / (i**2 * falloff_factor)
            elif falloff == "exponential":
                # Exponential falloff: e^(-factor*i)
                weight = np.exp(-falloff_factor * (i - 1))
            else:
                raise ValueError(f"Unsupported falloff type: {falloff}")
            harmonic_weights.append(weight)

        # Normalise weights to have max value of 1.0
        max_weight = max(harmonic_weights)
        if max_weight > 0:
            harmonic_weights = [w / max_weight for w in harmonic_weights]

    # Generate harmonics
    waveforms = []
    for i in range(num_harmonics):
        if harmonic_weights[i] > 0:  # Skip harmonics with zero amplitude
            harmonic_freq = fundamental_freq * (i + 1)
            harmonic = WaveformGenerator.generate_waveform(
                frequency=harmonic_freq,
                duration=duration,
                amplitude=harmonic_weights[i],
                sample_rate=sample_rate,
                waveform="sine",
            )
            waveforms.append(harmonic)

    if not waveforms:
        # Return silence if no harmonics were generated
        return np.zeros(int(sample_rate * duration))

    return Mixer.mix_waveforms(waveforms)

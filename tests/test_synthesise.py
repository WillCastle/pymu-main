"""
Unit tests for the synthesise module.
"""

import numpy as np
import pytest

from pymu.core.old import (
    apply_envelope,
    create_harmonic_series,
    frequency_modulation,
    generate_waveform,
    mix_waveforms,
)


class TestGenerateWaveform:
    def test_sine_wave_generation(self):
        """Test that sine wave generation works correctly."""
        freq = 440.0  # A4 note
        duration = 0.1
        sample_rate = 44100
        samples = generate_waveform(freq, duration, waveform="sine", sample_rate=sample_rate)

        # Check shape
        assert len(samples) == int(duration * sample_rate)

        # Check amplitude
        assert np.max(samples) <= 0.5
        assert np.min(samples) >= -0.5

        # Check that it actually looks like a sine wave
        # The zero crossings of a sine wave should be at specific points
        expected_period_samples = sample_rate / freq
        zero_indices = np.where(np.diff(np.signbit(samples)))[0]

        # Check at least one full period is generated
        assert len(zero_indices) >= 2

        # Check period is roughly correct (allow 10% tolerance)
        if len(zero_indices) >= 2:
            measured_period = zero_indices[2] - zero_indices[0]
            assert abs(measured_period - expected_period_samples) / expected_period_samples < 0.1

    def test_square_wave_generation(self):
        """Test square wave generation."""
        samples = generate_waveform(100, 0.1, waveform="square")

        # Square wave should only have values of amplitude and -amplitude
        unique_values = np.unique(np.abs(samples))
        assert len(unique_values) == 1
        assert abs(unique_values[0] - 0.5) < 1e-10

        # Check that it changes value an appropriate number of times
        value_changes = np.sum(np.diff(samples) != 0)
        assert value_changes > 0  # At least some changes

    def test_triangle_wave_generation(self):
        """Test triangle wave generation."""
        samples = generate_waveform(100, 0.1, waveform="triangle")

        # Check shape and amplitude
        assert len(samples) == 4410
        assert np.max(samples) <= 0.5
        assert np.min(samples) >= -0.5

        # Triangle wave should be continuous
        assert np.all(np.abs(np.diff(samples)) < 0.1)

    def test_sawtooth_wave_generation(self):
        """Test sawtooth wave generation."""
        samples = generate_waveform(100, 0.1, waveform="sawtooth")

        # Check amplitude
        assert np.max(samples) <= 0.5
        assert np.min(samples) >= -0.5

        # Sawtooth should have a distinctive pattern of gradual rise and sharp fall
        # Detect the sharp falls
        sharp_falls = np.where(np.diff(samples) < -0.5)[0]
        assert len(sharp_falls) > 0

    def test_noise_generation(self):
        """Test noise generation."""
        samples = generate_waveform(100, 0.1, waveform="noise")

        # Noise should be random and have many unique values
        unique_values = np.unique(samples)
        assert len(unique_values) > 100

        # Check amplitude
        assert np.max(samples) <= 0.5
        assert np.min(samples) >= -0.5

    def test_invalid_waveform_type(self):
        """Test that invalid waveform types raise errors."""
        with pytest.raises(ValueError):
            generate_waveform(100, 0.1, waveform="invalid_type")

    def test_phase_parameter(self):
        """Test that phase parameter shifts the waveform."""
        samples_no_phase = generate_waveform(100, 0.1, waveform="sine", phase=0.0)
        samples_quarter_phase = generate_waveform(100, 0.1, waveform="sine", phase=np.pi / 2)

        # The two should be different
        assert not np.allclose(samples_no_phase, samples_quarter_phase)

        # But they should have the same amplitude
        assert np.isclose(np.max(samples_no_phase), np.max(samples_quarter_phase))


class TestApplyEnvelope:
    @pytest.fixture
    def test_samples(self):
        """Generate test samples to use in envelope tests."""
        return np.ones(10000)

    def test_linear_envelope(self, test_samples):
        """Test linear envelope application."""
        result = apply_envelope(test_samples, envelope_type="linear")

        # Check that the envelope has been applied
        assert len(result) == len(test_samples)

        # Check fade in and fade out
        fade_samples = int(0.1 * len(test_samples))
        assert result[0] < 0.1  # Start near zero
        assert result[fade_samples - 1] > 0.9  # End of fade-in close to 1
        assert np.isclose(result[fade_samples], 1.0)  # Middle should be 1.0
        assert result[-1] < 0.1  # End near zero

    def test_exponential_envelope(self, test_samples):
        """Test exponential envelope application."""
        result = apply_envelope(test_samples, envelope_type="exponential")

        # Exponential should have more gradual initial fade-in than linear
        fade_samples = int(0.1 * len(test_samples))
        samples_10pct = int(0.1 * fade_samples)

        # Check that early samples in exponential fade are less than what linear would be
        assert result[samples_10pct] < samples_10pct / fade_samples

    def test_adsr_envelope(self, test_samples):
        """Test ADSR envelope application."""
        attack = 0.1
        decay = 0.1
        sustain = 0.7
        release = 0.1

        result = apply_envelope(
            test_samples, envelope_type="adsr", attack=attack, decay=decay, sustain=sustain, release=release
        )

        # Check overall length
        assert len(result) == len(test_samples)

        # Calculate expected sample positions
        num_samples = len(test_samples)
        a_samples = int(attack * num_samples)
        d_samples = int(decay * num_samples)
        r_samples = int(release * num_samples)

        # Check key points in the envelope
        assert result[0] < 0.1  # Start near zero
        assert np.isclose(result[a_samples - 1], 1.0)  # End of attack should be at peak
        assert np.isclose(result[a_samples + d_samples - 1], sustain)  # End of decay should be at sustain level
        assert np.isclose(result[a_samples + d_samples], sustain)  # Sustain portion should be at sustain level
        assert result[-1] < 0.1  # End near zero

    def test_invalid_envelope_type(self, test_samples):
        """Test that invalid envelope types raise errors."""
        with pytest.raises(ValueError):
            apply_envelope(test_samples, envelope_type="invalid_type")


class TestMixWaveforms:
    def test_empty_waveforms_list(self):
        """Test mixing an empty list of waveforms."""
        result = mix_waveforms([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_single_waveform(self):
        """Test mixing a single waveform."""
        waveform = np.array([0.1, 0.2, 0.3, 0.4])
        result = mix_waveforms([waveform])
        assert np.array_equal(result, waveform)

    def test_multiple_equal_length_waveforms(self):
        """Test mixing multiple waveforms of equal length."""
        waveform1 = np.array([0.1, 0.2, 0.3, 0.4])
        waveform2 = np.array([0.4, 0.3, 0.2, 0.1])
        expected = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights by default

        result = mix_waveforms([waveform1, waveform2])
        assert np.allclose(result, expected)

    def test_different_length_waveforms(self):
        """Test mixing waveforms of different lengths."""
        waveform1 = np.array([0.1, 0.2, 0.3, 0.4])
        waveform2 = np.array([0.5, 0.5])

        result = mix_waveforms([waveform1, waveform2])
        assert len(result) == 4  # Should match the longest waveform

        # The second waveform should be padded with zeros
        expected = np.array([0.3, 0.35, 0.15, 0.2])  # (waveform1 + padded_waveform2) / 2
        assert np.allclose(result, expected)

    def test_custom_weights(self):
        """Test mixing waveforms with custom weights."""
        waveform1 = np.array([0.2, 0.4, 0.6, 0.8])
        waveform2 = np.array([0.8, 0.6, 0.4, 0.2])
        weights = [0.8, 0.2]  # 80% of waveform1, 20% of waveform2

        result = mix_waveforms([waveform1, waveform2], weights=weights)
        expected = np.array([0.32, 0.44, 0.56, 0.68])
        assert np.allclose(result, expected)

    def test_normalization(self):
        """Test that mixing normalises to avoid clipping."""
        waveform1 = np.array([0.6, 0.7, 0.8, 0.9])
        waveform2 = np.array([0.6, 0.7, 0.8, 0.9])

        # Without normalization, this would clip above 1.0
        result = mix_waveforms([waveform1, waveform2])
        assert np.max(result) <= 1.0

    def test_mismatched_weights_length(self):
        """Test that mismatched weights raise an error."""
        waveforms = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        weights = [0.5, 0.3, 0.2]  # One too many weights

        with pytest.raises(ValueError):
            mix_waveforms(waveforms, weights=weights)


class TestFrequencyModulation:
    def test_basic_fm_synthesis(self):
        """Test basic FM synthesis."""
        carrier_freq = 440.0
        modulator_freq = 110.0
        modulation_index = 5.0
        duration = 0.1
        sample_rate = 44100

        samples = frequency_modulation(
            carrier_freq, modulator_freq, modulation_index, duration, sample_rate=sample_rate
        )

        # Check basic properties
        assert len(samples) == int(duration * sample_rate)
        assert np.max(samples) <= 1.0
        assert np.min(samples) >= -1.0

        # FM should have a more complex spectrum than a simple sine wave
        # We can verify this by checking that it has more zero crossings
        sine_wave = generate_waveform(carrier_freq, duration, sample_rate=sample_rate)
        fm_zero_crossings = np.sum(np.diff(np.signbit(samples)))
        sine_zero_crossings = np.sum(np.diff(np.signbit(sine_wave)))

        assert fm_zero_crossings != sine_zero_crossings


class TestCreateHarmonicSeries:
    def test_basic_harmonic_series(self):
        """Test basic harmonic series generation."""
        fundamental = 100.0
        num_harmonics = 3
        duration = 0.1

        samples = create_harmonic_series(fundamental, num_harmonics, duration=duration)

        # Check basic properties
        assert len(samples) == int(duration * 44100)
        assert np.max(samples) <= 1.0
        assert np.min(samples) >= -1.0

    def test_custom_harmonic_weights(self):
        """Test harmonic series with custom weights."""
        fundamental = 100.0
        num_harmonics = 3
        harmonic_weights = [0.5, 0.3, 0.2]

        samples = create_harmonic_series(fundamental, num_harmonics, amplitudes=harmonic_weights)

        # The resulting wave should have all harmonics present
        # This is hard to test directly without doing FFT analysis
        assert np.max(samples) <= 1.0
        assert np.min(samples) >= -1.0

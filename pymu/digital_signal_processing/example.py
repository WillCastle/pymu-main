"""
Example script demonstrating various digital signal processing techniques.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

from pymu.digital_signal_processing.analysis import AmplitudeAnalyzer, FFTAnalyzer, HarmonicAnalyzer
from pymu.digital_signal_processing.processors import Delay, Distortion, EnvelopeProcessor, Filter
from pymu.digital_signal_processing.synthesisers import HarmonicSeriesSynthesiser, Oscillator


def analyze_and_plot(signal, title, sample_rate, output_filename=None):
    """Analyze and plot a signal in time and frequency domains"""
    # Create analyzers
    fft_analyzer = FFTAnalyzer(sample_rate=sample_rate)
    amp_analyzer = AmplitudeAnalyzer(sample_rate=sample_rate)
    harmonic_analyzer = HarmonicAnalyzer(sample_rate=sample_rate)

    # Time domain
    timebase = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)

    # Get frequency domain data
    frequencies, magnitudes = fft_analyzer.compute_spectrum(signal)

    # Get signal metrics
    rms = amp_analyzer.rms(signal)
    peak = amp_analyzer.peak(signal)
    crest = amp_analyzer.crest_factor(signal)

    # Calculate spectrogram data
    times, freqs, spectrogram = fft_analyzer.compute_spectrogram(signal)

    # Create plots
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"{title} Analysis", fontsize=16)

    # Plot waveform
    ax1 = plt.subplot(3, 1, 1)
    short_len = min(3000, len(signal))  # Show first 3000 samples or fewer
    ax1.plot(timebase[:short_len], signal[:short_len])
    ax1.set_title(f"Waveform (RMS: {rms:.3f}, Peak: {peak:.3f}, Crest Factor: {crest:.2f})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Plot spectrum
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(frequencies[:20000], magnitudes[:20000])  # Only show up to 20kHz
    ax2.set_title("Frequency Spectrum")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")

    # Find highest peaks in spectrum
    if len(frequencies) > 0:
        num_peaks = 5  # Show top 5 peaks
        peak_indices = np.argsort(magnitudes)[-num_peaks:]
        peak_freqs = frequencies[peak_indices]
        peak_mags = magnitudes[peak_indices]

        # Add peak annotations
        for freq, mag in zip(peak_freqs, peak_mags):
            if mag > 0.005:  # Only annotate significant peaks
                ax2.annotate(
                    f"{freq:.1f}Hz", xy=(freq, mag), xytext=(0, 5), textcoords="offset points", fontsize=8, ha="center"
                )

    ax2.set_xlim(0, min(20000, max(frequencies)))  # Limit x-axis to 20kHz or max frequency
    ax2.grid(True)

    # Plot spectrogram
    ax3 = plt.subplot(3, 1, 3)
    spec_plot = ax3.pcolormesh(times, freqs, spectrogram, shading="gouraud")
    ax3.set_title("Spectrogram")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    fig.colorbar(spec_plot, ax=ax3, label="Magnitude (dB)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save the plot if filename provided
    if output_filename:
        plot_path = f"{os.path.splitext(output_filename)[0]}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved analysis plot to {plot_path}")

    plt.show()

    # Return analysis data
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        "peak_frequency": fft_analyzer.get_peak_frequency(signal),
        "spectral_centroid": fft_analyzer.get_spectral_centroid(signal),
    }


def save_audio(signal, filename, sample_rate):
    """Save audio to a WAV file with proper scaling"""
    # Scale to 16-bit int range
    scaled = np.int16(np.clip(signal, -1.0, 1.0) * 32767)
    write(filename, sample_rate, scaled)
    print(f"Saved audio to {filename}")


def demonstrate_oscillator(sample_rate):
    """Demonstrate different oscillator waveforms"""
    print("\n=== OSCILLATOR DEMONSTRATION ===")
    duration = 2.0

    waveforms = ["sine", "square", "triangle", "sawtooth", "noise"]

    # Create a figure for comparing all waveforms
    plt.figure(figsize=(15, 10))

    for i, waveform in enumerate(waveforms):
        # Create oscillator
        osc = Oscillator(frequency=440, waveform_type=waveform, amplitude=0.8, sample_rate=sample_rate)
        signal = osc.generate(duration=duration)

        # Save audio
        output_file = f"oscillator_{waveform}.wav"
        save_audio(signal, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {waveform.upper()} OSCILLATOR ANALYSIS ---")
        analysis = analyze_and_plot(signal, f"{waveform.capitalize()} Oscillator (440 Hz)", sample_rate, output_file)

        # Print analysis results
        print(f"RMS amplitude: {analysis['rms']:.4f}")
        print(f"Peak amplitude: {analysis['peak']:.4f}")
        print(f"Crest factor: {analysis['crest_factor']:.2f}")
        print(f"Peak frequency: {analysis['peak_frequency']:.1f} Hz")
        print(f"Spectral centroid: {analysis['spectral_centroid']:.1f} Hz")

    return


def demonstrate_harmonic_synthesis(sample_rate):
    """Demonstrate harmonic series synthesis"""
    print("\n=== HARMONIC SYNTHESIS DEMONSTRATION ===")
    duration = 2.0

    # Create different harmonic configurations
    configs = [
        {"title": "Basic Harmonic Series", "num_harmonics": 5, "amplitudes": [1.0, 0.5, 0.33, 0.25, 0.2]},
        {"title": "Even Harmonics Only", "num_harmonics": 6, "amplitudes": [1.0, 0.0, 0.33, 0.0, 0.2, 0.0]},
        {"title": "Odd Harmonics Only", "num_harmonics": 6, "amplitudes": [1.0, 0.5, 0.0, 0.25, 0.0, 0.16]},
        {
            "title": "Rich Harmonics",
            "num_harmonics": 10,
            "amplitudes": [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1],
        },
    ]

    for config in configs:
        # Create harmonic synthesiser
        synth = HarmonicSeriesSynthesiser()
        synth.fundamental_freq = 220
        synth.num_harmonics = config["num_harmonics"]
        synth.amplitudes = config["amplitudes"]

        # Generate audio
        signal = synth.generate(duration=duration)

        # Use safe filenames
        filename = config["title"].lower().replace(" ", "_")
        output_file = f"harmonic_{filename}.wav"
        save_audio(signal, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {config['title'].upper()} ANALYSIS ---")
        analysis = analyze_and_plot(signal, f"{config['title']} (220 Hz fundamental)", sample_rate, output_file)

        # Get harmonic analysis
        harmonic_analyzer = HarmonicAnalyzer(sample_rate=sample_rate)
        harmonic_peaks = harmonic_analyzer.find_harmonic_peaks(signal, fundamental_freq=220)
        hnr = harmonic_analyzer.harmonic_to_noise_ratio(signal, fundamental_freq=220)

        # Print analysis results
        print(f"Harmonic-to-Noise Ratio: {hnr:.2f} dB")
        print("Detected harmonic structure:")
        for harmonic, amplitude in sorted(harmonic_peaks.items()):
            if amplitude > 0.01:  # Only show significant harmonics
                print(f"  Harmonic {harmonic} ({220 * harmonic:.1f} Hz): {amplitude:.4f}")

    return


def demonstrate_envelope(sample_rate):
    """Demonstrate different envelope types"""
    print("\n=== ENVELOPE PROCESSOR DEMONSTRATION ===")
    duration = 2.0

    # Create a simple oscillator
    osc = Oscillator(frequency=440, waveform_type="sine", amplitude=0.8, sample_rate=sample_rate)
    base_signal = osc.generate(duration=duration)

    # Define different envelope configurations
    configs = [
        {"type": "linear", "attack": 0.1, "release": 0.5, "title": "Linear Envelope"},
        {"type": "exponential", "attack": 0.1, "release": 0.5, "title": "Exponential Envelope"},
        {"type": "adsr", "attack": 0.1, "decay": 0.2, "sustain": 0.6, "release": 0.5, "title": "ADSR Envelope"},
    ]

    for config in configs:
        # Create envelope processor
        if config["type"] == "adsr":
            env = EnvelopeProcessor(
                envelope_type=config["type"],
                attack=config["attack"],
                decay=config["decay"],
                sustain=config["sustain"],
                release=config["release"],
            )
        else:
            env = EnvelopeProcessor(envelope_type=config["type"], attack=config["attack"], release=config["release"])

        # Process signal
        processed = env.process(base_signal)

        # Generate output file
        filename = config["title"].lower().replace(" ", "_")
        output_file = f"envelope_{filename}.wav"
        save_audio(processed, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {config['title'].upper()} ---")
        analysis = analyze_and_plot(processed, config["title"], sample_rate, output_file)

        # Print analysis results
        print(f"RMS amplitude: {analysis['rms']:.4f}")
        print(f"Peak amplitude: {analysis['peak']:.4f}")
        print(f"Crest factor: {analysis['crest_factor']:.2f}")

    return


def demonstrate_filters(sample_rate):
    """Demonstrate different filter types"""
    print("\n=== FILTER DEMONSTRATION ===")
    duration = 2.0

    # Create a rich signal with harmonics to filter
    harmonic_synth = HarmonicSeriesSynthesiser()
    harmonic_synth.fundamental_freq = 220
    harmonic_synth.num_harmonics = 10
    base_signal = harmonic_synth.generate(duration=duration)

    # Define filter configurations
    filter_configs = [
        {"type": "lowpass", "cutoff": 500, "title": "Lowpass Filter (500 Hz)"},
        {"type": "highpass", "cutoff": 1000, "title": "Highpass Filter (1000 Hz)"},
        {"type": "bandpass", "cutoff": 880, "bandwidth": 200, "title": "Bandpass Filter (880 Hz)"},
        {"type": "notch", "cutoff": 880, "bandwidth": 200, "title": "Notch Filter (880 Hz)"},
    ]

    for config in filter_configs:
        # Create filter
        if config["type"] in ["bandpass", "notch"]:
            filt = Filter(
                filter_type=config["type"],
                cutoff_frequency=config["cutoff"],
                bandwidth=config["bandwidth"],
                sample_rate=sample_rate,
            )
        else:
            filt = Filter(filter_type=config["type"], cutoff_frequency=config["cutoff"], sample_rate=sample_rate)

        # Process signal
        processed = filt.process(base_signal)

        # Generate output file
        filename = config["title"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(" ", "_")
        output_file = f"filter_{filename}.wav"
        save_audio(processed, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {config['title'].upper()} ---")
        analysis = analyze_and_plot(processed, config["title"], sample_rate, output_file)

        # Print analysis results
        print(f"Spectral centroid: {analysis['spectral_centroid']:.1f} Hz")
        print(f"Peak frequency: {analysis['peak_frequency']:.1f} Hz")

    return


def demonstrate_delay(sample_rate):
    """Demonstrate delay effect with different settings"""
    print("\n=== DELAY DEMONSTRATION ===")
    duration = 3.0

    # Create a simple input signal (short notes)
    osc = Oscillator(frequency=440, waveform_type="sine", amplitude=0.8, sample_rate=sample_rate)
    env = EnvelopeProcessor(envelope_type="adsr", attack=0.01, decay=0.1, sustain=0.7, release=0.2)

    # Create a signal with two short notes
    note1 = osc.generate(duration=0.5)
    note1 = env.process(note1)

    # Silence between notes
    silence = np.zeros(int(sample_rate * 0.5))

    # Second note at different pitch
    osc.frequency = 550
    note2 = osc.generate(duration=0.5)
    note2 = env.process(note2)

    # Combine into input signal with trailing silence
    input_signal = np.concatenate([note1, silence, note2, np.zeros(int(sample_rate * 1.5))])

    # Define delay configurations
    delay_configs = [
        {"delay_time": 0.25, "feedback": 0.3, "title": "Short Delay (250ms, 30%)"},
        {"delay_time": 0.5, "feedback": 0.5, "title": "Medium Delay (500ms, 50%)"},
        {"delay_time": 0.5, "feedback": 0.7, "title": "Medium Delay High Feedback (500ms, 70%)"},
    ]

    # Save the input signal
    input_file = "delay_input_signal.wav"
    save_audio(input_signal, input_file, sample_rate)
    analyze_and_plot(input_signal, "Input Signal for Delay", sample_rate, input_file)

    for config in delay_configs:
        # Create delay effect
        delay_effect = Delay(delay_time=config["delay_time"], feedback=config["feedback"], sample_rate=sample_rate)

        # Process signal
        processed = delay_effect.process(input_signal)

        # Generate output file
        filename = config["title"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        output_file = f"delay_{filename}.wav"
        save_audio(processed, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {config['title'].upper()} ---")
        analysis = analyze_and_plot(processed, config["title"], sample_rate, output_file)

    return


def demonstrate_distortion(sample_rate):
    """Demonstrate different distortion types"""
    print("\n=== DISTORTION DEMONSTRATION ===")
    duration = 2.0

    # Create a clean sine wave
    osc = Oscillator(frequency=220, waveform_type="sine", amplitude=0.5, sample_rate=sample_rate)
    input_signal = osc.generate(duration=duration)

    # Save the input signal
    input_file = "distortion_input_signal.wav"
    save_audio(input_signal, input_file, sample_rate)
    analyze_and_plot(input_signal, "Clean Sine Wave Input (220 Hz)", sample_rate, input_file)

    # Define distortion configurations
    dist_configs = [
        {"type": "soft_clip", "drive": 2.0, "title": "Soft Clipping (Drive: 2.0)"},
        {"type": "hard_clip", "drive": 2.0, "title": "Hard Clipping (Drive: 2.0)"},
        {"type": "soft_clip", "drive": 5.0, "title": "Heavy Soft Clipping (Drive: 5.0)"},
        {"type": "bit_crush", "bits": 4, "title": "Bit Crusher (4-bit)"},
    ]

    for config in dist_configs:
        # Create distortion effect
        if config["type"] == "bit_crush":
            dist = Distortion(distortion_type=config["type"], bit_depth=config["bits"])
        else:
            dist = Distortion(distortion_type=config["type"], drive=config["drive"])

        # Process signal
        processed = dist.process(input_signal)

        # Generate output file
        filename = config["title"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
        output_file = f"distortion_{filename}.wav"
        save_audio(processed, output_file, sample_rate)

        # Analyze and plot
        print(f"\n--- {config['title'].upper()} ---")
        analysis = analyze_and_plot(processed, config["title"], sample_rate, output_file)

        # Print relevant analysis
        print(f"RMS amplitude: {analysis['rms']:.4f}")
        print(f"Peak amplitude: {analysis['peak']:.4f}")
        print(f"Crest factor: {analysis['crest_factor']:.2f}")

        # Get harmonic analysis
        harmonic_analyzer = HarmonicAnalyzer(sample_rate=sample_rate)
        harmonic_peaks = harmonic_analyzer.find_harmonic_peaks(processed, fundamental_freq=220)

        # Print harmonic content
        print("Generated harmonic content:")
        for harmonic, amplitude in sorted(harmonic_peaks.items()):
            if amplitude > 0.01:  # Only show significant harmonics
                print(f"  Harmonic {harmonic} ({220 * harmonic:.1f} Hz): {amplitude:.4f}")

    return


def demonstrate_effect_chain(sample_rate):
    """Demonstrate a complete effects chain"""
    print("\n=== EFFECTS CHAIN DEMONSTRATION ===")
    duration = 4.0

    # Create a signal to process
    synth = HarmonicSeriesSynthesiser()
    synth.fundamental_freq = 220
    synth.num_harmonics = 5
    base_signal = synth.generate(duration=duration)

    # Create effect processors
    envelope = EnvelopeProcessor(envelope_type="adsr", attack=0.1, decay=0.2, sustain=0.6, release=0.8)
    filter_effect = Filter(filter_type="lowpass", cutoff_frequency=2000, sample_rate=sample_rate)
    delay = Delay(delay_time=0.3, feedback=0.4, sample_rate=sample_rate)
    distortion = Distortion(distortion_type="soft_clip", drive=2.0)

    # Save the input signal
    input_file = "chain_input_signal.wav"
    save_audio(base_signal, input_file, sample_rate)

    # Process through the chain, saving at each stage
    stages = [
        {"name": "Original Signal", "signal": base_signal},
        {"name": "After Envelope", "signal": envelope.process(base_signal)},
        {"name": "After Filter", "signal": filter_effect.process(envelope.process(base_signal))},
        {"name": "After Delay", "signal": delay.process(filter_effect.process(envelope.process(base_signal)))},
        {
            "name": "After Distortion",
            "signal": distortion.process(delay.process(filter_effect.process(envelope.process(base_signal)))),
        },
    ]

    # Analyze each stage
    for i, stage in enumerate(stages):
        output_file = f"chain_stage_{i}_{stage['name'].lower().replace(' ', '_')}.wav"
        save_audio(stage["signal"], output_file, sample_rate)
        analyze_and_plot(stage["signal"], stage["name"], sample_rate, output_file)

    # Create a comparison plot showing all stages
    plt.figure(figsize=(14, 10))
    for i, stage in enumerate(stages):
        plt.subplot(len(stages), 1, i + 1)
        plt.plot(stage["signal"][:2000])
        plt.title(stage["name"])
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("effect_chain_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # Set up parameters
    SAMPLE_RATE = 44100

    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")

    # Change to output directory
    os.chdir("output")

    # Run demonstrations
    demonstrate_oscillator(SAMPLE_RATE)
    demonstrate_harmonic_synthesis(SAMPLE_RATE)
    demonstrate_envelope(SAMPLE_RATE)
    demonstrate_filters(SAMPLE_RATE)
    demonstrate_delay(SAMPLE_RATE)
    demonstrate_distortion(SAMPLE_RATE)
    demonstrate_effect_chain(SAMPLE_RATE)

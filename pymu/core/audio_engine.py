"""
Audio Engine module for PyMu DAW.

This module handles all real-time audio processing, device management,
and provides the core infrastructure for routing audio between tracks,
plugins, and audio hardware.
"""

import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd


class AudioBuffer:
    """Thread-safe audio buffer for passing audio data between components."""

    def __init__(self, channels: int = 2, max_size: int = 10):
        """
        Initialize an audio buffer.

        Args:
            channels: Number of audio channels
            max_size: Maximum size of the buffer queue
        """
        self.channels = channels
        self._buffer = queue.Queue(maxsize=max_size)
        self._lock = threading.RLock()

    def write(self, data: np.ndarray) -> bool:
        """
        Write audio data to the buffer.

        Args:
            data: Audio data as numpy array

        Returns:
            True if write succeeded, False if buffer full
        """
        try:
            self._buffer.put_nowait(data.copy())
            return True
        except queue.Full:
            return False

    def read(self, block: bool = True, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read audio data from the buffer.

        Args:
            block: Whether to block until data is available
            timeout: Timeout in seconds

        Returns:
            Audio data as numpy array, or None if no data available
        """
        try:
            return self._buffer.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self._lock:
            while not self._buffer.empty():
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    break

    @property
    def empty(self) -> bool:
        """Check if the buffer is empty."""
        return self._buffer.empty()

    @property
    def full(self) -> bool:
        """Check if the buffer is full."""
        return self._buffer.full()

    @property
    def size(self) -> int:
        """Get the current size of the buffer."""
        return self._buffer.qsize()


class AudioProcessor:
    """
    Base class for real-time audio processors.

    Subclass this to create processors like effects, instruments, or analyzers.
    """

    def __init__(self, channels: int = 2, sample_rate: int = 44100):
        """
        Initialise the audio processor.

        Args:
            channels: Number of audio channels
            sample_rate: Sample rate in Hz
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_active = False
        self.input_buffers: List[AudioBuffer] = []
        self.output_buffers: List[AudioBuffer] = []
        self.bypass = False
        self._processing_thread = None
        self._should_stop = threading.Event()

    def process_block(self, input_data: np.ndarray) -> np.ndarray:
        """
        Process a block of audio data.

        Args:
            input_data: Input audio data as numpy array

        Returns:
            Processed audio data as numpy array
        """
        if self.bypass:
            return input_data

        # Override this method in subclasses to implement actual processing
        return input_data

    def connect_input(self, buffer: AudioBuffer) -> None:
        """
        Connect an input buffer to this processor.

        Args:
            buffer: Input audio buffer
        """
        if buffer not in self.input_buffers:
            self.input_buffers.append(buffer)

    def connect_output(self, buffer: AudioBuffer) -> None:
        """
        Connect an output buffer to this processor.

        Args:
            buffer: Output audio buffer
        """
        if buffer not in self.output_buffers:
            self.output_buffers.append(buffer)

    def start(self) -> None:
        """Start the processor."""
        if not self.is_active:
            self._should_stop.clear()
            self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._processing_thread.start()
            self.is_active = True

    def stop(self) -> None:
        """Stop the processor."""
        if self.is_active:
            self._should_stop.set()
            if self._processing_thread:
                self._processing_thread.join(timeout=1.0)
            self.is_active = False

    def _processing_loop(self) -> None:
        """Main processing loop that runs in a separate thread."""
        while not self._should_stop.is_set():
            # Get input data from all input buffers
            input_blocks = []
            for buffer in self.input_buffers:
                data = buffer.read(block=False)
                if data is not None:
                    input_blocks.append(data)

            if not input_blocks:
                # No data to process, sleep briefly
                time.sleep(0.001)
                continue

            # Combine input blocks if there are multiple
            if len(input_blocks) > 1:
                input_data = np.mean(input_blocks, axis=0)
            else:
                input_data = input_blocks[0]

            # Process the data
            output_data = self.process_block(input_data)

            # Send to all output buffers
            for buffer in self.output_buffers:
                buffer.write(output_data)


class AudioIO:
    """
    Handles audio input and output using sounddevice.

    This class manages the interface between the application and the audio hardware.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 512,
        input_channels: int = 2,
        output_channels: int = 2,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ):
        """
        Initialise the audio I/O handler.

        Args:
            sample_rate: Sample rate in Hz
            block_size: Audio block size in samples
            input_channels: Number of input channels
            output_channels: Number of output channels
            input_device: Input device ID (None for default)
            output_device: Output device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_device = input_device
        self.output_device = output_device

        # Stream and buffer setup
        self._stream = None
        self.input_buffer = AudioBuffer(channels=input_channels, max_size=20)
        self.output_buffer = AudioBuffer(channels=output_channels, max_size=20)

        # State tracking
        self.is_running = False
        self._underflow_count = 0
        self._overflow_count = 0

        # Temporary storage for callback samples
        self._zeros = np.zeros((block_size, output_channels), dtype=np.float32)

    def list_devices(self) -> List[Dict]:
        """
        Get a list of available audio devices.

        Returns:
            List of device information dictionaries
        """
        return [device._asdict() for device in sd.query_devices()]

    def get_default_devices(self) -> Tuple[int, int]:
        """
        Get the default input and output devices.

        Returns:
            Tuple of (input_device_id, output_device_id)
        """
        return (
            sd.default.device[0] if sd.default.device is not None else -1,
            sd.default.device[1] if sd.default.device is not None else -1,
        )

    def start(self) -> bool:
        """
        Start the audio stream.

        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            return True

        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=(self.input_channels, self.output_channels),
                dtype=np.float32,
                callback=self._audio_callback,
                device=(self.input_device, self.output_device),
            )
            self._stream.start()
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False

    def stop(self) -> None:
        """Stop the audio stream."""
        if self.is_running and self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.is_running = False
            # Clear buffers
            self.input_buffer.clear()
            self.output_buffer.clear()

    def _audio_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: Dict, status: sd.CallbackFlags
    ) -> None:
        """
        Callback function for the sounddevice audio stream.

        Args:
            indata: Input audio data from hardware
            outdata: Output buffer to fill with audio data
            frames: Number of frames to process
            time_info: Timing information
            status: Status flags
        """
        if status.input_overflow:
            self._overflow_count += 1
            print(f"Input overflow ({self._overflow_count})")

        if status.output_underflow:
            self._underflow_count += 1
            print(f"Output underflow ({self._underflow_count})")

        # Store input data in buffer
        if not self.input_buffer.full:
            self.input_buffer.write(indata.copy())

        # Get output data from buffer
        output_data = self.output_buffer.read(block=False)

        if output_data is not None:
            # Ensure correct shape
            if output_data.shape[0] < frames:
                # Pad with zeros if too short
                padding = np.zeros((frames - output_data.shape[0], self.output_channels))
                output_data = np.vstack((output_data, padding))
            elif output_data.shape[0] > frames:
                # Truncate if too long
                output_data = output_data[:frames]

            outdata[:] = output_data
        else:
            # No output data available, send zeros
            outdata[:] = self._zeros


class StreamManager:
    """
    Manages audio streams between processors.

    This class handles the routing of audio between processors and audio I/O.
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialise the stream manager.

        Args:
            sample_rate: The sample rate for all audio processing
        """
        self.sample_rate = sample_rate
        self.processors: Dict[str, AudioProcessor] = {}
        self.connections: Dict[str, List[str]] = {}
        self.audio_io: Optional[AudioIO] = None

    def add_processor(self, name: str, processor: AudioProcessor) -> None:
        """
        Add a processor to the manager.

        Args:
            name: Name of the processor
            processor: The processor instance
        """
        if name not in self.processors:
            processor.sample_rate = self.sample_rate
            self.processors[name] = processor
            self.connections[name] = []

    def remove_processor(self, name: str) -> None:
        """
        Remove a processor from the manager.

        Args:
            name: Name of the processor to remove
        """
        if name in self.processors:
            # Stop the processor if it's active
            if self.processors[name].is_active:
                self.processors[name].stop()

            # Remove all connections to/from this processor
            for source, targets in list(self.connections.items()):
                if name in targets:
                    self.disconnect(source, name)

            # Remove the processor itself
            del self.processors[name]
            del self.connections[name]

    def connect(self, source: str, target: str) -> bool:
        """
        Connect two processors.

        Args:
            source: Name of the source processor
            target: Name of the target processor

        Returns:
            True if connection succeeded, False otherwise
        """
        if source not in self.processors or target not in self.processors:
            return False

        # Create a buffer for this connection
        buffer = AudioBuffer(channels=self.processors[source].channels)

        # Connect the buffer to both processors
        self.processors[source].connect_output(buffer)
        self.processors[target].connect_input(buffer)

        # Record the connection
        if target not in self.connections[source]:
            self.connections[source].append(target)

        return True

    def disconnect(self, source: str, target: str) -> bool:
        """
        Disconnect two processors.

        Args:
            source: Name of the source processor
            target: Name of the target processor

        Returns:
            True if disconnection succeeded, False otherwise
        """
        if source not in self.processors or target not in self.processors:
            return False

        # Find shared buffers and remove them
        source_processor = self.processors[source]
        target_processor = self.processors[target]

        for out_buf in list(source_processor.output_buffers):
            if out_buf in target_processor.input_buffers:
                source_processor.output_buffers.remove(out_buf)
                target_processor.input_buffers.remove(out_buf)

        # Update connection records
        if target in self.connections[source]:
            self.connections[source].remove(target)

        return True

    def connect_to_audio_output(self, source: str) -> bool:
        """
        Connect a processor to the audio output.

        Args:
            source: Name of the source processor

        Returns:
            True if connection succeeded, False otherwise
        """
        if source not in self.processors or self.audio_io is None:
            return False

        self.processors[source].connect_output(self.audio_io.output_buffer)
        return True

    def connect_from_audio_input(self, target: str) -> bool:
        """
        Connect the audio input to a processor.

        Args:
            target: Name of the target processor

        Returns:
            True if connection succeeded, False otherwise
        """
        if target not in self.processors or self.audio_io is None:
            return False

        self.processors[target].connect_input(self.audio_io.input_buffer)
        return True

    def start_all(self) -> None:
        """Start all processors and audio I/O."""
        # First start the audio I/O if available
        if self.audio_io:
            self.audio_io.start()

        # Then start all processors
        for name, processor in self.processors.items():
            processor.start()

    def stop_all(self) -> None:
        """Stop all processors and audio I/O."""
        # First stop all processors
        for name, processor in self.processors.items():
            processor.stop()

        # Then stop the audio I/O if available
        if self.audio_io:
            self.audio_io.stop()


class AudioEngine:
    """
    Main audio engine class that coordinates all audio processing.

    This class serves as the primary interface between the application
    and the audio processing infrastructure.
    """

    def __init__(
        self, sample_rate: int = 44100, block_size: int = 512, input_channels: int = 2, output_channels: int = 2
    ):
        """
        Initialize the audio engine.

        Args:
            sample_rate: Sample rate in Hz
            block_size: Audio block size in samples
            input_channels: Number of input channels
            output_channels: Number of output channels
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Create audio I/O handler
        self.audio_io = AudioIO(
            sample_rate=sample_rate,
            block_size=block_size,
            input_channels=input_channels,
            output_channels=output_channels,
        )

        # Create stream manager
        self.stream_manager = StreamManager(sample_rate=sample_rate)
        self.stream_manager.audio_io = self.audio_io

        # State tracking
        self.is_running = False
        self._processing_thread = None
        self._should_stop = threading.Event()

        # CPU usage tracking
        self._cpu_usage = 0.0
        self.latency = 0.0

    def initialise(self) -> bool:
        """
        Initialise the audio engine.

        Returns:
            True if initialisation succeeded, False otherwise
        """
        # Any additional setup that needs to happen
        return True

    def start(self) -> bool:
        """
        Start the audio engine.

        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            return True

        try:
            # Start all processors and audio I/O
            self.stream_manager.start_all()

            # Start monitoring thread
            self._should_stop.clear()
            self._processing_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._processing_thread.start()

            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting audio engine: {e}")
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the audio engine."""
        if not self.is_running:
            return

        # Stop monitoring thread
        self._should_stop.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)

        # Stop all processors and audio I/O
        self.stream_manager.stop_all()
        self.is_running = False

    def add_processor(self, name: str, processor: AudioProcessor) -> None:
        """
        Add an audio processor to the engine.

        Args:
            name: Name of the processor
            processor: The processor instance
        """
        self.stream_manager.add_processor(name, processor)

    def remove_processor(self, name: str) -> None:
        """
        Remove an audio processor from the engine.

        Args:
            name: Name of the processor to remove
        """
        self.stream_manager.remove_processor(name)

    def connect_processors(self, source: str, target: str) -> bool:
        """
        Connect two processors.

        Args:
            source: Name of the source processor
            target: Name of the target processor

        Returns:
            True if connection succeeded, False otherwise
        """
        return self.stream_manager.connect(source, target)

    def disconnect_processors(self, source: str, target: str) -> bool:
        """
        Disconnect two processors.

        Args:
            source: Name of the source processor
            target: Name of the target processor

        Returns:
            True if disconnection succeeded, False otherwise
        """
        return self.stream_manager.disconnect(source, target)

    def connect_to_output(self, source: str) -> bool:
        """
        Connect a processor to the audio output.

        Args:
            source: Name of the source processor

        Returns:
            True if connection succeeded, False otherwise
        """
        return self.stream_manager.connect_to_audio_output(source)

    def connect_from_input(self, target: str) -> bool:
        """
        Connect the audio input to a processor.

        Args:
            target: Name of the target processor

        Returns:
            True if connection succeeded, False otherwise
        """
        return self.stream_manager.connect_from_audio_input(target)

    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage of the audio processing.

        Returns:
            CPU usage as a percentage (0-100)
        """
        return self._cpu_usage

    def get_latency(self) -> float:
        """
        Get the current audio latency.

        Returns:
            Latency in milliseconds
        """
        return self.latency

    def set_devices(self, input_device: Optional[int], output_device: Optional[int]) -> bool:
        """
        Set the input and output devices.

        Args:
            input_device: Input device ID (None for default)
            output_device: Output device ID (None for default)

        Returns:
            True if successful, False otherwise
        """
        was_running = self.is_running
        if was_running:
            self.stop()

        self.audio_io.input_device = input_device
        self.audio_io.output_device = output_device

        if was_running:
            return self.start()
        return True

    def list_devices(self) -> List[Dict]:
        """
        Get a list of available audio devices.

        Returns:
            List of device information dictionaries
        """
        return self.audio_io.list_devices()

    def _monitor_loop(self) -> None:
        """Monitor the audio engine performance in a separate thread."""
        while not self._should_stop.is_set():
            # Calculate CPU usage based on buffer fill levels
            if self.audio_io and self.audio_io.is_running:
                input_fill = self.audio_io.input_buffer.size / 20.0  # Normalize to 0-1
                output_fill = self.audio_io.output_buffer.size / 20.0  # Normalize to 0-1

                # Estimate CPU usage based on buffer fill levels
                # If output buffer is empty, CPU might be struggling to keep up
                self._cpu_usage = 100.0 * (1.0 - min(output_fill, 0.9))

                # Estimate latency
                buffer_latency = self.block_size / self.sample_rate
                buffer_count = (self.audio_io.input_buffer.size + self.audio_io.output_buffer.size) / 2
                self.latency = buffer_count * buffer_latency * 1000  # Convert to milliseconds

            # Sleep to avoid consuming too much CPU
            time.sleep(0.1)


# Example Effect Processor
class GainProcessor(AudioProcessor):
    """
    Simple gain processor that applies amplitude changes to the audio.

    This is an example of how to create a processor subclass.
    """

    def __init__(self, gain: float = 1.0, channels: int = 2, sample_rate: int = 44100):
        """
        Initialise the gain processor.

        Args:
            gain: Initial gain value (1.0 = unity gain)
            channels: Number of audio channels
            sample_rate: Sample rate in Hz
        """
        super().__init__(channels=channels, sample_rate=sample_rate)
        self._gain = gain

    @property
    def gain(self) -> float:
        """Get the current gain value."""
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        """
        Set the gain value.

        Args:
            value: New gain value
        """
        self._gain = max(0.0, min(10.0, value))  # Clamp between 0 and 10

    def process_block(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply gain to the input data.

        Args:
            input_data: Input audio data as numpy array

        Returns:
            Processed audio data as numpy array
        """
        if self.bypass:
            return input_data

        # Simply multiply by the gain value
        return input_data * self._gain


# Example usage
if __name__ == "__main__":
    # Create the audio engine
    engine = AudioEngine(sample_rate=44100, block_size=512)

    # Create some processors
    gain_processor = GainProcessor(gain=0.5)
    engine.add_processor("gain", gain_processor)

    # Connect the processors
    engine.connect_from_input("gain")
    engine.connect_to_output("gain")

    # Start the engine
    if engine.start():
        print("Audio engine started successfully!")
        print("Press Ctrl+C to stop...")
        try:
            # Run for 10 seconds
            time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            # Stop the engine
            engine.stop()
            print("Audio engine stopped.")
    else:
        print("Failed to start audio engine.")

import time
import math
import numpy as np
import sounddevice as sd
from dataclasses import dataclass


@dataclass
class SirenState:
    label: str = "traffic"
    conf: float = 0.0
    consecutive_hits: int = 0
    triggered: bool = False
    last_error: str = ""
    rms: float = 0.0
    db: float = -120.0
    sr_used: int = 0
    last_cb_ts: float = 0.0   
    overflows: int = 0        


def list_mics():
    print("\n--- SOUND DEVICES ---")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        print(f"{i:>3} {d['name']} | in={d.get('max_input_channels',0)} out={d.get('max_output_channels',0)} "
              f"| default_sr={d.get('default_samplerate', None)}")
    print("---------------------\n")


class MicWorker:
    """
    Robust mic worker:
    - Uses device default sample rate (same behavior as your test)
    - Very light callback using a ring buffer (no np.concatenate)
    - Inference errors won't kill audio
    - Restarts stream if callback stalls
    """

    def __init__(self, device_id, infer, window_sec=3, sr=None, threshold=0.85, consecutive_needed=2):
        self.device_id = int(device_id)
        self.infer = infer
        self.window_sec = float(window_sec)
        self.threshold = float(threshold)
        self.consecutive_needed = int(consecutive_needed)

        d = sd.query_devices(self.device_id)
        default_sr = int(d.get("default_samplerate", 48000))
        self.sr = int(default_sr if sr is None else sr)

        self.state = SirenState(sr_used=self.sr)
        self._stop = False

        self._need = int(self.window_sec * self.sr)
        self._buf = np.zeros(self._need * 4, dtype=np.float32)
        self._w = 0  

        self._meter_rms = 0.0
        self._meter_alpha = 0.20

    def stop(self):
        self._stop = True

    def _rms_db(self, audio: np.ndarray):
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        db = 20.0 * math.log10(rms + 1e-12)
        return rms, db

    def _write_ring(self, x: np.ndarray):
        n = x.size
        if n <= 0:
            return

        end = self._w + n
        L = self._buf.size

        if end <= L:
            self._buf[self._w:end] = x
        else:
            first = L - self._w
            self._buf[self._w:] = x[:first]
            self._buf[:end - L] = x[first:]
        self._w = end % L

    def _read_latest_window(self) -> np.ndarray:
        """
        Returns last self._need samples as a contiguous array.
        """
        L = self._buf.size
        start = (self._w - self._need) % L
        if start < self._w:
            return self._buf[start:self._w].copy()
        else:
            return np.concatenate([self._buf[start:], self._buf[:self._w]]).copy()

    def _callback(self, indata, frames, time_info, status):
        if status:
            s = str(status)
            self.state.last_error = s
            if "overflow" in s.lower():
                self.state.overflows += 1

        x = indata[:, 0].astype(np.float32, copy=False)

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        self._meter_rms = self._meter_alpha * rms + (1.0 - self._meter_alpha) * self._meter_rms
        self.state.rms = self._meter_rms
        self.state.db = 20.0 * math.log10(self._meter_rms + 1e-12)

        self._write_ring(x)
        self.state.last_cb_ts = time.time()

    def _open_stream(self):
        extra = None
        try:
            extra = sd.WasapiSettings(exclusive=False)
        except Exception:
            extra = None

        d = sd.query_devices(self.device_id)
        self.sr = int(d.get("default_samplerate", self.sr))
        self.state.sr_used = self.sr
        self._need = int(self.window_sec * self.sr)

        return sd.InputStream(
            device=self.device_id,
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,    
            latency="low",
            extra_settings=extra,
        )

    def run_loop(self):
        while not self._stop:
            stream = None
            try:
                stream = self._open_stream()
                stream.start()
                self.state.last_error = ""  
                last_run = 0.0

                while not self._stop:
                    now = time.time()

                    if self.state.last_cb_ts and (now - self.state.last_cb_ts) > 1.5:
                        self.state.last_error = "Audio callback stalled. Restarting stream..."
                        break

                    if (now - last_run) >= self.window_sec:
                        audio = self._read_latest_window()
                    
                        label, conf = "traffic", 0.0
                        is_emergency = False
                    
                        self.state.consecutive_hits = 0
                        self.state.label = label
                        self.state.conf = conf
                        self.state.triggered = False
                    
                        last_run = now


                    time.sleep(0.02)

            except Exception as e:
                self.state.last_error = f"Stream error: {type(e).__name__}: {e}"
                time.sleep(0.5)

            finally:
                try:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                except Exception:
                    pass

                if not self._stop:
                    time.sleep(0.25)

"""Network communication for JEPA UI subprocess."""
import sys
import json
import threading
import queue
from typing import Dict, Any


class NetworkHandler:
    """Handles stdin/stdout communication with parent process."""
    
    def __init__(self):
        # Keep only the latest message to prevent UI lag
        self.input_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self.running = True
        self.parent_alive = True  # Track if parent process is still connected
        self._start_reader()

    def _start_reader(self):
        t = threading.Thread(target=self._read_stdin, daemon=True)
        t.start()

    def _read_stdin(self):
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    # Parent process closed stdin - keep UI alive but stop reading
                    self.parent_alive = False
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Coalesce: replace queued message with the newest
                try:
                    if self.input_queue.full():
                        _ = self.input_queue.get_nowait()
                    self.input_queue.put_nowait(data)
                except queue.Full:
                    pass
            except (ValueError, OSError, IOError):
                # Pipe broken - parent process likely crashed
                self.parent_alive = False
                break

    def send_action(self, action_name: str, **kwargs):
        msg = {"action": action_name}
        msg.update(kwargs)
        try:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()
        except IOError:
            pass

    def get_messages(self) -> list:
        messages: list[Dict[str, Any]] = []
        while not self.input_queue.empty():
            messages.append(self.input_queue.get())
        return messages

    def stop(self):
        self.running = False

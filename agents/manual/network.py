import sys
import json
import threading
import queue
from typing import Dict, Any, Optional

class NetworkHandler:
    def __init__(self):
        # Keep only the latest message so the UI never lags behind trying to process a backlog.
        # This also prevents pipe backpressure from slowing down the training process.
        self.input_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self.running = True
        self._start_reader()

    def _start_reader(self):
        t = threading.Thread(target=self._read_stdin, daemon=True)
        t.start()

    def _read_stdin(self):
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
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
            except ValueError:
                break

    def send_action(self, action_name: str, **kwargs):
        msg = {"action": action_name}
        msg.update(kwargs)
        try:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()
        except IOError:
            pass # Handle broken pipe or closed stdout

    def get_messages(self) -> list:
        # Drain and return the latest message (at most 1 due to queue maxsize=1)
        messages: list[Dict[str, Any]] = []
        while not self.input_queue.empty():
            messages.append(self.input_queue.get())
        return messages

    def stop(self):
        self.running = False


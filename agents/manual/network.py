import sys
import json
import threading
import queue
from typing import Dict, Any, Optional

class NetworkHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
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
                self.input_queue.put(line)
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
        messages = []
        while not self.input_queue.empty():
            line = self.input_queue.get()
            try:
                data = json.loads(line)
                messages.append(data)
            except json.JSONDecodeError:
                pass
        return messages

    def stop(self):
        self.running = False


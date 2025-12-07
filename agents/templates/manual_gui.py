import sys
import json
import base64
import tkinter as tk
from io import BytesIO
from PIL import Image, ImageTk

def main():
    with open("gui_log.txt", "w") as f:
        f.write("Starting GUI...\n")
    
    try:
        root = tk.Tk()
        with open("gui_log.txt", "a") as f:
            f.write("Tk created.\n")
    except Exception as e:
        with open("gui_log.txt", "a") as f:
            f.write(f"Tk creation failed: {e}\n")
        return

    root.title("ARC-AGI-3 Viewer")
    root.geometry("800x850")

    # Info Label
    info_frame = tk.Frame(root)
    info_frame.pack(side=tk.TOP, fill=tk.X)
    
    lbl_instructions = tk.Label(info_frame, text="Controls: Arrows(1-4), Space(5), Click(6), Enter(7), R(eset), Q(uit)", font=("Arial", 10, "bold"))
    lbl_instructions.pack(anchor=tk.W)

    canvas = tk.Canvas(root, bg="black")
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Store current game info for click handling
    current_state = {"scale": 15, "game_id": ""}

    def send_action(action_name, **kwargs):
        msg = {"action": action_name}
        msg.update(kwargs)
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    # Key bindings
    root.bind("<Up>", lambda e: send_action("ACTION1"))
    root.bind("<Down>", lambda e: send_action("ACTION2"))
    root.bind("<Left>", lambda e: send_action("ACTION3"))
    root.bind("<Right>", lambda e: send_action("ACTION4"))
    root.bind("<space>", lambda e: send_action("ACTION5"))
    root.bind("<Return>", lambda e: send_action("ACTION7"))
    root.bind("r", lambda e: send_action("RESET"))
    root.bind("R", lambda e: send_action("RESET"))
    root.bind("q", lambda e: send_action("QUIT"))
    root.bind("Q", lambda e: send_action("QUIT"))

    def on_click(event):
        # Action 6
        scale = current_state["scale"]
        grid_x = event.x // scale
        grid_y = event.y // scale
        if 0 <= grid_x <= 63 and 0 <= grid_y <= 63:
            send_action("ACTION6", data={"game_id": current_state["game_id"], "x": grid_x, "y": grid_y})

    canvas.bind("<Button-1>", on_click)
    
    # Check for stdin input
    def check_stdin():
        while True:
            # Non-blocking read would be ideal, but in Tkinter we can just rely on
            # the fact that the agent sends one line per frame and waits for action.
            # But the agent might wait for us.
            # So we use a separate thread to read stdin and put into a queue?
            # Or just use root.after to poll?
            # Stdin read is blocking.
            pass
            break
    
    # We need a thread to read stdin so we don't block GUI
    import threading
    import queue
    input_queue = queue.Queue()

    def read_stdin():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                input_queue.put(line)
            except ValueError:
                break
    
    t = threading.Thread(target=read_stdin, daemon=True)
    t.start()

    def process_queue():
        while not input_queue.empty():
            line = input_queue.get()
            try:
                data = json.loads(line)
                if "image" in data:
                    # Render image
                    img_bytes = base64.b64decode(data["image"])
                    pil_img = Image.open(BytesIO(img_bytes))
                    tk_img = ImageTk.PhotoImage(pil_img)
                    
                    # Keep reference
                    canvas.image = tk_img 
                    canvas.delete("all")
                    canvas.config(width=pil_img.width, height=pil_img.height)
                    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
                    
                    if "game_id" in data:
                        current_state["game_id"] = data["game_id"]
                        root.title(f"Game: {data['game_id']}")

            except json.JSONDecodeError:
                pass
            except Exception as e:
                # Log to stderr so it doesn't mess up stdout protocol
                sys.stderr.write(f"GUI Error: {e}\n")
        
        root.after(50, process_queue)

    root.after(50, process_queue)
    
    # Handle close
    def on_close():
        with open("gui_log.txt", "a") as f:
            f.write("Window closing.\n")
        send_action("QUIT")
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    
    with open("gui_log.txt", "a") as f:
        f.write("Entering mainloop.\n")
    try:
        root.mainloop()
    except Exception as e:
        with open("gui_log.txt", "a") as f:
            f.write(f"Mainloop error: {e}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("gui_log.txt", "a") as f:
            f.write(f"Fatal error: {e}\n")


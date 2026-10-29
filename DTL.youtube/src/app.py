"""
Hand Sign Recognition GUI Application
Main application with modern Tkinter interface
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import time
import config
from simple_classifier import SimpleHandSignClassifier
from data_collector import HandSignDataCollector


class HandSignApp:
    """Main application class for Hand Sign Recognition"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Hand Sign Recognition System Pro")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg=config.THEME_BG)
        
        # Initialize components
        self.classifier = SimpleHandSignClassifier()
        self.data_collector = HandSignDataCollector()
        self.cap = self._open_camera(config.CAMERA_ID)
        self.running = False
        
        if self.cap is not None and self.cap.isOpened():
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.running = True
        else:
            messagebox.showerror("Camera Error", "Unable to open camera. Please check device index.")
        
        # State variables
        self.mode = "recognition"  # recognition or collection
        self.recording = False
        self.current_hand_sign = tk.StringVar()
        
        # Setup UI
        self._setup_styles()
        self._create_ui()
        self._start_video_thread()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _open_camera(self, camera_id):
        """Open camera with fallback backends."""
        for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap = cv2.VideoCapture(camera_id, backend) if backend else cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok: return cap
                cap.release()
        return None

    def _setup_styles(self):
        """Setup custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Base Frame Style
        style.configure("TFrame", background=config.THEME_BG)
        style.configure("Panel.TFrame", background=config.THEME_PANEL)
        
        # Label Styles
        style.configure("TLabel", background=config.THEME_BG, foreground=config.THEME_FG, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=config.THEME_BG, foreground=config.THEME_ACCENT, font=("Segoe UI", 18, "bold"))
        style.configure("Header.TLabel", background=config.THEME_PANEL, foreground=config.THEME_ACCENT, font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", background=config.THEME_BG, foreground="#888888", font=("Segoe UI", 9))
        
        # Button Styles
        style.configure("TButton", padding=8, font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background=config.THEME_ACCENT, foreground="white")
        style.map("Accent.TButton", background=[('active', config.THEME_ACCENT_HOVER)])
        
        style.configure("Success.TButton", background=config.THEME_SUCCESS, foreground="white")
        style.configure("Danger.TButton", background=config.THEME_DANGER, foreground="white")
        
        # Entry Style
        style.configure("TEntry", fieldbackground=config.THEME_PANEL, foreground="white", insertcolor="white")
        
        # Labelframe Style
        style.configure("TLabelframe", background=config.THEME_PANEL, foreground=config.THEME_ACCENT)
        style.configure("TLabelframe.Label", background=config.THEME_PANEL, foreground=config.THEME_ACCENT, font=("Segoe UI", 10, "bold"))

    def _create_ui(self):
        """Create the user interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Hand Sign System", style="Title.TLabel").pack(side=tk.LEFT)
        
        self.mode_label = ttk.Label(header_frame, text="MODE: RECOGNITION", foreground=config.THEME_ACCENT, font=("Segoe UI", 10, "bold"))
        self.mode_label.pack(side=tk.RIGHT, pady=5)
        
        # Content area split into Left (Video) and Right (Controls)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- LEFT SIDE: Video Feed ---
        video_container = ttk.Frame(content_frame, style="Panel.TFrame")
        video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_container, background="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # --- RIGHT SIDE: Controls ---
        control_panel = ttk.Frame(content_frame, width=350, style="Panel.TFrame")
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_panel.pack_propagate(False)
        
        # Mode Switcher
        mode_frame = ttk.Frame(control_panel, style="Panel.TFrame")
        mode_frame.pack(fill=tk.X, padx=15, pady=15)
        
        self.rec_btn = ttk.Button(mode_frame, text="Recognition", command=lambda: self._set_mode("recognition"))
        self.rec_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.coll_btn = ttk.Button(mode_frame, text="Collection", command=lambda: self._set_mode("collection"))
        self.coll_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # separator
        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, padx=15)
        # Dynamic Content (Changes based on mode)
        self.dynamic_frame = ttk.Frame(control_panel, style="Panel.TFrame")
        self.dynamic_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self._setup_recognition_ui()
        self._setup_collection_ui()
        
        # Footer / Status Bar
        footer = ttk.Frame(main_frame)
        footer.pack(fill=tk.X, pady=(20, 0))
        
        self.fps_label = ttk.Label(footer, text="FPS: 0", style="Status.TLabel")
        self.fps_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(footer, text="System Ready", style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(footer, text="Exit Application", command=self._on_close).pack(side=tk.RIGHT)

        self._set_mode("recognition") # Default

    def _setup_recognition_ui(self):
        """UI for Recognition Mode"""
        self.rec_ui = ttk.Frame(self.dynamic_frame, style="Panel.TFrame")
        
        ttk.Label(self.rec_ui, text="DETECTION RESULTS", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 15))
        
        res_box = ttk.Frame(self.rec_ui, style="Panel.TFrame")
        res_box.pack(fill=tk.X, pady=10)
        
        self.prediction_label = ttk.Label(res_box, text="Searching...", font=("Segoe UI", 16, "bold"), foreground=config.THEME_ACCENT)
        self.prediction_label.pack(pady=5)
        
        self.confidence_bar = ttk.Progressbar(res_box, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.confidence_bar.pack(fill=tk.X, pady=10)
        
        self.confidence_label = ttk.Label(res_box, text="Confidence: 0%", style="Status.TLabel")
        self.confidence_label.pack()

    def _setup_collection_ui(self):
        """UI for Collection Mode"""
        self.coll_ui = ttk.Frame(self.dynamic_frame, style="Panel.TFrame")
        
        ttk.Label(self.coll_ui, text="NEW HAND SIGN", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        input_frame = ttk.Frame(self.coll_ui, style="Panel.TFrame")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.hand_sign_entry = ttk.Entry(input_frame)
        self.hand_sign_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(input_frame, text="Add", command=self._add_new_hand_sign).pack(side=tk.RIGHT)
        
        ttk.Label(self.coll_ui, text="EXISTING SIGNS", style="Header.TLabel").pack(anchor=tk.W, pady=(20, 10))
        
        list_frame = ttk.Frame(self.coll_ui, style="Panel.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.signs_list = tk.Listbox(list_frame, bg="#252525", fg=config.THEME_FG, borderwidth=0, highlightthickness=0, font=("Segoe UI", 10))
        self.signs_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.signs_list.bind('<<ListboxSelect>>', self._on_sign_selected)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.signs_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.signs_list.config(yscrollcommand=scrollbar.set)
        
        self.record_btn = ttk.Button(self.coll_ui, text="Start Recording", command=self._toggle_recording)
        self.record_btn.pack(fill=tk.X, pady=20)
        
        self.stats_label = ttk.Label(self.coll_ui, text="Total Images: 0", style="Status.TLabel")
        self.stats_label.pack()

    def _set_mode(self, mode):
        """Switch between recognition and collection modes"""
        self.mode = mode
        if mode == "recognition":
            self.coll_ui.pack_forget()
            self.rec_ui.pack(fill=tk.BOTH, expand=True)
            self.mode_label.config(text="MODE: RECOGNITION", foreground=config.THEME_ACCENT)
            if self.recording: self._toggle_recording()
        else:
            self.rec_ui.pack_forget()
            self.coll_ui.pack(fill=tk.BOTH, expand=True)
            self.mode_label.config(text="MODE: COLLECTION", foreground="#FFB703")
            self._update_hand_signs_list()
        
        self.status_label.config(text=f"Switched to {mode} mode")

    def _add_new_hand_sign(self):
        """Handle adding new sign"""
        name = self.hand_sign_entry.get().strip()
        if not name: return
        
        if self.data_collector.start_collection(name):
            self._update_hand_signs_list()
            self.current_hand_sign.set(name)
            self.hand_sign_entry.delete(0, tk.END)
            self.status_label.config(text=f"Added '{name}'")
        else:
            messagebox.showerror("Error", "Could not create folder")

    def _update_hand_signs_list(self):
        """Refresh the listbox"""
        self.signs_list.delete(0, tk.END)
        for sign in self.data_collector.get_available_hand_signs():
            count = self.data_collector.get_collection_stats(sign)
            self.signs_list.insert(tk.END, f"  {sign} ({count})")

    def _on_sign_selected(self, event):
        """Handle selection in listbox"""
        selection = self.signs_list.curselection()
        if selection:
            text = self.signs_list.get(selection[0])
            name = text.strip().split(" (")[0]
            self.current_hand_sign.set(name)
            self.status_label.config(text=f"Selected: {name}")

    def _toggle_recording(self):
        """Toggle the image saving process"""
        if not self.current_hand_sign.get():
            messagebox.showwarning("Warning", "Select a hand sign first")
            return
            
        self.recording = not self.recording
        if self.recording:
            self.data_collector.start_collection(self.current_hand_sign.get())
            self.record_btn.config(text="Stop Recording")
            self.status_label.config(text=f"Recording {self.current_hand_sign.get()}...")
        else:
            self.data_collector.stop_collection()
            self.record_btn.config(text="Start Recording")
            self._update_hand_signs_list()
            self.status_label.config(text="Recording Stopped")

    def _start_video_thread(self):
        """Main processing loop in background thread"""
        def video_loop():
            prev_time = time.time()
            while self.running:
                try:
                    success, frame = self.cap.read()
                    if not success: break
                    
                    frame = cv2.flip(frame, 1)
                    
                    if self.mode == "recognition":
                        result = self.classifier.predict(frame)
                    else:
                        result = self.data_collector.collect_frame(frame)
                        if self.recording and result['detected']:
                            self.data_collector.save_image(result['processed_image'])
                    
                    # Schedule UI updates on the main thread
                    self.root.after(0, self._update_ui_elements, frame, result if self.mode == "recognition" else None, result if self.mode == "collection" else None)
                    
                    # FPS calculation
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    self.root.after(0, lambda f=fps: self.fps_label.config(text=f"FPS: {f:.1f}"))
                    
                except Exception as e:
                    import traceback
                    print(f"Loop error: {e}")
                    traceback.print_exc()
                    time.sleep(0.1)
        
        threading.Thread(target=video_loop, daemon=True).start()

    def _update_ui_elements(self, frame, rec_result=None, coll_result=None):
        """Thread-safe UI updates"""
        try:
            if rec_result:
                self._update_rec_ui(rec_result)
                frame = self._draw_recognition(frame, rec_result)
            elif coll_result:
                self._update_coll_ui(coll_result)
                frame = self._draw_collection(frame, coll_result)
            
            self._display_frame(frame)
        except Exception as e:
            print(f"UI Update error: {e}")

    def _update_rec_ui(self, result):
        """Update recognition labels"""
        if result['detected']:
            self.prediction_label.config(text=result['label'], foreground="#00FF00")
            self.confidence_bar['value'] = result['confidence'] * 100
            self.confidence_label.config(text=f"Confidence: {result['confidence']*100:.1f}%")
        else:
            self.prediction_label.config(text="Searching...", foreground=config.THEME_ACCENT)
            self.confidence_bar['value'] = 0
            self.confidence_label.config(text="Confidence: 0%")

    def _update_coll_ui(self, result):
        """Update collection stats"""
        self.stats_label.config(text=f"Images Collected: {self.data_collector.counter}")

    def _draw_recognition(self, frame, result):
        if result['detected'] and result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_PURPLE, 2)
            cv2.putText(frame, result['label'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_PURPLE, 2)
        return frame

    def _draw_collection(self, frame, result):
        if result['detected'] and result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            color = config.COLOR_GREEN if self.recording else config.COLOR_CYAN
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status = "REC" if self.recording else "READY"
            cv2.putText(frame, status, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    def _display_frame(self, frame):
        """Render frame to UI"""
        if frame is None or frame.size == 0:
            return
        frame = cv2.resize(frame, (800, 500))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _on_close(self):
        self.running = False
        if self.cap: self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = HandSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
Hand Sign Recognition - Modern UI Application
Professional interface with smooth animations and modern design
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


class ModernHandSignApp:
    """Modern Hand Sign Recognition Application"""
    
    # Modern Color Palette
    COLORS = {
        'bg': '#0f0f1a',
        'panel': '#1a1a2e',
        'card': '#16162a',
        'accent': '#6366f1',
        'accent_hover': '#818cf8',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'text': '#f8fafc',
        'text_secondary': '#94a3b8',
        'border': '#2d2d44',
        'gradient_start': '#6366f1',
        'gradient_end': '#8b5cf6'
    }
    
    def __init__(self, root):
        """Initialize the modern application"""
        self.root = root
        self.root.title("HandSign AI - Smart Gesture Recognition")
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.minsize(1000, 700)
        
        # Initialize components
        self.classifier = SimpleHandSignClassifier()
        self.data_collector = HandSignDataCollector()
        self.cap = self._open_camera(config.CAMERA_ID)
        self.running = False
        
        if self.cap is not None and self.cap.isOpened():
            self.running = True
        else:
            messagebox.showerror("Camera Error", "Unable to open camera. Please check device index.")
        
        # State variables
        self.mode = "recognition"
        self.recording = False
        self.current_hand_sign = tk.StringVar()
        self.fps_value = 0
        
        # Setup custom styles
        self._setup_styles()
        
        # Create UI
        self._create_ui()
        
        # Start video thread
        self._start_video_thread()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _open_camera(self, camera_id):
        """Open camera with fallback backends."""
        for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            cap = cv2.VideoCapture(camera_id, backend) if backend else cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok: 
                    return cap
                cap.release()
        return None
    
    def _setup_styles(self):
        """Setup custom ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main styles
        style.configure("Modern.TFrame", background=self.COLORS['bg'])
        style.configure("Panel.TFrame", background=self.COLORS['panel'])
        style.configure("Card.TFrame", background=self.COLORS['card'])
        
        # Modern Button Styles
        style.configure("Accent.TButton",
                       background=self.COLORS['accent'],
                       foreground=self.COLORS['text'],
                       font=('Inter', 11, 'bold'),
                       padding=(20, 12),
                       borderwidth=0)
        style.map("Accent.TButton",
                 background=[('active', self.COLORS['accent_hover']), ('pressed', self.COLORS['accent'])],
                 foreground=[('active', self.COLORS['text'])])
        
        style.configure("Success.TButton",
                       background=self.COLORS['success'],
                       foreground=self.COLORS['text'],
                       font=('Inter', 10, 'bold'),
                       padding=(15, 10))
        
        style.configure("Danger.TButton",
                       background=self.COLORS['danger'],
                       foreground=self.COLORS['text'],
                       font=('Inter', 10, 'bold'),
                       padding=(15, 10))
        
        # Label Styles
        style.configure("Title.TLabel",
                       background=self.COLORS['bg'],
                       foreground=self.COLORS['text'],
                       font=('Inter', 24, 'bold'))
        
        style.configure("Header.TLabel",
                       background=self.COLORS['panel'],
                       foreground=self.COLORS['accent'],
                       font=('Inter', 14, 'bold'))
        
        style.configure("Status.TLabel",
                       background=self.COLORS['bg'],
                       foreground=self.COLORS['text_secondary'],
                       font=('Inter', 10))
        
        style.configure("Result.TLabel",
                       background=self.COLORS['card'],
                       foreground=self.COLORS['text'],
                       font=('Inter', 20, 'bold'))
        
        # Entry Style
        style.configure("Modern.TEntry",
                       fieldbackground=self.COLORS['panel'],
                       foreground=self.COLORS['text'],
                       insertcolor=self.COLORS['text'],
                       font=('Inter', 11),
                       padding=10)
        
        # Progress Bar
        style.configure("Modern.Horizontal.TProgressbar",
                       background=self.COLORS['accent'],
                       troughcolor=self.COLORS['panel'],
                       borderwidth=0,
                       lightcolor=self.COLORS['accent'],
                       darkcolor=self.COLORS['accent'])
    
    def _create_ui(self):
        """Create modern user interface"""
        # Main Container
        main_container = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header Section
        self._create_header(main_container)
        
        # Content Area
        content = tk.Frame(main_container, bg=self.COLORS['bg'])
        content.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left: Video Feed Area
        video_section = self._create_video_section(content)
        video_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Right: Control Panel
        control_section = self._create_control_section(content)
        control_section.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        
        # Footer Status Bar
        self._create_footer(main_container)
    
    def _create_header(self, parent):
        """Create modern header with logo and mode switcher"""
        header = tk.Frame(parent, bg=self.COLORS['bg'])
        header.pack(fill=tk.X, pady=(0, 10))
        
        # Logo & Title
        logo_frame = tk.Frame(header, bg=self.COLORS['bg'])
        logo_frame.pack(side=tk.LEFT)
        
        # Animated gradient text effect
        title = tk.Label(logo_frame, 
                        text="✋ HandSign AI",
                        bg=self.COLORS['bg'],
                        fg=self.COLORS['accent'],
                        font=('Inter', 28, 'bold'))
        title.pack(side=tk.LEFT)
        
        subtitle = tk.Label(logo_frame,
                           text="Smart Gesture Recognition",
                           bg=self.COLORS['bg'],
                           fg=self.COLORS['text_secondary'],
                           font=('Inter', 12))
        subtitle.pack(side=tk.LEFT, padx=(10, 0), pady=(8, 0))
        
        # Mode Switcher (Segmented Control)
        mode_frame = tk.Frame(header, bg=self.COLORS['panel'], highlightbackground=self.COLORS['border'], highlightthickness=1)
        mode_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.recognition_btn = tk.Button(mode_frame,
                                        text="🔍 Recognition",
                                        bg=self.COLORS['accent'],
                                        fg=self.COLORS['text'],
                                        font=('Inter', 11, 'bold'),
                                        padx=20, pady=8,
                                        borderwidth=0,
                                        cursor='hand2',
                                        command=lambda: self._set_mode("recognition"))
        self.recognition_btn.pack(side=tk.LEFT, padx=3, pady=3)
        
        self.collection_btn = tk.Button(mode_frame,
                                       text="📁 Collection",
                                       bg=self.COLORS['panel'],
                                       fg=self.COLORS['text_secondary'],
                                       font=('Inter', 11),
                                       padx=20, pady=8,
                                       borderwidth=0,
                                       cursor='hand2',
                                       command=lambda: self._set_mode("collection"))
        self.collection_btn.pack(side=tk.LEFT, padx=3, pady=3)
    
    def _create_video_section(self, parent):
        """Create video feed display area"""
        video_container = tk.Frame(parent, bg=self.COLORS['panel'], highlightbackground=self.COLORS['border'], highlightthickness=2)
        
        # Video Header
        video_header = tk.Frame(video_container, bg=self.COLORS['panel'])
        video_header.pack(fill=tk.X, padx=15, pady=10)
        
        self.camera_status = tk.Label(video_header,
                                     text="● Camera Active",
                                     bg=self.COLORS['panel'],
                                     fg=self.COLORS['success'],
                                     font=('Inter', 10, 'bold'))
        self.camera_status.pack(side=tk.LEFT)
        
        self.fps_display = tk.Label(video_header,
                                   text="60 FPS",
                                   bg=self.COLORS['panel'],
                                   fg=self.COLORS['text_secondary'],
                                   font=('Inter', 10))
        self.fps_display.pack(side=tk.RIGHT)
        
        # Video Canvas
        self.video_frame = tk.Frame(video_container, bg='black', width=800, height=500)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        return video_container
    
    def _create_control_section(self, parent):
        """Create control panel with dynamic content"""
        control_container = tk.Frame(parent, bg=self.COLORS['panel'], width=380, highlightbackground=self.COLORS['border'], highlightthickness=2)
        control_container.pack_propagate(False)
        
        # Dynamic Content Frame
        self.control_content = tk.Frame(control_container, bg=self.COLORS['panel'])
        self.control_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create both mode UIs
        self._create_recognition_ui()
        self._create_collection_ui()
        
        return control_container
    
    def _create_recognition_ui(self):
        """Create recognition mode UI"""
        self.rec_ui = tk.Frame(self.control_content, bg=self.COLORS['panel'])
        
        # Section Header
        header = tk.Label(self.rec_ui,
                         text="🎯 Recognition Results",
                         bg=self.COLORS['panel'],
                         fg=self.COLORS['text'],
                         font=('Inter', 16, 'bold'))
        header.pack(anchor='w', pady=(0, 20))
        
        # Result Card
        result_card = tk.Frame(self.rec_ui, bg=self.COLORS['card'], highlightbackground=self.COLORS['border'], highlightthickness=1)
        result_card.pack(fill=tk.X, pady=10)
        
        # Prediction Label
        self.prediction_label = tk.Label(result_card,
                                        text="Show a hand sign",
                                        bg=self.COLORS['card'],
                                        fg=self.COLORS['text_secondary'],
                                        font=('Inter', 18, 'bold'))
        self.prediction_label.pack(pady=(20, 10))
        
        # Confidence Section
        conf_frame = tk.Frame(result_card, bg=self.COLORS['card'])
        conf_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.confidence_label = tk.Label(conf_frame,
                                        text="Confidence: 0%",
                                        bg=self.COLORS['card'],
                                        fg=self.COLORS['text_secondary'],
                                        font=('Inter', 11))
        self.confidence_label.pack(anchor='w')
        
        # Progress Bar
        self.confidence_bar = ttk.Progressbar(conf_frame,
                                             orient=tk.HORIZONTAL,
                                             length=100,
                                             mode='determinate',
                                             style="Modern.Horizontal.TProgressbar")
        self.confidence_bar.pack(fill=tk.X, pady=(8, 0))
        self.confidence_bar['value'] = 0
        
        # Instructions
        instructions = tk.Frame(self.rec_ui, bg=self.COLORS['panel'])
        instructions.pack(fill=tk.X, pady=(30, 0))
        
        tk.Label(instructions,
                text="💡 Tips:",
                bg=self.COLORS['panel'],
                fg=self.COLORS['accent'],
                font=('Inter', 12, 'bold')).pack(anchor='w')
        
        tips_text = "• Keep hand within frame\n• Ensure good lighting\n• Hold sign steady for best results"
        tk.Label(instructions,
                text=tips_text,
                bg=self.COLORS['panel'],
                fg=self.COLORS['text_secondary'],
                font=('Inter', 10),
                justify=tk.LEFT).pack(anchor='w', pady=(5, 0))
    
    def _create_collection_ui(self):
        """Create collection mode UI"""
        self.coll_ui = tk.Frame(self.control_content, bg=self.COLORS['panel'])
        
        # Section Header
        header = tk.Label(self.coll_ui,
                         text="📸 Data Collection",
                         bg=self.COLORS['panel'],
                         fg=self.COLORS['text'],
                         font=('Inter', 16, 'bold'))
        header.pack(anchor='w', pady=(0, 20))
        
        # New Sign Input
        input_card = tk.Frame(self.coll_ui, bg=self.COLORS['card'], highlightbackground=self.COLORS['border'], highlightthickness=1)
        input_card.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(input_card,
                text="New Hand Sign",
                bg=self.COLORS['card'],
                fg=self.COLORS['text_secondary'],
                font=('Inter', 11)).pack(anchor='w', padx=15, pady=(15, 5))
        
        input_row = tk.Frame(input_card, bg=self.COLORS['card'])
        input_row.pack(fill=tk.X, padx=15, pady=(5, 15))
        
        self.hand_sign_entry = tk.Entry(input_row,
                                       bg=self.COLORS['panel'],
                                       fg=self.COLORS['text'],
                                       font=('Inter', 12),
                                       relief=tk.FLAT,
                                       insertbackground=self.COLORS['text'])
        self.hand_sign_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        
        add_btn = tk.Button(input_row,
                           text="Add",
                           bg=self.COLORS['success'],
                           fg=self.COLORS['text'],
                           font=('Inter', 10, 'bold'),
                           padx=15, pady=5,
                           borderwidth=0,
                           cursor='hand2',
                           command=self._add_new_hand_sign)
        add_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Existing Signs List
        tk.Label(self.coll_ui,
                text="Saved Signs",
                bg=self.COLORS['panel'],
                fg=self.COLORS['accent'],
                font=('Inter', 12, 'bold')).pack(anchor='w', pady=(10, 10))
        
        # List container with scrollbar
        list_container = tk.Frame(self.coll_ui, bg=self.COLORS['card'], highlightbackground=self.COLORS['border'], highlightthickness=1)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Custom styled listbox
        self.signs_list = tk.Listbox(list_container,
                                    bg=self.COLORS['card'],
                                    fg=self.COLORS['text'],
                                    selectbackground=self.COLORS['accent'],
                                    selectforeground=self.COLORS['text'],
                                    font=('Inter', 11),
                                    relief=tk.FLAT,
                                    highlightthickness=0,
                                    borderwidth=0,
                                    activestyle='none')
        self.signs_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.signs_list.bind('<<ListboxSelect>>', self._on_sign_selected)
        
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.signs_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        self.signs_list.config(yscrollcommand=scrollbar.set)
        
        # Recording Controls
        controls_card = tk.Frame(self.coll_ui, bg=self.COLORS['card'], highlightbackground=self.COLORS['border'], highlightthickness=1)
        controls_card.pack(fill=tk.X, pady=(0, 10))
        
        self.record_btn = tk.Button(controls_card,
                                   text="● Start Recording",
                                   bg=self.COLORS['danger'],
                                   fg=self.COLORS['text'],
                                   font=('Inter', 12, 'bold'),
                                   padx=20, pady=12,
                                   borderwidth=0,
                                   cursor='hand2',
                                   command=self._toggle_recording)
        self.record_btn.pack(fill=tk.X, padx=15, pady=15)
        
        # Stats
        self.stats_label = tk.Label(controls_card,
                                   text="Images: 0",
                                   bg=self.COLORS['card'],
                                   fg=self.COLORS['text_secondary'],
                                   font=('Inter', 11))
        self.stats_label.pack(pady=(0, 15))
    
    def _create_footer(self, parent):
        """Create status footer"""
        footer = tk.Frame(parent, bg=self.COLORS['bg'])
        footer.pack(fill=tk.X, pady=(15, 0))
        
        # Left: Status message
        self.status_label = tk.Label(footer,
                                    text="System ready",
                                    bg=self.COLORS['bg'],
                                    fg=self.COLORS['text_secondary'],
                                    font=('Inter', 10))
        self.status_label.pack(side=tk.LEFT)
        
        # Right: Exit button
        exit_btn = tk.Button(footer,
                            text="Exit Application",
                            bg=self.COLORS['panel'],
                            fg=self.COLORS['text_secondary'],
                            font=('Inter', 10),
                            padx=15, pady=6,
                            borderwidth=0,
                            cursor='hand2',
                            command=self._on_close)
        exit_btn.pack(side=tk.RIGHT)
        
        # Version
        version = tk.Label(footer,
                          text="v2.0",
                          bg=self.COLORS['bg'],
                          fg=self.COLORS['text_secondary'],
                          font=('Inter', 9))
        version.pack(side=tk.RIGHT, padx=(0, 20))
    
    def _set_mode(self, mode):
        """Switch between recognition and collection modes"""
        self.mode = mode
        
        if mode == "recognition":
            self.recognition_btn.configure(bg=self.COLORS['accent'], fg=self.COLORS['text'], font=('Inter', 11, 'bold'))
            self.collection_btn.configure(bg=self.COLORS['panel'], fg=self.COLORS['text_secondary'], font=('Inter', 11))
            self.coll_ui.pack_forget()
            self.rec_ui.pack(fill=tk.BOTH, expand=True)
            self.status_label.config(text="Recognition mode active - Show a hand sign")
            if self.recording: 
                self._toggle_recording()
        else:
            self.recognition_btn.configure(bg=self.COLORS['panel'], fg=self.COLORS['text_secondary'], font=('Inter', 11))
            self.collection_btn.configure(bg=self.COLORS['accent'], fg=self.COLORS['text'], font=('Inter', 11, 'bold'))
            self.rec_ui.pack_forget()
            self.coll_ui.pack(fill=tk.BOTH, expand=True)
            self._update_hand_signs_list()
            self.status_label.config(text="Collection mode active - Add and record hand signs")
    
    def _add_new_hand_sign(self):
        """Handle adding new sign"""
        name = self.hand_sign_entry.get().strip()
        if not name: 
            self.status_label.config(text="Please enter a sign name")
            return
        
        if self.data_collector.start_collection(name):
            self._update_hand_signs_list()
            self.current_hand_sign.set(name)
            self.hand_sign_entry.delete(0, tk.END)
            self.status_label.config(text=f"✓ Added sign '{name}' - Select it and start recording")
        else:
            messagebox.showerror("Error", "Could not create folder")
    
    def _update_hand_signs_list(self):
        """Refresh the listbox"""
        self.signs_list.delete(0, tk.END)
        for sign in self.data_collector.get_available_hand_signs():
            count = self.data_collector.get_collection_stats(sign)
            self.signs_list.insert(tk.END, f"  {sign} ({count} images)")
    
    def _on_sign_selected(self, event):
        """Handle selection in listbox"""
        selection = self.signs_list.curselection()
        if selection:
            text = self.signs_list.get(selection[0])
            name = text.strip().split(" (")[0].strip()
            self.current_hand_sign.set(name)
            self.status_label.config(text=f"Selected: {name}")
    
    def _toggle_recording(self):
        """Toggle the image saving process"""
        if not self.current_hand_sign.get():
            self.status_label.config(text="⚠ Select a hand sign first")
            return
        
        self.recording = not self.recording
        if self.recording:
            self.data_collector.start_collection(self.current_hand_sign.get())
            self.record_btn.config(text="⏹ Stop Recording", bg=self.COLORS['danger'])
            self.status_label.config(text=f"● Recording {self.current_hand_sign.get()}...")
        else:
            self.data_collector.stop_collection()
            self.record_btn.config(text="● Start Recording", bg=self.COLORS['success'])
            self._update_hand_signs_list()
            self.status_label.config(text=f"✓ Recording saved for '{self.current_hand_sign.get()}'")
    
    def _start_video_thread(self):
        """Main processing loop in background thread"""
        def video_loop():
            prev_time = time.time()
            while self.running:
                try:
                    success, frame = self.cap.read()
                    if not success: 
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    if self.mode == "recognition":
                        result = self.classifier.predict(frame)
                    else:
                        result = self.data_collector.collect_frame(frame)
                        if self.recording and result['detected']:
                            self.data_collector.save_image(result['processed_image'])
                    
                    # FPS calculation
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    
                    # Schedule UI updates on the main thread
                    self.root.after(0, self._update_ui, frame, result, fps)
                    
                except Exception as e:
                    print(f"Loop error: {e}")
                    time.sleep(0.1)
        
        threading.Thread(target=video_loop, daemon=True).start()
    
    def _update_ui(self, frame, result, fps):
        """Thread-safe UI updates"""
        try:
            # Update FPS
            self.fps_display.config(text=f"{int(fps)} FPS")
            
            # Update mode-specific UI
            if self.mode == "recognition":
                self._update_recognition_ui(result)
                frame = self._draw_recognition_overlay(frame, result)
            else:
                self._update_collection_ui(result)
                frame = self._draw_collection_overlay(frame, result)
            
            # Display frame
            self._display_frame(frame)
            
        except Exception as e:
            print(f"UI Update error: {e}")
    
    def _update_recognition_ui(self, result):
        """Update recognition mode UI"""
        if result['detected']:
            label = result['label']
            confidence = result['confidence'] * 100
            
            self.prediction_label.config(text=label, fg=self.COLORS['success'])
            self.confidence_bar['value'] = confidence
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        else:
            self.prediction_label.config(text="Show a hand sign", fg=self.COLORS['text_secondary'])
            self.confidence_bar['value'] = 0
            self.confidence_label.config(text="Confidence: 0%")
    
    def _update_collection_ui(self, result):
        """Update collection mode UI"""
        self.stats_label.config(text=f"Images: {self.data_collector.counter}")
        
        if self.recording and result['detected']:
            self.record_btn.config(text=f"⏹ Stop ({self.data_collector.counter})")
    
    def _draw_recognition_overlay(self, frame, result):
        """Draw overlay for recognition mode"""
        if result['detected'] and result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            # Modern cyan border
            cv2.rectangle(frame, (x1, y1), (x2, y2), (155, 102, 99), 3)
            # Label background
            label = f"{result['label']} ({result['confidence']*100:.0f}%)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1 + text_w + 10, y1), (155, 102, 99), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def _draw_collection_overlay(self, frame, result):
        """Draw overlay for collection mode"""
        if result['detected'] and result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            color = (29, 185, 129) if self.recording else (245, 158, 11)  # Green if recording, amber if not
            status = "REC" if self.recording else "READY"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
    
    def _display_frame(self, frame):
        """Render frame to UI"""
        if frame is None or frame.size == 0:
            return
        
        # Get video frame dimensions
        frame_width = self.video_frame.winfo_width()
        frame_height = self.video_frame.winfo_height()
        
        if frame_width < 100 or frame_height < 100:
            frame_width, frame_height = 800, 500
        
        frame = cv2.resize(frame, (frame_width, frame_height))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def _on_close(self):
        """Clean shutdown"""
        self.running = False
        if self.cap: 
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModernHandSignApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

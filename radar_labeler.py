import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import json
import os
import re
import math
from datetime import datetime, timedelta

# OCR and AIS
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from pyais import decode
    HAS_AIS = True
except ImportError:
    HAS_AIS = False


# Helper function to convert Decimal Degrees to Degrees Decimal Minutes (DMM)
def to_dmm(decimal_degrees, is_lat):
    deg = int(abs(decimal_degrees))
    mins = (abs(decimal_degrees) - deg) * 60
    
    if is_lat:
        hemi = 'N' if decimal_degrees >= 0 else 'S'
        return f"{deg:02d}°{mins:06.4f}'{hemi}"
    else:
        hemi = 'E' if decimal_degrees >= 0 else 'W'
        return f"{deg:03d}°{mins:06.4f}'{hemi}"


class RadarLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Dataset Labeler")
        
        # Image State
        self.image_list = []
        self.image_filenames = []
        self.current_idx = -1
        self.image_path = None
        self.orig_img = None  
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        self.scale = 1.0      
        
        # Annotation & Class State
        self.annotations = []
        self.current_id = 1
        self.categories = {}  
        self.is_dirty = False 
        
        # Log & Map State
        self.raw_log_lines = [] 
        self.ais_targets_to_draw = [] 
        self.current_own_lat = None   
        self.current_own_lon = None
        self.current_heading = None
        self.current_cursor_lat = None 
        self.current_cursor_lon = None
        self.current_radar_range = None
        
        # DYNAMIC CENTER COORDINATES
        self.MY_SHIP_X = 472
        self.MY_SHIP_Y = 815 # Default Off-Centered
        
        # LOCKED CURSOR STATE
        self.locked_cursor_px = None 
        self.custom_pixels_per_nm = None
        
        # Drawing State
        self.start_raw_x = None
        self.start_raw_y = None
        self.current_rect = None
        self.current_polygon_points = []
        
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        # --- Top Control Panel ---
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(control_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save JSON (S)", command=self.save_json).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="<< Prev (A)", command=self.prev_image).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Next (D) >>", command=self.next_image).pack(side=tk.LEFT, padx=0)
        
        # --- NEW: Image Dropdown ---
        tk.Label(control_frame, text="  Image:").pack(side=tk.LEFT, padx=5)
        self.image_combo_var = tk.StringVar()
        self.image_combo = ttk.Combobox(control_frame, textvariable=self.image_combo_var, state="readonly", width=16)
        self.image_combo.pack(side=tk.LEFT)
        self.image_combo.bind("<<ComboboxSelected>>", self.on_image_select)
        
        tk.Label(control_frame, text="  Center:").pack(side=tk.LEFT, padx=15)
        self.center_var = tk.StringVar(value="Off-Centered")
        self.center_combo = ttk.Combobox(control_frame, textvariable=self.center_var, state="readonly", width=12, values=["Off-Centered", "Centered"])
        self.center_combo.pack(side=tk.LEFT)
        self.center_combo.bind("<<ComboboxSelected>>", self.on_center_change)
        
        tk.Label(control_frame, text="  Category:").pack(side=tk.LEFT, padx=15)
        self.category_var = tk.StringVar(value="1 - Default")
        self.category_combo = ttk.Combobox(control_frame, textvariable=self.category_var, state="readonly", width=15)
        self.category_combo.pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="  Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="bbox")
        tk.Radiobutton(control_frame, text="BBox", variable=self.mode_var, value="bbox").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="Mask", variable=self.mode_var, value="mask").pack(side=tk.LEFT)
        
        tk.Button(control_frame, text="Clear Last (L)", command=self.clear_last).pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(control_frame, text="● Saved", fg="green", font=("Arial", 11, "bold"))
        self.status_label.pack(side=tk.RIGHT, padx=15)

        # --- Bottom Status Bar (For Coordinates) ---
        self.coord_label = tk.Label(self.root, text=f"Ready. Right-click to probe coordinates.", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 10, "bold"), bg="#222", fg="#0f0")
        self.coord_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=4)

        # --- Main Layout (Canvas + Sidebar) ---
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left side: Canvas
        canvas_container = tk.Frame(self.main_pane)
        self.main_pane.add(canvas_container, weight=5) 
        
        self.h_scroll = tk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(canvas_container, cursor="crosshair", 
                                xscrollcommand=self.h_scroll.set, 
                                yscrollcommand=self.v_scroll.set, bg="#333333")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)

        # Right side: Vertical Split Sidebar
        self.sidebar_pane = ttk.PanedWindow(self.main_pane, orient=tk.VERTICAL)
        self.main_pane.add(self.sidebar_pane, weight=1)

        # Right-Top: Categories and Annotations
        self.upper_sidebar = tk.Frame(self.sidebar_pane, bg="#f0f0f0", width=250) 
        self.sidebar_pane.add(self.upper_sidebar, weight=3) 

        tk.Label(self.upper_sidebar, text="Categories Legend", bg="#ddd", fg="black", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(0, 5))
        self.legend_frame = tk.Frame(self.upper_sidebar, bg="#f0f0f0")
        self.legend_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(self.upper_sidebar, text="Current Annotations", bg="#ddd", fg="black", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(10, 5))
        self.create_scrollable_sidebar()

        # Right-Bottom: Log/AIS Data Viewer
        self.lower_sidebar = tk.Frame(self.sidebar_pane, bg="#1e1e1e")
        self.sidebar_pane.add(self.lower_sidebar, weight=2)
        
        tk.Label(self.lower_sidebar, text="Synchronized AIS/Log Data", bg="#333", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X)
        
        self.log_text = tk.Text(self.lower_sidebar, wrap=tk.WORD, state=tk.DISABLED, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        log_scroll = tk.Scrollbar(self.lower_sidebar, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Mouse Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # MAC AND WINDOWS RIGHT CLICK COMPATIBILITY
        self.canvas.bind("<Button-2>", self.on_right_click) 
        self.canvas.bind("<Button-3>", self.on_right_click) 
        
        self.canvas.bind("<MouseWheel>", self.zoom_mousewheel) 
        self.canvas.bind("<Button-4>", self.zoom_in)           
        self.canvas.bind("<Button-5>", self.zoom_out)          

    def on_image_select(self, event=None):
        if not self.check_unsaved():
            # Revert dropdown value if user cancels
            if self.image_list and self.current_idx != -1:
                self.image_combo.current(self.current_idx)
            return
            
        selected_idx = self.image_combo.current()
        if selected_idx != self.current_idx and selected_idx != -1:
            self.current_idx = selected_idx
            self.load_image_by_index()

    def on_center_change(self, event=None):
        if self.center_var.get() == "Off-Centered":
            self.MY_SHIP_Y = 815
        else:
            self.MY_SHIP_Y = 472
            
        self.redraw_canvas()
        self.draw_ais_dots()
        self.refresh_sidebar_labels()
        
        if not self.current_own_lat:
            self.coord_label.config(text=f"Ready. Fixed Center set at [{self.MY_SHIP_X}, {self.MY_SHIP_Y}].", fg="#0f0")

    def create_scrollable_sidebar(self):
        self.labels_canvas = tk.Canvas(self.upper_sidebar, bg="#f0f0f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.upper_sidebar, orient="vertical", command=self.labels_canvas.yview)
        self.labels_inner_frame = tk.Frame(self.labels_canvas, bg="#f0f0f0")

        self.labels_inner_frame.bind(
            "<Configure>",
            lambda e: self.labels_canvas.configure(scrollregion=self.labels_canvas.bbox("all"))
        )

        self.labels_canvas.create_window((0, 0), window=self.labels_inner_frame, anchor="nw")
        self.labels_canvas.configure(yscrollcommand=scrollbar.set)
        self.labels_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_shortcuts(self):
        self.root.bind("<Return>", self.on_right_click)
        self.root.bind("<a>", self.prev_image)
        self.root.bind("<A>", self.prev_image)
        self.root.bind("<d>", self.next_image)
        self.root.bind("<D>", self.next_image)
        self.root.bind("<s>", self.save_json)
        self.root.bind("<S>", self.save_json)
        self.root.bind("<l>", self.clear_last)
        self.root.bind("<L>", self.clear_last)

    def set_dirty(self, state):
        self.is_dirty = state
        if self.is_dirty:
            self.status_label.config(text="● Unsaved Changes", fg="red")
        else:
            self.status_label.config(text="● Saved", fg="green")

    def check_unsaved(self):
        if self.is_dirty:
            return messagebox.askyesno("Unsaved Changes", "You have unsaved edits! Do you want to discard them and continue?")
        return True

    def get_selected_cat_id(self, val_str):
        try:
            return int(val_str.split(" - ")[0])
        except:
            return 1

    def load_folder(self):
        if not self.check_unsaved(): 
            return
        
        folder_path = filedialog.askdirectory(title="Select Folder with Images", initialdir=os.getcwd())
        if not folder_path: 
            return
        
        self.categories.clear()
        cat_file = os.path.join(folder_path, "categories.txt")
        if os.path.exists(cat_file):
            with open(cat_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or '=' not in line: 
                        continue
                    parts = line.split('=')
                    cat_name = parts[0].strip()
                    try:
                        cat_id = int(parts[1].strip())
                        self.categories[cat_id] = cat_name
                    except ValueError:
                        pass
                        
        self.update_legend_and_dropdowns()

        valid_exts = {".png", ".jpg", ".jpeg"}
        self.image_list = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
        if not self.image_list:
            messagebox.showwarning("No Images", "No valid images found.")
            return
            
        self.image_filenames = [os.path.basename(p) for p in self.image_list]
        self.image_combo['values'] = self.image_filenames

        self.load_log_data(folder_path)
            
        self.current_idx = 0
        self.load_image_by_index()

    def load_log_data(self, image_folder_path):
        parent_dir = os.path.dirname(image_folder_path)
        base_name = os.path.basename(image_folder_path)
        log_folder = os.path.join(parent_dir, base_name + "_log")
        
        self.raw_log_lines = []
        
        if not os.path.exists(log_folder):
            return

        for log_file in sorted(os.listdir(log_folder)):
            if log_file.lower().endswith(".log"):
                with open(os.path.join(log_folder, log_file), "rb") as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    self.raw_log_lines.extend([line.strip() for line in content.split('\n') if line.strip()])

    def sync_logs_from_image(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.ais_targets_to_draw = []
        self.current_heading = None
        self.current_own_lat = None
        self.current_own_lon = None
        self.current_cursor_lat = None
        self.current_cursor_lon = None
        self.current_radar_range = None 
        
        if not HAS_OCR:
            self.log_text.insert(tk.END, "Missing 'pytesseract'.")
            self.log_text.config(state=tk.DISABLED)
            return

        if not self.raw_log_lines:
            self.log_text.insert(tk.END, "[!] No AIS/Log data found for this folder. Operating in OCR-Only mode.\n\n")

        try:
            w, h = self.orig_img.size
            custom_config = r'--psm 6'
            coord_regex = r"(\d{2,3})[^\d]*(\d{2})[\.,]?\s*(\d{2,})"
            
            # Center Dropdown Sync
            self.MY_SHIP_X = 472
            if self.center_var.get() == "Off-Centered":
                self.MY_SHIP_Y = 815
                center_mode = "Off-Centered"
            else:
                self.MY_SHIP_Y = 472
                center_mode = "Centered"
                
            # --- CROP 1: CURSOR SECTION ---
            cursor_box = (int(w * 0.55), 0, int(w * 0.79), int(h * 0.12))
            cursor_img = self.orig_img.crop(cursor_box)
            c_gray = cursor_img.convert('L')
            c_inv = ImageOps.invert(c_gray)
            c_enh = ImageEnhance.Contrast(c_inv).enhance(3.0)
            cursor_text = pytesseract.image_to_string(c_enh, config=custom_config)
            
            c_text_clean = cursor_text.replace('B', '8').replace('O', '0').replace('o', '0').replace('L.', '.').replace('L', '1').replace('l', '1').replace(',', '.')
            c_matches = re.findall(coord_regex, c_text_clean)
            
            if len(c_matches) >= 2:
                c_lat_val = float(f"{c_matches[0][1]}.{c_matches[0][2]}")
                c_lon_val = float(f"{c_matches[1][1]}.{c_matches[1][2]}")
                self.current_cursor_lat = float(c_matches[0][0]) + (c_lat_val / 60.0)
                self.current_cursor_lon = float(c_matches[1][0]) + (c_lon_val / 60.0)
            
            # --- CROP 2: OWN SHIP SECTION ---
            own_box = (int(w * 0.80), 0, w, int(h * 0.35)) 
            own_img = self.orig_img.crop(own_box)
            o_gray = own_img.convert('L')
            o_inv = ImageOps.invert(o_gray)
            o_enh = ImageEnhance.Contrast(o_inv).enhance(3.0) 
            
            o_big = o_enh.resize((o_enh.width * 2, o_enh.height * 2), Image.Resampling.LANCZOS)
            
            text_o1 = pytesseract.image_to_string(o_enh, config=custom_config)
            text_o2 = pytesseract.image_to_string(o_big, config=custom_config)
            
            def sanitize(t):
                return t.replace('B', '8').replace('O', '0').replace('o', '0').replace('L.', '.').replace('L', '1').replace('l', '1').replace(',', '.')
                
            o1_clean = sanitize(text_o1)
            o2_clean = sanitize(text_o2)
            o_text_combined = o1_clean + "\n\n" + o2_clean
            
            # --- EXTRACT HEADING ---
            self.current_heading = 0.0
            for txt in [o2_clean, o1_clean]:
                heading_line_match = re.search(r"(?:HDG|H.G|CMPS)[^\n]*", txt, re.IGNORECASE)
                if heading_line_match:
                    hdg_line = heading_line_match.group(0)
                    hdg_numbers = re.findall(r"(\d+(?:[\.,]\d+)?)", hdg_line)
                    if hdg_numbers:
                        raw_hdg = hdg_numbers[-1].replace(',', '.')
                        val = float(raw_hdg)
                        if val >= 360:
                            val /= 10.0
                        self.current_heading = val
                        break

            # --- EXTRACT COORDINATES ---
            o_matches = re.findall(coord_regex, o1_clean)

            # --- CROP 3: RADAR RANGE SECTION (WITH 1.5 VS 0.25 FIX) ---
            range_box = (0, 0, int(w * 0.15), int(h * 0.08))
            range_img = self.orig_img.crop(range_box)
            
            r_gray = range_img.convert('L')
            r_inv = ImageOps.invert(r_gray)
            r_enh = ImageEnhance.Contrast(r_inv).enhance(3.0)
            text_a = pytesseract.image_to_string(r_enh, config=custom_config)
            
            r_big = range_img.resize((range_img.width * 2, range_img.height * 2), Image.Resampling.LANCZOS)
            text_b = pytesseract.image_to_string(r_big, config=custom_config)
            
            combined_range_text = (text_a + " " + text_b).upper().replace(',', '.')
            
            # SMART FALLBACK: Calculate Rings FIRST
            fallback_range = None
            rings_match = re.search(r"RINGS?[^\d]*(\d+(?:\.\d+)?)", combined_range_text)
            if rings_match:
                try:
                    r_val = float(rings_match.group(1))
                    if r_val == 0.25: fallback_range = 1.5
                    elif r_val == 0.125: fallback_range = 0.75
                    else: fallback_range = r_val * 5.0
                except ValueError:
                    pass
            
            # MASSIVE FIX: Scrub the Rings text completely before looking for the main Range
            r_text_no_rings = re.sub(r'RINGS?[^\d]*\d+(?:\.\d+)?', ' ', combined_range_text)
            
            r_text_clean = re.sub(r'[^0-9\.\s]', ' ', r_text_no_rings)
            
            for bad, good in [("0 5", "0.5"), ("0 25", "0.25"), ("0 75", "0.75"), ("1 5", "1.5"), (" .", "0."), (". ", ".")]:
                r_text_clean = r_text_clean.replace(bad, good)
                
            valid_ranges = [0.125, 0.25, 0.5, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 96.0]
            
            valid_range_candidates = []
            range_matches = re.findall(r"(\d+(?:\.\d+)?)", r_text_clean)
            for rm in range_matches:
                try:
                    val = float(rm)
                    if val in valid_ranges:
                        valid_range_candidates.append(val)
                    elif (val / 100.0) in valid_ranges and val < 100:
                        valid_range_candidates.append(val / 100.0)
                    elif (val / 10.0) in valid_ranges and val < 20: 
                        valid_range_candidates.append(val / 10.0)
                except ValueError:
                    pass
            
            # The Range is ALWAYS larger than the Rings spacing.
            if valid_range_candidates:
                self.current_radar_range = max(valid_range_candidates)
                
            # If we missed the main range completely, map it dynamically from the rings
            if not self.current_radar_range and fallback_range:
                self.current_radar_range = fallback_range
                
            # If the OCR read "0.25" as the max, but fallback_range says "1.5" (meaning "0.25" was actually the rings)
            if self.current_radar_range and fallback_range and fallback_range > self.current_radar_range:
                self.current_radar_range = fallback_range
            
            # --- PRIMARY LOGIC: ONLY PROCEED IF WE FOUND COORDINATES ---
            if len(o_matches) >= 2:
                o_lat_val = float(f"{o_matches[-2][1]}.{o_matches[-2][2]}")
                o_lon_val = float(f"{o_matches[-1][1]}.{o_matches[-1][2]}")
                
                self.current_own_lat = float(o_matches[-2][0]) + (o_lat_val / 60.0)
                self.current_own_lon = float(o_matches[-1][0]) + (o_lon_val / 60.0)
                
                ocr_own_lat = self.current_own_lat
                ocr_own_lon = self.current_own_lon
                
                # --- EXTRACT TIME ---
                target_utc_hhmm = "UNKNOWN"
                ocr_utc_datetime_str = "UNKNOWN"
                target_total_mins = -1
                
                time_match_1 = None
                time_match_2 = None
                
                t1_matches = re.findall(r"(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2})\s*[:\.]?\s*(\d{2})", o_text_combined)
                if t1_matches: time_match_1 = t1_matches[-1]
                    
                t2_matches = re.findall(r"(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\s+(\d{2})\s*[:\.]?\s*(\d{2})", o_text_combined)
                if t2_matches: time_match_2 = t2_matches[-1]

                try:
                    if time_match_1:
                        day, month, year, hh, mm = time_match_1
                        dt_local = datetime.strptime(f"{day} {month.title()} {year} {hh}:{mm}", "%d %b %Y %H:%M")
                    elif time_match_2:
                        year, month, day, hh, mm = time_match_2
                        dt_local = datetime.strptime(f"{year}-{month.zfill(2)}-{day.zfill(2)} {hh}:{mm}", "%Y-%m-%d %H:%M")
                        
                    if time_match_1 or time_match_2:
                        dt_utc = dt_local - timedelta(hours=2)
                        target_utc_hhmm = dt_utc.strftime("%H%M")
                        ocr_utc_datetime_str = dt_utc.strftime("%Y-%m-%d %H:%M:00")
                        target_total_mins = int(target_utc_hhmm[:2]) * 60 + int(target_utc_hhmm[2:])
                except ValueError:
                    pass

                # --- PULL LOG TRUTH & AIS TIMESTAMPS ---
                best_diff = 999 
                best_lat = None
                best_lon = None
                best_heading = None
                best_line_idx = -1
                latest_log_heading = None
                unique_ais_targets = {}

                if self.raw_log_lines and target_total_mins != -1:
                    for i, line in enumerate(self.raw_log_lines):
                        if "$HEHDT" in line:
                            try:
                                latest_log_heading = float(line.split(',')[1])
                            except: pass
                        elif "$GPGGA" in line:
                            parts = line.split(',')
                            if len(parts) > 5 and len(parts[1]) >= 4:
                                log_hhmm = parts[1][:4]
                                if log_hhmm.isdigit():
                                    log_hr = int(log_hhmm[:2])
                                    log_min = int(log_hhmm[2:])
                                    log_total_mins = log_hr * 60 + log_min
                                    
                                    time_diff = abs(log_total_mins - target_total_mins)
                                    if time_diff > 1000: time_diff = 1440 - time_diff 
                                    
                                    if time_diff <= 3: 
                                        log_lat_str = parts[2]
                                        log_lon_str = parts[4]
                                        if len(log_lat_str) >= 4 and len(log_lon_str) >= 5:
                                            try:
                                                l_lat = float(log_lat_str[:2]) + float(log_lat_str[2:])/60.0
                                                l_lon = float(log_lon_str[:3]) + float(log_lon_str[3:])/60.0
                                                
                                                diff = abs(l_lat - ocr_own_lat) + abs(l_lon - ocr_own_lon)
                                                if diff < best_diff:
                                                    best_diff = diff
                                                    best_lat = l_lat
                                                    best_lon = l_lon
                                                    best_heading = latest_log_heading
                                                    best_line_idx = i
                                            except ValueError:
                                                pass
                    
                    if best_line_idx != -1 and best_diff < 0.05:
                        start_idx = max(0, best_line_idx - 5000)
                        end_idx = min(len(self.raw_log_lines), best_line_idx + 1000)
                        
                        latest_log_time = "Unknown"
                        for i in range(start_idx, end_idx):
                            line = self.raw_log_lines[i]
                            
                            if "$GPZDA" in line:
                                parts = line.split(',')
                                if len(parts) >= 5 and parts[1]:
                                    hhmmss = parts[1].split('.')[0]
                                    day = parts[2].zfill(2)
                                    month = parts[3].zfill(2)
                                    year = parts[4]
                                    latest_log_time = f"{year}-{month}-{day} {hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
                            elif "$GPGGA" in line:
                                parts = line.split(',')
                                if len(parts) > 1 and parts[1]:
                                    hhmmss = parts[1].split('.')[0]
                                    if latest_log_time != "Unknown" and "-" in latest_log_time:
                                        date_part = latest_log_time.split(' ')[0]
                                        latest_log_time = f"{date_part} {hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
                                    else:
                                        latest_log_time = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
                                        
                            if "!AIVDM" in line:
                                if HAS_AIS:
                                    try:
                                        msg = decode(line[line.find("!"):])
                                        if hasattr(msg, 'lat') and hasattr(msg, 'lon') and msg.lat and msg.lon:
                                            unique_ais_targets[msg.mmsi] = (msg.lat, msg.lon, latest_log_time) 
                                    except: pass

                # Apply Log Truth Overrides (Or Fallback to OCR)
                if best_line_idx != -1 and best_diff < 0.05:
                    self.current_own_lat = best_lat
                    self.current_own_lon = best_lon
                    if best_heading is not None:
                        self.current_heading = best_heading 
                        heading_source = "Log Truth"
                    else:
                        heading_source = "OCR Backup"
                else:
                    heading_source = "OCR Only"

                # --- PRINT CONSOLE OUTPUT ---
                self.log_text.insert(tk.END, f"=== IMAGE ANALYSIS ===\n")
                self.log_text.insert(tk.END, f"Radar Center: {center_mode}\n")
                self.log_text.insert(tk.END, f"OCR Datetime: {ocr_utc_datetime_str} UTC\n")
                
                if self.current_radar_range:
                    self.log_text.insert(tk.END, f"Range: {self.current_radar_range} NM\n")
                else:
                    self.log_text.insert(tk.END, f"Range: Failed! (Raw seen: {r_text_clean.strip()})\n")
                
                heading_str = f"{self.current_heading}° ({heading_source})"
                self.log_text.insert(tk.END, f"Heading: {heading_str}\n")
                
                self.log_text.insert(tk.END, f"[Own Ship OCR: {to_dmm(ocr_own_lat, True)}, {to_dmm(ocr_own_lon, False)}]\n")
                
                if self.current_cursor_lat:
                    self.log_text.insert(tk.END, f"[Cursor OCR:   {to_dmm(self.current_cursor_lat, True)}, {to_dmm(self.current_cursor_lon, False)}]\n")
                else:
                    self.log_text.insert(tk.END, f"[Cursor OCR:   Failed]\n")
                    
                self.log_text.insert(tk.END, "="*22 + "\n\n")

                matched_blocks = []
                if best_line_idx != -1 and best_diff < 0.05:
                    matched_blocks.append(f" MY_SHIP (Log Truth) | Lat: {to_dmm(self.current_own_lat, True)} | Lon: {to_dmm(self.current_own_lon, False)}")
                    
                    for mmsi, coords_data in unique_ais_targets.items():
                        t_lat, t_lon, t_time = coords_data
                        fmt_time = t_time
                        if len(t_time) >= 6 and t_time.isdigit():
                            fmt_time = f"{t_time[0:2]}:{t_time[2:4]}:{t_time[4:6]}"
                            
                        matched_blocks.append(f" AIS_TGT | MMSI: {mmsi} | Time: {fmt_time} | Lat: {to_dmm(t_lat, True)} | Lon: {to_dmm(t_lon, False)}")
                        self.ais_targets_to_draw.append((t_lon, t_lat, mmsi))
                else:
                    if target_total_mins == -1:
                         self.log_text.insert(tk.END, "[!] Missing Time data. Operating in OCR-Only Mode.\n")
                    elif self.raw_log_lines:
                         self.log_text.insert(tk.END, "[!] Log sync failed. Falling back to OCR coordinates.\n")
                    
                    matched_blocks.append(f" MY_SHIP (OCR Mode) | Lat: {to_dmm(self.current_own_lat, True)} | Lon: {to_dmm(self.current_own_lon, False)}")
                
                if matched_blocks:
                    self.log_text.insert(tk.END, "\n".join(matched_blocks))
                    self.draw_ais_dots() 
                    self.refresh_sidebar_labels() 

                # --- INTELLIGENT STATUS BAR ---
                if self.current_radar_range:
                    self.coord_label.config(text=f"Scale Auto-Calibrated (Range: {self.current_radar_range} NM). Right-click to probe coordinates.", fg="#0f0")
                else:
                    self.coord_label.config(text="⚠️ SCALE UNCALIBRATED (Range OCR Failed). Defaulting to 0.5 NM. Right-click to probe.", fg="yellow")

            else:
                self.log_text.insert(tk.END, f"[!] OCR failed to find Latitude/Longitude Coordinates.\nText seen:\n{o_text_combined}\n")
                
        except Exception as e:
            self.log_text.insert(tk.END, f"[!] Error analyzing image:\n{e}")
            
        self.log_text.config(state=tk.DISABLED)

    # --- GEOMETRY MATH HELPERS ---
    def get_radar_geometry(self):
        h = self.img_height
        cx = self.MY_SHIP_X
        cy = self.MY_SHIP_Y
        
        current_range = self.current_radar_range if self.current_radar_range else 0.5
        RADAR_RADIUS_MODIFIER = 0.895 
        radar_radius_pixels = (h / 2.0) * RADAR_RADIUS_MODIFIER
        pixels_per_nm = radar_radius_pixels / current_range 
            
        RADAR_HEADING_DEG = self.current_heading if self.current_heading is not None else 0.0
        
        return cx, cy, pixels_per_nm, RADAR_HEADING_DEG

    def calculate_target_lat_lon(self, cx, cy, target_x, target_y, pixels_per_nm, heading, own_lat, own_lon):
        dx = target_x - cx
        dy = cy - target_y 
        
        dist_pixels = math.sqrt(dx**2 + dy**2)
        dist_nm = dist_pixels / pixels_per_nm
        
        if dist_nm == 0:
            return own_lat, own_lon
            
        angle_rad = math.atan2(dx, dy)
        rel_bearing = (math.degrees(angle_rad) + 360) % 360
        true_bearing = (rel_bearing + heading) % 360
        
        R = 3440.065 
        d = dist_nm / R
        lat1 = math.radians(own_lat)
        lon1 = math.radians(own_lon)
        brng = math.radians(true_bearing)
        
        lat2 = math.asin(math.sin(lat1)*math.cos(d) + math.cos(lat1)*math.sin(d)*math.cos(brng))
        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d)*math.cos(lat1), math.cos(d)-math.sin(lat1)*math.sin(lat2))
        
        return math.degrees(lat2), math.degrees(lon2)

    def draw_ais_dots(self):
        self.canvas.delete("ais_dot")
        self.canvas.delete("my_ship_dot")
        
        if not self.orig_img or self.current_own_lon is None or self.current_own_lat is None: 
            return
            
        cx, cy, pixels_per_nm, heading = self.get_radar_geometry()
        
        scaled_cx = cx * self.scale
        scaled_cy = cy * self.scale
        
        # --- DRAW MY SHIP ---
        my_ship_text = "MY SHIP"
        if self.current_own_lat is not None and self.current_own_lon is not None:
            my_ship_text += f"\n[{to_dmm(self.current_own_lat, True)}, {to_dmm(self.current_own_lon, False)}]"

        self.canvas.create_oval(scaled_cx-6, scaled_cy-6, scaled_cx+6, scaled_cy+6, fill="cyan", outline="black", width=2, tags="my_ship_dot")
        self.canvas.create_text(scaled_cx+12, scaled_cy, text=my_ship_text, fill="cyan", anchor=tk.W, font=("Arial", 10, "bold"), tags="my_ship_dot")
        
        def haversine(lon1, lat1, lon2, lat2):
            R = 3440.065 
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            
            x = math.sin(dlambda) * math.cos(phi2)
            y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
            bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
            return distance, bearing

        for ais_lon, ais_lat, mmsi in self.ais_targets_to_draw:
            dist_nm, true_bearing = haversine(self.current_own_lon, self.current_own_lat, ais_lon, ais_lat)
            
            current_range = self.current_radar_range if self.current_radar_range else 0.5
            if dist_nm > current_range * 1.5: 
                continue
            
            rel_bearing = (true_bearing - heading) % 360
            angle_rad = math.radians(rel_bearing - 90)
            dist_pixels = dist_nm * pixels_per_nm * self.scale
            
            px = scaled_cx + (dist_pixels * math.cos(angle_rad))
            py = scaled_cy + (dist_pixels * math.sin(angle_rad))
            
            r = 4
            self.canvas.create_oval(px-r, py-r, px+r, py+r, fill="red", outline="white", tags="ais_dot")
            self.canvas.create_text(px+10, py, text=f"MMSI: {mmsi}\n[{to_dmm(ais_lat, True)}, {to_dmm(ais_lon, False)}]", fill="red", anchor=tk.W, tags="ais_dot", font=("Arial", 8))

    # --- Drawing Logic ---
    def get_color_for_class(self, cat_id):
        colors = ["cyan", "yellow", "lime", "magenta", "orange", "red", "dodgerblue", "pink", "gold", "purple"]
        return colors[(cat_id - 1) % len(colors)]

    def redraw_canvas(self):
        self.canvas.delete("all")
        if not self.image_path or not self.orig_img: return

        new_w = max(1, int(self.img_width * self.scale))
        new_h = max(1, int(self.img_height * self.scale))
        resized_img = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)

        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        cx, cy, pixels_per_nm, heading = self.get_radar_geometry()

        for ann in self.annotations:
            cat_id = ann.get("category_id", 1)
            cat_name = self.categories.get(cat_id, f"C:{cat_id}")
            label_text = f"ID:{ann['id']} {cat_name}"
            
            target_x, target_y = None, None
            if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                target_x = ann["xmin"] + (ann["width"] / 2.0)
                target_y = ann["ymin"] + (ann["height"] / 2.0)
            elif ann.get("type") == "segmentation" and "segmentation" in ann:
                pts = ann["segmentation"][0]
                if len(pts) > 0:
                    xs = pts[0::2]
                    ys = pts[1::2]
                    target_x = sum(xs) / len(xs)
                    target_y = sum(ys) / len(ys)

            if target_x is not None and self.current_own_lat is not None:
                t_lat, t_lon = self.calculate_target_lat_lon(cx, cy, target_x, target_y, pixels_per_nm, heading, self.current_own_lat, self.current_own_lon)
                label_text += f"\n[{to_dmm(t_lat, True)}, {to_dmm(t_lon, False)}]"

            color = self.get_color_for_class(cat_id)

            if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                sx, sy = ann["xmin"] * self.scale, ann["ymin"] * self.scale
                ex, ey = (ann["xmin"] + ann["width"]) * self.scale, (ann["ymin"] + ann["height"]) * self.scale
                self.canvas.create_rectangle(sx, sy, ex, ey, outline=color, width=2)
                self.canvas.create_text(sx, sy-20, text=label_text, fill=color, anchor=tk.W)

            elif ann.get("type") == "segmentation" and "segmentation" in ann:
                scaled_pts = [p * self.scale for p in ann["segmentation"][0]]
                if len(scaled_pts) >= 6:
                    self.canvas.create_polygon(scaled_pts, outline=color, fill="", width=2)
                    self.canvas.create_text(scaled_pts[0], scaled_pts[1]-20, text=label_text, fill=color, anchor=tk.W)

        if self.current_polygon_points:
            cat_id = self.get_selected_cat_id(self.category_var.get())
            color = self.get_color_for_class(cat_id)
            
            scaled_pts = [p * self.scale for p in self.current_polygon_points]
            for i in range(0, len(scaled_pts), 2):
                x, y = scaled_pts[i], scaled_pts[i+1]
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=color, outline=color, tags="temp_poly")
                if i >= 2:
                    px, py = scaled_pts[i-2], scaled_pts[i-1]
                    self.canvas.create_line(px, py, x, y, fill=color, width=2, tags="temp_poly")

    def on_press(self, event):
        if not self.image_path: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        cat_id = self.get_selected_cat_id(self.category_var.get())
        color = self.get_color_for_class(cat_id)
        
        if self.mode_var.get() == "bbox":
            self.start_raw_x = raw_x
            self.start_raw_y = raw_y
            sx, sy = raw_x * self.scale, raw_y * self.scale
            self.current_rect = self.canvas.create_rectangle(sx, sy, sx, sy, outline=color, width=2)
            
        elif self.mode_var.get() == "mask":
            if len(self.current_polygon_points) >= 6:
                first_x, first_y = self.current_polygon_points[0], self.current_polygon_points[1]
                dist = ((raw_x - first_x)**2 + (raw_y - first_y)**2) ** 0.5
                if dist * self.scale <= 15:
                    # Treat clicking near the start as a "close polygon" (right-click)
                    self.on_right_click(event) 
                    return

            self.current_polygon_points.extend([raw_x, raw_y])
            self.redraw_canvas() 

    def on_drag(self, event):
        if not self.image_path or self.mode_var.get() != "bbox" or self.start_raw_x is None: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        sx, sy = self.start_raw_x * self.scale, self.start_raw_y * self.scale
        ex, ey = raw_x * self.scale, raw_y * self.scale
        self.canvas.coords(self.current_rect, sx, sy, ex, ey)

    def on_release(self, event):
        if not self.image_path or self.mode_var.get() != "bbox" or self.start_raw_x is None: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        xmin, ymin = min(self.start_raw_x, raw_x), min(self.start_raw_y, raw_y)
        xmax, ymax = max(self.start_raw_x, raw_x), max(self.start_raw_y, raw_y)
        width, height = xmax - xmin, ymax - ymin
        
        if width > 5 and height > 5:
            self.annotations.append({
                "id": self.current_id,
                "category_id": self.get_selected_cat_id(self.category_var.get()),
                "xmin": int(xmin),
                "ymin": int(ymin),
                "width": int(width),
                "height": int(height),
                "type": "bbox"
            })
            self.current_id += 1
            self.refresh_sidebar_labels()
            self.set_dirty(True) 
            
        self.start_raw_x = None
        self.start_raw_y = None
        self.redraw_canvas()
        self.draw_ais_dots() 

    def on_right_click(self, event=None):
        if not event or not hasattr(event, 'num') or event.num not in (2, 3) or not self.image_path:
            return

        raw_x, raw_y = self.get_raw_coords(event)
        px, py = raw_x * self.scale, raw_y * self.scale

        # --- MASK COMPLETION LOGIC ---
        if self.mode_var.get() == "mask" and len(self.current_polygon_points) >= 6:
            self.annotations.append({
                "id": self.current_id,
                "category_id": self.get_selected_cat_id(self.category_var.get()),
                "segmentation": [self.current_polygon_points],
                "type": "segmentation"
            })
            self.current_id += 1
            self.current_polygon_points = []
            self.refresh_sidebar_labels()
            self.redraw_canvas()
            self.draw_ais_dots() 
            self.set_dirty(True) 

        # --- INSTANT PROBE LOGIC ---
        self.canvas.delete("temp_coord")
        pixel_str = f"Pixel: [{int(raw_x)}, {int(raw_y)}]"

        if self.current_own_lat is not None and self.current_own_lon is not None:
            cx, cy, pixels_per_nm, heading = self.get_radar_geometry()
            t_lat, t_lon = self.calculate_target_lat_lon(cx, cy, raw_x, raw_y, pixels_per_nm, heading, self.current_own_lat, self.current_own_lon)
            
            lat_str = to_dmm(t_lat, True)
            lon_str = to_dmm(t_lon, False)
            
            full_info = f"{pixel_str} | Geo: {lat_str}, {lon_str}"
            self.coord_label.config(text=f"Probed: {full_info}", fg="#0f0")
            display_text = f"{pixel_str}\n{lat_str}\n{lon_str}"
        else:
            self.coord_label.config(text=f"Probed: {pixel_str} (No GPS Sync)", fg="#0f0")
            display_text = pixel_str
        
        self.canvas.create_oval(px-4, py-4, px+4, py+4, fill="yellow", outline="black", width=2, tags="temp_coord")
        self.canvas.create_text(px+2, py-23, text=display_text, fill="black", tags="temp_coord", font=("Arial", 10, "bold"), justify=tk.CENTER)
        self.canvas.create_text(px, py-25, text=display_text, fill="yellow", tags="temp_coord", font=("Arial", 10, "bold"), justify=tk.CENTER)

            
    def clear_last(self, event=None):
        if self.current_polygon_points:
            self.current_polygon_points = []
            self.redraw_canvas()
            self.draw_ais_dots()
        elif self.annotations:
            self.annotations.pop()
            self._update_current_id()
            self.refresh_sidebar_labels()
            self.redraw_canvas()
            self.draw_ais_dots()
            self.set_dirty(True)

    def delete_annotation(self, ann_id):
        self.annotations = [a for a in self.annotations if a["id"] != ann_id]
        self._update_current_id()
        self.refresh_sidebar_labels()
        self.redraw_canvas()
        self.draw_ais_dots()
        self.set_dirty(True)

    def _update_current_id(self):
        if self.annotations:
            self.current_id = max([ann["id"] for ann in self.annotations]) + 1
        else:
            self.current_id = 1

    def on_sidebar_class_change(self, event, ann_id, var):
        new_cat_id = self.get_selected_cat_id(var.get())
        for ann in self.annotations:
            if ann["id"] == ann_id:
                if ann["category_id"] != new_cat_id:
                    ann["category_id"] = new_cat_id
                    self.set_dirty(True)
                    self.redraw_canvas()
                    self.draw_ais_dots()
                    self.refresh_sidebar_labels()
                break

    def refresh_sidebar_labels(self):
        for widget in self.labels_inner_frame.winfo_children():
            widget.destroy()
            
        cat_list = [f"{k} - {v}" for k, v in self.categories.items()] if self.categories else ["1 - Default"]
        cx, cy, pixels_per_nm, heading = self.get_radar_geometry()
            
        for ann in self.annotations:
            row = tk.Frame(self.labels_inner_frame, bg="#f0f0f0", pady=2)
            row.pack(fill=tk.X, padx=5)
            
            cat_id = ann.get("category_id", 1)
            cat_name = self.categories.get(cat_id, "Default")
            t = "Box" if ann.get("type") == "bbox" else "Mask"
            
            coord_str = ""
            if self.current_own_lat is not None and self.current_own_lon is not None:
                target_x, target_y = None, None
                if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                    target_x = ann["xmin"] + (ann["width"] / 2.0)
                    target_y = ann["ymin"] + (ann["height"] / 2.0)
                elif ann.get("type") == "segmentation" and "segmentation" in ann:
                    pts = ann["segmentation"][0]
                    if len(pts) > 0:
                        xs = pts[0::2]
                        ys = pts[1::2]
                        target_x = sum(xs) / len(xs)
                        target_y = sum(ys) / len(ys)
                
                if target_x is not None:
                    t_lat, t_lon = self.calculate_target_lat_lon(cx, cy, target_x, target_y, pixels_per_nm, heading, self.current_own_lat, self.current_own_lon)
                    coord_str = f"\n[{to_dmm(t_lat, True)}, {to_dmm(t_lon, False)}]"

            lbl_text = f"{t} {ann['id']}:{coord_str}"
            lbl = tk.Label(row, text=lbl_text, fg="black", bg="#f0f0f0", anchor=tk.W, justify=tk.LEFT)
            lbl.pack(side=tk.LEFT)
            
            c_var = tk.StringVar(value=f"{cat_id} - {cat_name}")
            combo = ttk.Combobox(row, textvariable=c_var, values=cat_list, state="readonly", width=11)
            combo.pack(side=tk.LEFT, padx=2)
            
            combo.bind("<<ComboboxSelected>>", lambda e, aid=ann['id'], cv=c_var: self.on_sidebar_class_change(e, aid, cv))
            
            btn = tk.Button(row, text=" X ", fg="red", highlightbackground="#ffcccc", font=("Arial", 10, "bold"), 
                            command=lambda id=ann['id']: self.delete_annotation(id))
            btn.pack(side=tk.RIGHT)

    def update_legend_and_dropdowns(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
            
        if not self.categories:
            tk.Label(self.legend_frame, text="No categories.txt found", fg="black", bg="#f0f0f0").pack(anchor=tk.W)
            cat_list = ["1 - Default"]
        else:
            cat_list = []
            for cid, name in self.categories.items():
                tk.Label(self.legend_frame, text=f"C:{cid} : {name}", fg="black", bg="#f0f0f0").pack(anchor=tk.W)
                cat_list.append(f"{cid} - {name}")
                
        self.category_combo['values'] = cat_list
        if cat_list:
            self.category_combo.current(0)

    # --- Zoom Logic ---
    def zoom_in(self, event=None):
        self.scale *= 1.1
        self.redraw_canvas()
        self.draw_ais_dots() 

    def zoom_out(self, event=None):
        self.scale /= 1.1
        self.redraw_canvas()
        self.draw_ais_dots()

    def zoom_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def get_raw_coords(self, event):
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        return x, y

    def on_image_select(self, event=None):
        if not self.check_unsaved():
            if self.image_list and self.current_idx != -1:
                self.image_combo.current(self.current_idx)
            return
            
        selected_idx = self.image_combo.current()
        if selected_idx != self.current_idx and selected_idx != -1:
            self.current_idx = selected_idx
            self.load_image_by_index()

    def load_image_by_index(self):
        if not (0 <= self.current_idx < len(self.image_list)): return
        
        # Sync the dropdown menu value with the next/prev buttons
        if self.image_combo.current() != self.current_idx:
            self.image_combo.current(self.current_idx)
        
        self.image_path = self.image_list[self.current_idx]
        self.orig_img = Image.open(self.image_path)
        self.img_width, self.img_height = self.orig_img.size
        
        self.annotations = []
        self.current_id = 1
        self.current_polygon_points = []
        self.current_own_lat = None
        self.current_own_lon = None
        self.current_heading = None
        self.current_cursor_lat = None
        self.current_cursor_lon = None
        self.current_radar_range = None
        
        filename = os.path.basename(self.image_path)
        self.root.title(f"Radar Labeler | [{self.current_idx + 1}/{len(self.image_list)}] - {filename}")
        
        self.load_existing_json()
        self.redraw_canvas()
        self.refresh_sidebar_labels()
        
        self.sync_logs_from_image()
        
        self.set_dirty(False)

    def load_existing_json(self):
        json_path = os.path.splitext(self.image_path)[0] + ".json"
        if not os.path.exists(json_path): return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            max_id = 0
            for ann in data.get("annotations", []):
                ann_id = ann.get("id", 1)
                if ann_id > max_id:
                    max_id = ann_id
                self.annotations.append(ann)
                
            self.current_id = max_id + 1
        except Exception as e:
            print(f"Error loading JSON: {e}")

    # --- Navigation & Saving ---
    def next_image(self, event=None):
        if not self.check_unsaved(): return
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_image_by_index()

    def prev_image(self, event=None):
        if not self.check_unsaved(): return
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image_by_index()

    def save_json(self, event=None):
        if not self.image_path: return
            
        output_data = {
            "type": "mixed_annotations",
            "copyright": "seadronix corp.",
            "date": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "verification": False,
            "image": {
                "file_name": os.path.basename(self.image_path),
                "width": self.img_width,
                "height": self.img_height
            },
            "annotations": self.annotations 
        }
        
        try:
            save_path = os.path.splitext(self.image_path)[0] + ".json"
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            self.set_dirty(False) 
            messagebox.showinfo("Save Successful", f"Saved to:\n{os.path.basename(save_path)}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.state('zoomed') 
    except tk.TclError:
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.geometry(f"{width}x{height}")
        
    app = RadarLabeler(root)
    root.mainloop()
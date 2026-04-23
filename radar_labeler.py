import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
from datetime import datetime

class RadarLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Dataset Labeler")
        
        # Image State
        self.image_list = []
        self.current_idx = -1
        self.image_path = None
        self.orig_img = None  # Store original PIL Image for clean scaling
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        self.scale = 1.0      # Zoom scale factor
        
        # Annotation & Class State
        self.annotations = []
        self.current_id = 1
        self.categories = {}  # {1: "Car", 2: "Pedestrian", ...}
        
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
        
        # Added fg="black" to control labels
        tk.Label(control_frame, text="  Category ID:").pack(side=tk.LEFT, padx=5)
        self.class_id_var = tk.IntVar(value=1)
        tk.Entry(control_frame, textvariable=self.class_id_var, width=5).pack(side=tk.LEFT)
        
        tk.Label(control_frame, text="  Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="bbox")
        tk.Radiobutton(control_frame, text="BBox", variable=self.mode_var, value="bbox").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="Mask", variable=self.mode_var, value="mask").pack(side=tk.LEFT)
        
        tk.Button(control_frame, text="Clear Last (L)", command=self.clear_last).pack(side=tk.RIGHT, padx=5)

        # --- Main Layout (Canvas + Sidebar) ---
        self.main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left side: Canvas
        canvas_container = tk.Frame(self.main_pane)
        self.main_pane.add(canvas_container, weight=5) # Increased weight so canvas takes more space
        
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

        # Right side: Sidebar (Legend + Labels)
        # Reduced initial width from 250 to 150
        self.sidebar = tk.Frame(self.main_pane, bg="#f0f0f0", width=150)
        self.main_pane.add(self.sidebar, weight=1)

        # Legend section (Added fg="black")
        tk.Label(self.sidebar, text="Categories Legend", bg="#ddd", fg="black", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(0, 5))
        self.legend_frame = tk.Frame(self.sidebar, bg="#f0f0f0")
        self.legend_frame.pack(fill=tk.X, padx=5, pady=5)

        # Labels List Section (Added fg="black")
        tk.Label(self.sidebar, text="Current Annotations", bg="#ddd", fg="black", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=(10, 5))
        self.create_scrollable_sidebar()

        # --- Mouse Bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        
        # Zoom bindings
        self.canvas.bind("<MouseWheel>", self.zoom_mousewheel) # Windows/Mac
        self.canvas.bind("<Button-4>", self.zoom_in)           # Linux up
        self.canvas.bind("<Button-5>", self.zoom_out)          # Linux down

    def create_scrollable_sidebar(self):
        # A scrollable frame for annotations
        self.labels_canvas = tk.Canvas(self.sidebar, bg="#f0f0f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.sidebar, orient="vertical", command=self.labels_canvas.yview)
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

    def load_folder(self):
        # Added initialdir=os.getcwd() to open in the local code folder
        folder_path = filedialog.askdirectory(
            title="Select Folder with Images", 
            initialdir=os.getcwd()
        )
        if not folder_path: return
        
        # --- UPDATED: Load Categories Legend from 'name = id' format ---
        self.categories.clear()
        cat_file = os.path.join(folder_path, "categories.txt")
        if os.path.exists(cat_file):
            with open(cat_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines or lines without an equals sign
                    if not line or '=' not in line: 
                        continue
                    
                    # Split the string at '=' and remove extra spaces
                    parts = line.split('=')
                    cat_name = parts[0].strip()
                    
                    try:
                        cat_id = int(parts[1].strip())
                        self.categories[cat_id] = cat_name
                    except ValueError:
                        print(f"Skipping invalid category line: {line}")
                        
        self.update_legend_ui()

        valid_exts = {".png", ".jpg", ".jpeg"}
        self.image_list = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
        if not self.image_list:
            messagebox.showwarning("No Images", "No valid images found.")
            return
            
        self.current_idx = 0
        self.load_image_by_index()

    def update_legend_ui(self):
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
            
        if not self.categories:
            # Added fg="black" for dark mode compatibility
            tk.Label(self.legend_frame, text="No categories.txt found", fg="black", bg="#f0f0f0").pack(anchor=tk.W)
            return
            
        for cid, name in self.categories.items():
            # CHANGED "ID" to "C:" and added fg="black"
            tk.Label(self.legend_frame, text=f"C:{cid} : {name}", fg="black", bg="#f0f0f0").pack(anchor=tk.W)

    def load_image_by_index(self):
        if not (0 <= self.current_idx < len(self.image_list)): return
        
        self.image_path = self.image_list[self.current_idx]
        self.orig_img = Image.open(self.image_path)
        self.img_width, self.img_height = self.orig_img.size
        
        self.annotations = []
        self.current_id = 1
        self.current_polygon_points = []
        
        filename = os.path.basename(self.image_path)
        self.root.title(f"Radar Labeler | [{self.current_idx + 1}/{len(self.image_list)}] - {filename}")
        
        self.load_existing_json()
        self.redraw_canvas()
        self.refresh_sidebar_labels()

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
                    
                # We store only the raw unscaled logic in self.annotations
                self.annotations.append(ann)
                
            self.current_id = max_id + 1
            print(f"Loaded existing annotations from {os.path.basename(json_path)}")
        except Exception as e:
            print(f"Error loading JSON: {e}")

    # --- Zoom Logic ---
    def zoom_in(self, event=None):
        self.scale *= 1.1
        self.redraw_canvas()

    def zoom_out(self, event=None):
        self.scale /= 1.1
        self.redraw_canvas()

    def zoom_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def get_raw_coords(self, event):
        # Convert window coordinate to raw image coordinate using scale
        x = self.canvas.canvasx(event.x) / self.scale
        y = self.canvas.canvasy(event.y) / self.scale
        return x, y

    # --- Drawing Logic ---
    def redraw_canvas(self):
        self.canvas.delete("all")
        if not self.image_path or not self.orig_img: return

        # 1. Redraw scaled image
        new_w = max(1, int(self.img_width * self.scale))
        new_h = max(1, int(self.img_height * self.scale))
        resized_img = self.orig_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)

        self.canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 2. Redraw scaled annotations
        for ann in self.annotations:
            cat_id = ann.get("category_id", 1)
            cat_name = self.categories.get(cat_id, f"C:{cat_id}")
            label_text = f"ID:{ann['id']} {cat_name}"

            if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                sx, sy = ann["xmin"] * self.scale, ann["ymin"] * self.scale
                ex, ey = (ann["xmin"] + ann["width"]) * self.scale, (ann["ymin"] + ann["height"]) * self.scale
                self.canvas.create_rectangle(sx, sy, ex, ey, outline="cyan", width=2)
                self.canvas.create_text(sx, sy-10, text=label_text, fill="cyan", anchor=tk.W)

            elif ann.get("type") == "segmentation" and "segmentation" in ann:
                scaled_pts = [p * self.scale for p in ann["segmentation"][0]]
                if len(scaled_pts) >= 6:
                    self.canvas.create_polygon(scaled_pts, outline="yellow", fill="", width=2)
                    self.canvas.create_text(scaled_pts[0], scaled_pts[1]-10, text=f"{label_text} (Mask)", fill="yellow", anchor=tk.W)

        # 3. Redraw temp polygon (if currently drawing)
        if self.current_polygon_points:
            scaled_pts = [p * self.scale for p in self.current_polygon_points]
            for i in range(0, len(scaled_pts), 2):
                x, y = scaled_pts[i], scaled_pts[i+1]
                self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="yellow", tags="temp_poly")
                if i >= 2:
                    px, py = scaled_pts[i-2], scaled_pts[i-1]
                    self.canvas.create_line(px, py, x, y, fill="yellow", width=2, tags="temp_poly")

    def on_press(self, event):
        if not self.image_path: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        if self.mode_var.get() == "bbox":
            self.start_raw_x = raw_x
            self.start_raw_y = raw_y
            sx, sy = raw_x * self.scale, raw_y * self.scale
            self.current_rect = self.canvas.create_rectangle(sx, sy, sx, sy, outline="cyan", width=2)
            
        elif self.mode_var.get() == "mask":
            if len(self.current_polygon_points) >= 6:
                first_x, first_y = self.current_polygon_points[0], self.current_polygon_points[1]
                dist = ((raw_x - first_x)**2 + (raw_y - first_y)**2) ** 0.5
                if dist * self.scale <= 15: # Auto-close if clicked near start (scaled tolerance)
                    self.on_right_click(None)
                    return

            self.current_polygon_points.extend([raw_x, raw_y])
            self.redraw_canvas() # redraws the temp polygon cleanly

    def on_drag(self, event):
        if not self.image_path or self.mode_var.get() != "bbox" or self.start_raw_x is None: return
        raw_x, raw_y = self.get_raw_coords(event)
        
        # Update rect on screen with scaled coords
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
                "category_id": self.class_id_var.get(),
                "xmin": int(xmin),
                "ymin": int(ymin),
                "width": int(width),
                "height": int(height),
                "type": "bbox"
            })
            self.current_id += 1
            self.refresh_sidebar_labels()
            
        self.start_raw_x = None
        self.start_raw_y = None
        self.redraw_canvas()

    def on_right_click(self, event=None):
        if self.mode_var.get() == "mask" and len(self.current_polygon_points) >= 6:
            self.annotations.append({
                "id": self.current_id,
                "category_id": self.class_id_var.get(),
                "segmentation": [self.current_polygon_points],
                "type": "segmentation"
            })
            self.current_id += 1
            self.current_polygon_points = []
            self.refresh_sidebar_labels()
            self.redraw_canvas()
            
    def clear_last(self, event=None):
        if self.current_polygon_points:
            self.current_polygon_points = []
        elif self.annotations:
            self.annotations.pop()
            
        self._update_current_id()
        self.refresh_sidebar_labels()
        self.redraw_canvas()

    def delete_annotation(self, ann_id):
        self.annotations = [a for a in self.annotations if a["id"] != ann_id]
        self._update_current_id()
        self.refresh_sidebar_labels()
        self.redraw_canvas()

    def _update_current_id(self):
        if self.annotations:
            self.current_id = max([ann["id"] for ann in self.annotations]) + 1
        else:
            self.current_id = 1

    def refresh_sidebar_labels(self):
        for widget in self.labels_inner_frame.winfo_children():
            widget.destroy()
            
        for ann in self.annotations:
            row = tk.Frame(self.labels_inner_frame, bg="#f0f0f0", pady=2)
            row.pack(fill=tk.X, padx=5)
            
            cat_id = ann.get("category_id", 1)
            cat_name = self.categories.get(cat_id, "")
            disp_name = f" [{cat_name}]" if cat_name else ""
            t = "Box" if ann.get("type") == "bbox" else "Mask"
            
            # Added fg="black" so the text doesn't turn white in macOS dark mode
            lbl = tk.Label(row, text=f"{t} ID:{ann['id']} C:{cat_id}{disp_name}", fg="black", bg="#f0f0f0", anchor=tk.W)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            btn = tk.Button(row, text=" X ", fg="red", highlightbackground="#ffcccc", font=("Arial", 10, "bold"), 
                            command=lambda id=ann['id']: self.delete_annotation(id))
            btn.pack(side=tk.RIGHT)

    # --- Navigation & Saving ---
    def next_image(self, event=None):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_image_by_index()

    def prev_image(self, event=None):
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
            "annotations": self.annotations # Stored directly since canvas_ids were removed
        }
        
        try:
            save_path = os.path.splitext(self.image_path)[0] + ".json"
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            messagebox.showinfo("Save Successful", f"Saved to:\n{os.path.basename(save_path)}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Attempt to maximize the window cross-platform
    try:
        root.state('zoomed') # Works on Windows and most Linux environments
    except tk.TclError:
        # Fallback for macOS
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.geometry(f"{width}x{height}")
        
    app = RadarLabeler(root)
    root.mainloop()
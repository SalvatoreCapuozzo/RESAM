import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
from datetime import datetime

class RadarLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Dataset Labeler")
        
        # Folder & Image State
        self.image_list = []
        self.current_idx = -1
        self.image_path = None
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        
        # Annotation State
        self.annotations = []
        self.current_id = 1
        
        # Drawing State
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.current_polygon_points = []
        
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # File operations
        tk.Button(control_frame, text="Load Folder", command=self.load_folder, bg="lightblue").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Save JSON (S)", command=self.save_json, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        
        # Navigation
        tk.Button(control_frame, text="<< Prev (A)", command=self.prev_image).pack(side=tk.LEFT, padx=10)
        tk.Button(control_frame, text="Next (D) >>", command=self.next_image).pack(side=tk.LEFT, padx=0)
        
        # Class Selection
        tk.Label(control_frame, text="  Category ID:").pack(side=tk.LEFT, padx=5)
        self.class_id_var = tk.IntVar(value=1)
        tk.Entry(control_frame, textvariable=self.class_id_var, width=5).pack(side=tk.LEFT)
        
        # Mode Selection
        tk.Label(control_frame, text="  Mode:").pack(side=tk.LEFT, padx=5)
        self.mode_var = tk.StringVar(value="bbox")
        tk.Radiobutton(control_frame, text="BBox", variable=self.mode_var, value="bbox").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="Mask", variable=self.mode_var, value="mask").pack(side=tk.LEFT)
        
        # Utilities
        tk.Button(control_frame, text="Clear Last (L)", command=self.clear_last, bg="lightcoral").pack(side=tk.RIGHT, padx=5)
        
        # Scrollable Canvas Setup
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(canvas_frame, cursor="crosshair", 
                                xscrollcommand=self.h_scroll.set, 
                                yscrollcommand=self.v_scroll.set, bg="#333333")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        # Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click) # Right click to close polygon

    def setup_shortcuts(self):
        self.root.bind("<Return>", self.on_right_click) # Enter to close polygon
        self.root.bind("<a>", self.prev_image)
        self.root.bind("<A>", self.prev_image)
        self.root.bind("<d>", self.next_image)
        self.root.bind("<D>", self.next_image)
        self.root.bind("<s>", self.save_json)
        self.root.bind("<S>", self.save_json)
        self.root.bind("<l>", self.clear_last)
        self.root.bind("<L>", self.clear_last)

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path: return
        
        valid_exts = {".png", ".jpg", ".jpeg"}
        self.image_list = sorted([
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        
        if not self.image_list:
            messagebox.showwarning("No Images", "No valid images found in the selected folder.")
            return
            
        self.current_idx = 0
        self.load_image_by_index()

    def load_image_by_index(self):
        if not (0 <= self.current_idx < len(self.image_list)): return
        
        self.image_path = self.image_list[self.current_idx]
        img = Image.open(self.image_path)
        self.img_width, self.img_height = img.size
        self.tk_image = ImageTk.PhotoImage(img)
        
        self.canvas.config(scrollregion=(0, 0, self.img_width, self.img_height))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self.annotations = []
        self.current_id = 1
        self.current_polygon_points = []
        
        filename = os.path.basename(self.image_path)
        self.root.title(f"Radar Labeler | [{self.current_idx + 1} of {len(self.image_list)}] - {filename}")
        
        # Attempt to load existing JSON for this image
        self.load_existing_json()

    def load_existing_json(self):
        json_path = os.path.splitext(self.image_path)[0] + ".json"
        if not os.path.exists(json_path): return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            max_id = 0
            for ann in data.get("annotations", []):
                cat_id = ann.get("category_id", 1)
                ann_id = ann.get("id", 1)
                if ann_id > max_id:
                    max_id = ann_id
                
                # Check if it's a bounding box
                if ann.get("type", "bbox") == "bbox" and "xmin" in ann:
                    xmin, ymin = ann["xmin"], ann["ymin"]
                    width, height = ann["width"], ann["height"]
                    xmax, ymax = xmin + width, ymin + height
                    
                    rect = self.canvas.create_rectangle(xmin, ymin, xmax, ymax, outline="cyan", width=2)
                    text_id = self.canvas.create_text(xmin, ymin-10, text=f"ID:{ann_id} C:{cat_id}", fill="cyan", anchor=tk.W)
                    
                    self.annotations.append({
                        "id": ann_id,
                        "category_id": cat_id,
                        "xmin": xmin,
                        "ymin": ymin,
                        "width": width,
                        "height": height,
                        "type": "bbox",
                        "canvas_id": rect,
                        "text_id": text_id
                    })
                    
                # Check if it's a segmentation mask
                elif ann.get("type") == "segmentation" and "segmentation" in ann:
                    points = ann["segmentation"][0]
                    poly = self.canvas.create_polygon(points, outline="yellow", fill="", width=2)
                    x0, y0 = points[0], points[1]
                    text_id = self.canvas.create_text(x0, y0-10, text=f"ID:{ann_id} C:{cat_id} (Mask)", fill="yellow", anchor=tk.W)
                    
                    self.annotations.append({
                        "id": ann_id,
                        "category_id": cat_id,
                        "segmentation": [points],
                        "type": "segmentation",
                        "canvas_id": poly,
                        "text_id": text_id
                    })
            
            # Ensure the next drawn item gets a fresh ID
            self.current_id = max_id + 1
            print(f"Loaded existing annotations from {os.path.basename(json_path)}")
            
        except Exception as e:
            print(f"Error loading JSON for {os.path.basename(self.image_path)}: {e}")

    def next_image(self, event=None):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_image_by_index()

    def prev_image(self, event=None):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image_by_index()

    def get_canvas_coords(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return x, y

    def on_press(self, event):
        if not self.image_path: return
        x, y = self.get_canvas_coords(event)
        
        if self.mode_var.get() == "bbox":
            self.start_x = x
            self.start_y = y
            self.current_rect = self.canvas.create_rectangle(x, y, x, y, outline="cyan", width=2)
            
        elif self.mode_var.get() == "mask":
            # Auto-close logic: if click is near start point
            if len(self.current_polygon_points) >= 6:
                first_x, first_y = self.current_polygon_points[0], self.current_polygon_points[1]
                dist = ((x - first_x)**2 + (y - first_y)**2) ** 0.5
                if dist <= 10:
                    self.on_right_click(None)
                    return

            self.current_polygon_points.extend([x, y])
            r = 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="yellow", tags="temp_poly")
            if len(self.current_polygon_points) >= 4:
                x1, y1 = self.current_polygon_points[-4], self.current_polygon_points[-3]
                self.canvas.create_line(x1, y1, x, y, fill="yellow", width=2, tags="temp_poly")

    def on_drag(self, event):
        if not self.image_path or self.mode_var.get() != "bbox" or not self.start_x: return
        x, y = self.get_canvas_coords(event)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, x, y)

    def on_release(self, event):
        if not self.image_path or self.mode_var.get() != "bbox" or not self.start_x: return
        x, y = self.get_canvas_coords(event)
        
        xmin, ymin = min(self.start_x, x), min(self.start_y, y)
        xmax, ymax = max(self.start_x, x), max(self.start_y, y)
        width, height = xmax - xmin, ymax - ymin
        
        if width > 5 and height > 5:
            cat_id = self.class_id_var.get()
            text_id = self.canvas.create_text(xmin, ymin-10, text=f"ID:{self.current_id} C:{cat_id}", fill="cyan", anchor=tk.W)
            
            self.annotations.append({
                "id": self.current_id,
                "category_id": cat_id,
                "xmin": int(xmin),
                "ymin": int(ymin),
                "width": int(width),
                "height": int(height),
                "type": "bbox",
                "canvas_id": self.current_rect,
                "text_id": text_id
            })
            self.current_id += 1
        else:
            self.canvas.delete(self.current_rect)
            
        self.start_x = None
        self.start_y = None

    def on_right_click(self, event=None):
        if self.mode_var.get() == "mask" and len(self.current_polygon_points) >= 6:
            x1, y1 = self.current_polygon_points[-2], self.current_polygon_points[-1]
            x0, y0 = self.current_polygon_points[0], self.current_polygon_points[1]
            self.canvas.create_line(x1, y1, x0, y0, fill="yellow", width=2, tags="temp_poly")
            
            poly = self.canvas.create_polygon(self.current_polygon_points, outline="yellow", fill="", width=2)
            self.canvas.delete("temp_poly")
            
            cat_id = self.class_id_var.get()
            text_id = self.canvas.create_text(x0, y0-10, text=f"ID:{self.current_id} C:{cat_id} (Mask)", fill="yellow", anchor=tk.W)
            
            self.annotations.append({
                "id": self.current_id,
                "category_id": cat_id,
                "segmentation": [self.current_polygon_points],
                "type": "segmentation",
                "canvas_id": poly,
                "text_id": text_id
            })
            
            self.current_id += 1
            self.current_polygon_points = []
            
    def clear_last(self, event=None):
        if self.annotations:
            last_ann = self.annotations.pop()
            self.canvas.delete(last_ann["canvas_id"]) # Delete shape
            self.canvas.delete(last_ann["text_id"])   # Delete text label
            
            # Recalculate current_id based on remaining annotations
            if self.annotations:
                self.current_id = max([ann["id"] for ann in self.annotations]) + 1
            else:
                self.current_id = 1
                
        elif self.current_polygon_points:
            self.canvas.delete("temp_poly")
            self.current_polygon_points = []

    def save_json(self, event=None):
        if not self.image_path: return
        
        clean_annotations = []
        for ann in self.annotations:
            clean_ann = ann.copy()
            clean_ann.pop("canvas_id", None)
            clean_ann.pop("text_id", None)
            clean_annotations.append(clean_ann)
            
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
            "annotations": clean_annotations
        }
        
        try:
            save_path = os.path.splitext(self.image_path)[0] + ".json"
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=4)
                
            # --- NEW: Clear Popup Confirmation ---
            messagebox.showinfo(
                "Save Successful", 
                f"The JSON file was properly saved!\n\nFile: {os.path.basename(save_path)}"
            )
            
            # We keep the green visual flash on the canvas as an extra visual cue
            save_msg = self.canvas.create_text(self.img_width/2, 40, text="SAVED!", fill="green", font=("Arial", 24, "bold"))
            self.root.after(1000, lambda: self.canvas.delete(save_msg))
            
        except Exception as e:
            # Added error handling just in case of folder permission issues
            messagebox.showerror("Save Error", f"An error occurred while saving the JSON:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = RadarLabeler(root)
    root.mainloop()
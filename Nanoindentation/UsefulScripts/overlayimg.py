from tkinter import Tk, Canvas, Button, Frame, NW
from PIL import Image, ImageTk

class RectangleSelector:
    def __init__(self, root, image):
        self.root = root
        self.image = image
        self.frame = Frame(root)
        self.frame.pack()

        self.canvas = Canvas(self.frame, width=image.width, height=image.height)
        self.canvas.pack(side="left")
        self.photo_img = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo_img, anchor=NW)
        self.rectangle = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        # Create buttons
        self.accept_button = Button(self.frame, text="Accept", command=self.accept_selection)
        self.accept_button.pack(side="left")
        self.clear_button = Button(self.frame, text="Clear", command=self.clear_selection)
        self.clear_button.pack(side="left")

        # Register mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def draw_rectangle(self):
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        self.rectangle = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.end_x, self.end_y, outline="red"
        )

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.end_x = event.x
        self.end_y = event.y
        self.draw_rectangle()

    def on_mouse_move(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.draw_rectangle()

    def on_mouse_up(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.draw_rectangle()

    def accept_selection(self):
        if self.rectangle:
            x = min(self.start_x, self.end_x)
            y = min(self.start_y, self.end_y)
            width = abs(self.end_x - self.start_x)
            height = abs(self.end_y - self.start_y)

            # Resize the heatmap image to fit within the selected area
            heatmap_img_resized = heatmap_img.resize((width, height))

            # Create a new image for the overlay
            overlay_img = self.image.copy()

            # Apply transparency to the heatmap image
            heatmap_img_transparent = heatmap_img_resized.copy()
            heatmap_img_transparent.putalpha(128)  # Adjust the alpha value (0-255) for desired transparency

            # Paste the transparent heatmap image onto the overlay image
            overlay_img.paste(heatmap_img_transparent, (x, y))

            # Save the final overlay image
            overlay_img.save("Overlay.png")

            self.clear_selection()
            self.root.destroy()

    def clear_selection(self):
        if self.rectangle:
            self.canvas.delete(self.rectangle)
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None 

# Load the images
heatmap_img = Image.open("HeatMap.png")
microscope_img = Image.open("MicroImg.png")

# Create a Tkinter window
root = Tk()
root.title("Select Area")

# Create the rectangle selector
selector = RectangleSelector(root, microscope_img)

# Start the Tkinter event loop
root.mainloop()

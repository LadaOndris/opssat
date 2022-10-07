from tkinter import ALL, BOTH, Canvas, Frame, NW, Text, Tk

from PIL import Image, ImageTk


class Example(Frame):

    def __init__(self, root):
        super().__init__()
        self.root = root

        self.image_x_start = 0
        self.image_y_start = 0

        self.right_panel_width = 200
        self.bottom_panel_height = 100

        self.minitile_size = 40
        self.active_minitile = (0, 0)

        self.initUI()

    def get_zoom_function(self, canvas):
        def do_zoom(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            factor = 1.001 ** event.delta
            canvas.scale(ALL, x, y, factor, factor)
            print("zooming")

        return do_zoom

    def get_on_canvas_mouse_click(self, canvas):
        def on_canvas_mouse_click(event):
            canvas.scan_mark(event.x, event.y)
            print(f"clicked at {event.x}, {event.y}")

            self.update_active_minitile(event.x, event.y)
            self.highlight_active_minitile(canvas)

        return on_canvas_mouse_click

    def update_active_minitile(self, x, y):
        row = self.pixel_to_index(x)
        col = self.pixel_to_index(y)
        self.active_minitile = (row, col)

    def index_to_pixel(self, index) -> int:
        return index * self.minitile_size

    def pixel_to_index(self, pixel) -> int:
        return int(pixel // self.minitile_size)

    def initUI(self):
        self.master.title("Image annotation tool")
        self.pack(fill=BOTH, expand=1)

        self.img = Image.open("datasets/opssat/raw/001.png")
        self.photo_image = ImageTk.PhotoImage(self.img)

        canvas = Canvas(self, width=self.img.size[0] + self.right_panel_width,
                        height=self.img.size[1] + self.bottom_panel_height)
        canvas.bind("<MouseWheel>", self.get_zoom_function(canvas))
        canvas.bind('<ButtonPress-1>', self.get_on_canvas_mouse_click(canvas))
        canvas.bind("<B1-Motion>", lambda event: canvas.scan_dragto(event.x, event.y, gain=1))

        self.draw_image(canvas)
        self.fill_image_with_minitiles(canvas)
        self.highlight_active_minitile(canvas)
        canvas.pack(fill=BOTH, expand=1)

        self.draw_press_space_to_annotate_text(self.root)

    def draw_image(self, canvas: Canvas):
        canvas.create_image(self.image_x_start, self.image_y_start, anchor=NW, image=self.photo_image)

    def fill_image_with_minitiles(self, canvas):
        for row in range(self.image_y_start, self.image_y_start + self.img.size[1], self.minitile_size):
            for col in range(self.image_x_start, self.image_x_start + self.img.size[0], self.minitile_size):
                canvas.create_rectangle(col, row, self.minitile_size, self.minitile_size, outline="#6e2a2a")

    def highlight_active_minitile(self, canvas):
        col = self.index_to_pixel(self.active_minitile[0])
        row = self.index_to_pixel(self.active_minitile[1])
        canvas.create_rectangle(col, row, col + self.minitile_size, row + self.minitile_size, outline="#bf1b1b", width=3)

    def draw_press_space_to_annotate_text(self, parent):
        y = self.img.size[1] + 50
        x = 50
        text = "Press space to annotate a minitile."
        text_element = Text(parent, width=x, height=y)
        text_element.insert(1.0, text)


def main():
    root = Tk()
    # root.attributes('-zoomed', True)
    ex = Example(root)
    root.mainloop()


if __name__ == '__main__':
    main()

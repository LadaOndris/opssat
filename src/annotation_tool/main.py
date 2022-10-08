from tkinter import ALL, BOTH, Canvas, Frame, NW, Text, Tk

from PIL import Image, ImageTk

from src.annotation_tool.core import index_to_pixel, pixel_to_index, StateStorage


class UI(Frame):

    def __init__(self, root, img, num_classes, minitile_size: int, state_storage: StateStorage):
        super().__init__()
        self.root = root
        self.img = img
        self.num_classes = num_classes
        self.state_storage = state_storage

        self.image_x_start = 0
        self.image_y_start = 0

        self.right_panel_width = 200
        self.bottom_panel_height = 100

        self.minitile_size = minitile_size
        self.active_minitile = (0, 0)
        self.active_type = 0

        self.annotation_tag = 'annotated_minitiles'

        self.drag_x = 0
        self.drag_y = 0

        self.types = ['Agricultural', 'Cloud', 'Mountain', 'Natural',
                      'River', 'Sea ice', 'Snow', 'Water']

        self.initUI()

    def initUI(self):
        self.master.title("Image annotation tool")
        self.pack(fill=BOTH, expand=1)
        self.photo_image = ImageTk.PhotoImage(self.img)

        canvas = Canvas(self, width=self.img.size[0] + self.right_panel_width,
                        height=self.img.size[1] + self.bottom_panel_height)
        self._register_num_key_press(canvas)
        canvas.bind('<ButtonPress-1>', self._get_on_canvas_mouse_click(canvas))
        self.draw_image(canvas)
        self.fill_image_with_minitiles(canvas)
        self._redraw(canvas)
        canvas.pack(fill=BOTH, expand=1)

        self.draw_press_space_to_annotate_text(self.root)

    def _register_num_key_press(self, canvas):
        self.root.bind('<KeyPress>', self.get_on_key_press(canvas))

    def get_zoom_function(self, canvas):
        def do_zoom(event):
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            factor = 1.001 ** event.delta
            canvas.scale(ALL, x, y, factor, factor)
            print("zooming")

        return do_zoom

    def _get_on_canvas_mouse_click(self, canvas):
        def on_canvas_mouse_click(event):
            # Adjust click location with respect to canvas' dragged position
            clicked_x = event.x - index_to_pixel(self.drag_x, self.minitile_size)
            clicked_y = event.y - index_to_pixel(self.drag_y, self.minitile_size)

            if self._has_clicked_on_minitile(clicked_x, clicked_y):
                self._annotate_minitile(clicked_x, clicked_y)
                self._redraw(canvas)

        return on_canvas_mouse_click

    def _has_clicked_on_minitile(self, x, y):
        x_within_bounds = 0 <= x < self.img.size[0]
        y_within_bounds = 0 <= y < self.img.size[1]
        return x_within_bounds and y_within_bounds

    def _annotate_minitile(self, x, y):
        row = pixel_to_index(x, self.minitile_size)
        col = pixel_to_index(y, self.minitile_size)
        self.state_storage.set_annotation(col, row, self.active_type)

    def get_on_key_press(self, canvas):
        left = 113
        up = 111
        right = 114
        down = 116

        def drag_canvas(key):
            delta_x = 0
            delta_y = 0
            gain = 2

            if key == left:
                delta_x += gain
            elif key == right:
                delta_x -= gain
            elif key == up:
                delta_y += gain
            elif key == down:
                delta_y -= gain

            self.drag_x += delta_x
            self.drag_y += delta_y

            x_pix = index_to_pixel(delta_x, self.minitile_size)
            y_pix = index_to_pixel(delta_y, self.minitile_size)
            canvas.scan_mark(0, 0)
            canvas.scan_dragto(x_pix, y_pix, gain=1)

        def on_key_press(e):
            if len(e.char) != 0:
                key = ord(e.char)
                zero_ascii = 48

                if self._is_existing_type(key - zero_ascii):
                    self._change_active_type(key - zero_ascii)
                    self._redraw(canvas)

                if e.char == 's':
                    print('Saving annotations...')
                    self.state_storage.save_annotations()
                    self.state_storage.print_annotations_stats()

            drag_canvas(e.keycode)


        return on_key_press

    def _is_existing_type(self, type_index: int) -> bool:
        return 0 <= type_index < self.num_classes

    def _change_active_type(self, type: int) -> None:
        self.active_type = type
        type_name = self.types[type]
        print(f'Active type {type}: {type_name}')

    def _redraw(self, canvas: Canvas):
        self._delete_drawn_annotations(canvas)
        self._draw_annotations(canvas)

    def _delete_drawn_annotations(self, canvas: Canvas) -> None:
        canvas.delete(self.annotation_tag)

    def _draw_annotations(self, canvas: Canvas) -> None:
        annotations = self.state_storage.get_annotations()
        for row in range(annotations.shape[1]):
            for col in range(annotations.shape[0]):
                annotation = annotations[col][row]
                if self._is_annotation(annotation):
                    color = self._get_annotation_color(annotation)
                    self._draw_annotation(col, row, canvas, color)

    def _is_annotation(self, annotation: int) -> bool:
        return 0 <= annotation < self.num_classes

    def _get_annotation_color(self, annotation: int) -> str:
        if self._is_inactive_annotation(annotation):
            return "#ab80cf"
        return "#342469"

    def _is_inactive_annotation(self, annotation: int) -> bool:
        return annotation != self.active_type

    def _draw_annotation(self, col: int, row: int, canvas: Canvas, color: str) -> None:
        x = index_to_pixel(col, self.minitile_size)
        y = index_to_pixel(row, self.minitile_size)

        canvas.create_rectangle(x, y, x + self.minitile_size, y + self.minitile_size,
                                outline=color, tags=self.annotation_tag, width=3)

    def draw_image(self, canvas: Canvas):
        canvas.create_image(self.image_x_start, self.image_y_start, anchor=NW, image=self.photo_image)

    def fill_image_with_minitiles(self, canvas):
        for row in range(self.image_y_start, self.image_y_start + self.img.size[1], self.minitile_size):
            for col in range(self.image_x_start, self.image_x_start + self.img.size[0], self.minitile_size):
                canvas.create_rectangle(col, row, self.minitile_size, self.minitile_size, outline="#6b5c5c")

    def highlight_active_minitile(self, canvas):
        col = index_to_pixel(self.active_minitile[0], self.minitile_size)
        row = index_to_pixel(self.active_minitile[1], self.minitile_size)
        canvas.create_rectangle(col, row, col + self.minitile_size, row + self.minitile_size, outline="#bf1b1b",
                                width=3)

    def draw_press_space_to_annotate_text(self, parent):
        y = self.img.size[1] + 50
        x = 50
        text = "Press space to annotate a minitile."
        text_element = Text(parent, width=x, height=y)
        text_element.insert(1.0, text)


def main():
    root = Tk()

    minitile_size = 40
    image_path = "datasets/opssat/raw/001.png"
    img = Image.open(image_path)
    minitiles_shape = (pixel_to_index(img.size[0], minitile_size) + 1,
                       pixel_to_index(img.size[1], minitile_size) + 1)
    storage = StateStorage(image_path, minitiles_shape)
    ex = UI(root, img, num_classes=8, minitile_size=minitile_size, state_storage=storage)
    root.mainloop()


if __name__ == '__main__':
    main()

from PIL.Image import Image as PilImage
import textwrap, os
import matplotlib.pyplot as plt


def display_images(
    image_list: [[PilImage]],
    columns=2, width=50, height=35, max_images=20,
    label_wrap_length=50, label_font_size=8):

    # flatten images first
    images = [x for image in image_list for x in image]

    if not images:
        print("No images to display.")
        return 

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size)
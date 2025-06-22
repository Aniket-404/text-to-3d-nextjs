from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_image(text="Placeholder Image", output_path=None):
    """Create a visually appealing placeholder image with the given text."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "placeholder.png")
    
    # Create a dark blue background
    image = Image.new('RGB', (512, 512), color=(15, 25, 60))
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to find a font on the system
        font_path = None
        possible_font_locations = [
            # Windows fonts
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            # Mac fonts
            "/Library/Fonts/Arial.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
        
        for path in possible_font_locations:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            title_font = ImageFont.truetype(font_path, 30)
            subtitle_font = ImageFont.truetype(font_path, 20)
        else:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
    except Exception as e:
        print(f"Error loading font: {e}")
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
    
    # Add a gradient background
    for y in range(512):
        r = int(15 + (y / 512) * 20)
        g = int(25 + (y / 512) * 20)
        b = int(60 + (y / 512) * 40)
        for x in range(512):
            r_offset = int((x / 512) * 10)
            b_offset = int((1 - x / 512) * 20)
            draw.point((x, y), fill=(r + r_offset, g, b + b_offset))
    
    # Draw a border
    border_width = 5
    draw.rectangle([
        (border_width, border_width),
        (512-border_width, 512-border_width)
    ], outline=(0, 180, 255), width=border_width)
    
    # Add title
    title_text = "Image Generation Failed"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (512 - title_width) // 2
    draw.text((title_x, 100), title_text, fill=(255, 255, 255), font=title_font)
    
    # Add error message
    subtitle_text = text
    lines = []
    words = subtitle_text.split()
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= 30:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Draw each line
    y_offset = 180
    for line in lines:
        subtitle_bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (512 - subtitle_width) // 2
        draw.text(
            (subtitle_x, y_offset),
            line,
            fill=(200, 200, 255),
            font=subtitle_font
        )
        y_offset += 30
    
    # Add footer
    footer_text = "Please try again with a different prompt"
    footer_font = subtitle_font
    footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
    footer_width = footer_bbox[2] - footer_bbox[0]
    footer_x = (512 - footer_width) // 2
    draw.text(
        (footer_x, 450),
        footer_text,
        fill=(180, 180, 220),
        font=footer_font
    )
    
    # Save the image
    image.save(output_path)
    print(f"Placeholder image saved to {output_path}")
    return image

if __name__ == "__main__":
    # Create default placeholder when run directly
    create_placeholder_image(
        "No image could be generated for this prompt. Please try again."
    )

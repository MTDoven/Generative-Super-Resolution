from pathlib import Path
from typing import Tuple, Union
from PIL import Image
import fitz


def _resize_with_center_padding(image: Image.Image, target_hw: Tuple[int, int]) -> Image.Image:
    target_h, target_w = target_hw
    assert target_h % 16 == 0 and target_w % 16 == 0, \
        f"Output size must be multiples of 16, got height={target_h}, width={target_w}."
    # Resize the image while maintaining aspect ratio, then center-pad to target size.
    src_w, src_h = image.size
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image size: {(src_h, src_w)}")
    # Calculate the scaling factor to fit the image within the target dimensions.
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = image.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)
    # Keep image centered and pad with black pixels.
    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    paste_x = (target_w - resized_w) // 2
    paste_y = (target_h - resized_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def render_pdf_to_image(pdf_path: Union[str, Path], target_hw: Tuple[int, int]) -> Image.Image:
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"
    with fitz.open(pdf_path) as doc:
        assert len(doc) == 1, f"Expected exactly 1 page, got {len(doc)} pages."
        page = doc.load_page(0)
        pix = page.get_pixmap(alpha=False)
        page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    output = _resize_with_center_padding(page_image, target_hw)
    return output


def render_pdf_to_text(pdf_path: Union[str, Path]) -> str:
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"PDF file not found: {pdf_path}"
    with fitz.open(pdf_path) as doc:
        assert len(doc) == 1, f"Expected exactly 1 page, got {len(doc)} pages."
        page = doc.load_page(0)
        page_dict = page.get_text("dict")
        block_texts = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            line_texts = []
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                line_text = line_text.strip()
                if line_text:
                    line_texts.append(line_text)
            if line_texts:
                block_texts.append("\n".join(line_texts))
        return "\n\n".join(block_texts).strip()

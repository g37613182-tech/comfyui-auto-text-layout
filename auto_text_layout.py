import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class AutoTextLayout:
    """
    ComfyUI node: Analyze image composition via subject mask,
    find the optimal blank region, and render text automatically.
    
    Upstream: IMAGE + MASK (from any segment/SAM node, 1=subject)
    Downstream: Composed IMAGE + text area MASK + placement coordinates
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "text": ("STRING", {"default": "Your text here", "multiline": True}),
                "font_path": ("STRING", {"default": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"}),
                "font_size": ("INT", {"default": 48, "min": 8, "max": 512, "step": 1}),
                "font_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "alignment": (["auto", "left", "center", "right"],),
                "placement": (["auto_largest", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"],),
                "margin": ("INT", {"default": 30, "min": 0, "max": 300, "step": 5}),
                "line_spacing": ("FLOAT", {"default": 1.3, "min": 0.8, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "stroke_color_hex": ("STRING", {"default": ""}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
                "auto_font_size": ("BOOLEAN", {"default": False}),
                "max_text_width_ratio": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05}),
                "subject_padding": ("INT", {"default": 15, "min": 0, "max": 100, "step": 5,
                                             "tooltip": "Extra padding around subject to avoid text overlap"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "text_mask", "text_x", "text_y", "text_w", "text_h")
    FUNCTION = "execute"
    CATEGORY = "text/layout"
    DESCRIPTION = "Analyze image composition via subject mask and auto-place text in the best blank area."

    # ------------------------------------------------------------------ #
    #                          MAIN EXECUTE                               #
    # ------------------------------------------------------------------ #

    def execute(self, image, mask, text, font_path, font_size, font_color_hex,
                alignment, placement, margin, line_spacing,
                stroke_color_hex="", stroke_width=0, auto_font_size=False,
                max_text_width_ratio=0.45, subject_padding=15):

        B = image.shape[0]
        out_images = []
        out_text_masks = []
        # batch 中只取第一帧坐标作为标量输出
        first_x = first_y = first_w = first_h = 0

        for b in range(B):
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            mask_np = mask[b].cpu().numpy()  # 1 = subject
            H, W = img_np.shape[:2]

            # ---- 1. Build availability map (1 = safe to place text) ---- #
            avail = (1.0 - mask_np)  # invert mask
            avail_binary = (avail > 0.5).astype(np.uint8)

            # Erode: push text away from subject boundary
            erode_iter = max(1, subject_padding // 3)
            avail_binary = self._erode(avail_binary, erode_iter)

            # Also keep margin from image edges
            if margin > 0:
                avail_binary[:margin, :] = 0
                avail_binary[-margin:, :] = 0
                avail_binary[:, :margin] = 0
                avail_binary[:, -margin:] = 0

            # ---- 2. Find best placement region ---- #
            region = self._find_region(avail_binary, W, H, placement, max_text_width_ratio, margin)
            rx, ry, rw, rh = region

            # ---- 3. Load font ---- #
            font = self._load_font(font_path, font_size)

            # ---- 4. Auto font size (optional) ---- #
            if auto_font_size and font_path and os.path.isfile(font_path):
                font_size = self._calc_auto_font_size(text, font_path, region, line_spacing)
                font = self._load_font(font_path, font_size)

            # ---- 5. Wrap text ---- #
            lines = self._wrap_text(text, font, rw)

            # ---- 6. Measure text block ---- #
            line_metrics = []
            for line in lines:
                bbox = font.getbbox(line)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                y_offset = bbox[1]  # top bearing
                line_metrics.append((lw, lh, y_offset))

            gap = int(font_size * (line_spacing - 1))
            total_h = sum(m[1] for m in line_metrics) + gap * max(0, len(lines) - 1)
            max_w = max((m[0] for m in line_metrics), default=0)

            # ---- 7. Calculate origin ---- #
            # Vertical: center in region
            start_y = ry + max(0, (rh - total_h) // 2)

            # Horizontal alignment
            actual_align = alignment
            if alignment == "auto":
                region_cx = rx + rw / 2
                img_cx = W / 2
                if abs(region_cx - img_cx) < W * 0.1:
                    actual_align = "center"
                elif region_cx < img_cx:
                    actual_align = "left"
                else:
                    actual_align = "right"

            # ---- 8. Render ---- #
            pil_img = Image.fromarray(img_np, 'RGB')
            draw = ImageDraw.Draw(pil_img)

            text_mask_img = Image.new('L', (W, H), 0)
            tm_draw = ImageDraw.Draw(text_mask_img)

            fg = self._hex_to_rgb(font_color_hex)
            sc = self._hex_to_rgb(stroke_color_hex) if stroke_color_hex.strip() else None

            cur_y = start_y
            for idx, line in enumerate(lines):
                lw, lh, y_off = line_metrics[idx]

                if actual_align == "center":
                    lx = rx + (rw - lw) // 2
                elif actual_align == "right":
                    lx = rx + rw - lw
                else:
                    lx = rx

                pos = (lx, cur_y - y_off)

                # Stroke
                if sc and stroke_width > 0:
                    for ox in range(-stroke_width, stroke_width + 1):
                        for oy in range(-stroke_width, stroke_width + 1):
                            if ox * ox + oy * oy <= stroke_width * stroke_width:
                                draw.text((pos[0] + ox, pos[1] + oy), line, font=font, fill=sc)
                                tm_draw.text((pos[0] + ox, pos[1] + oy), line, font=font, fill=255)

                draw.text(pos, line, font=font, fill=fg)
                tm_draw.text(pos, line, font=font, fill=255)

                cur_y += lh + gap

            # ---- 9. Convert back ---- #
            result_np = np.array(pil_img).astype(np.float32) / 255.0
            out_images.append(torch.from_numpy(result_np))

            tmask_np = np.array(text_mask_img).astype(np.float32) / 255.0
            out_text_masks.append(torch.from_numpy(tmask_np))

            if b == 0:
                first_x, first_y, first_w, first_h = rx, start_y, max_w, total_h

        return (
            torch.stack(out_images),
            torch.stack(out_text_masks),
            first_x, first_y, first_w, first_h,
        )

    # ------------------------------------------------------------------ #
    #                       REGION FINDING                                #
    # ------------------------------------------------------------------ #

    def _find_region(self, avail, W, H, strategy, max_w_ratio, margin):
        """Return (x, y, w, h) of the best rectangle for text."""

        if strategy == "auto_largest":
            return self._largest_blank_rect(avail, W, H, margin)

        max_w = int(W * max_w_ratio)
        m = margin

        preset = {
            "top":          (m, m, W - 2 * m, H // 3 - m),
            "bottom":       (m, 2 * H // 3, W - 2 * m, H // 3 - m),
            "left":         (m, m, max_w, H - 2 * m),
            "right":        (W - max_w - m, m, max_w, H - 2 * m),
            "top_left":     (m, m, W // 2 - m, H // 3 - m),
            "top_right":    (W // 2, m, W // 2 - m, H // 3 - m),
            "bottom_left":  (m, 2 * H // 3, W // 2 - m, H // 3 - m),
            "bottom_right": (W // 2, 2 * H // 3, W // 2 - m, H // 3 - m),
        }

        if strategy in preset:
            return self._clamp_region(preset[strategy], W, H)

        return self._largest_blank_rect(avail, W, H, margin)

    def _largest_blank_rect(self, avail, W, H, margin):
        """
        Maximal rectangle in a binary matrix.
        Downsamples for performance, then scales back.
        """
        scale = max(1, min(W, H) // 200)  # adaptive downscale
        sH, sW = H // scale, W // scale

        small = self._downsample(avail, sW, sH)

        # Histogram-based maximal rectangle
        heights = np.zeros(sW, dtype=int)
        best_area = 0
        best = (0, 0, sW, sH)

        for row in range(sH):
            for col in range(sW):
                heights[col] = heights[col] + 1 if small[row, col] else 0

            # Largest rectangle in histogram (stack-based)
            stack = []
            for col in range(sW + 1):
                cur_h = heights[col] if col < sW else 0
                start = col
                while stack and stack[-1][1] > cur_h:
                    s_col, s_h = stack.pop()
                    area = s_h * (col - s_col)
                    if area > best_area:
                        best_area = area
                        best = (s_col, row - s_h + 1, col - s_col, s_h)
                    start = s_col
                stack.append((start, cur_h))

        # Scale back
        rx = best[0] * scale
        ry = best[1] * scale
        rw = best[2] * scale
        rh = best[3] * scale

        # Shrink by small inner margin for aesthetics
        inner = min(margin // 2, 10)
        rx += inner
        ry += inner
        rw -= 2 * inner
        rh -= 2 * inner

        return self._clamp_region((rx, ry, max(rw, 50), max(rh, 50)), W, H)

    # ------------------------------------------------------------------ #
    #                         TEXT UTILITIES                               #
    # ------------------------------------------------------------------ #

    def _wrap_text(self, text, font, max_width):
        """
        Character-level wrapping (CJK-safe).
        Respects explicit newlines in input text.
        """
        if max_width <= 0:
            return [text]

        all_lines = []
        for paragraph in text.split('\n'):
            if not paragraph:
                all_lines.append("")
                continue

            current = ""
            for ch in paragraph:
                test = current + ch
                bbox = font.getbbox(test)
                tw = bbox[2] - bbox[0]
                if tw <= max_width:
                    current = test
                else:
                    if current:
                        all_lines.append(current)
                    current = ch
            if current:
                all_lines.append(current)

        return all_lines if all_lines else [""]

    def _calc_auto_font_size(self, text, font_path, region, line_spacing):
        """Binary search for a font size that fills ~60% of the region."""
        _, _, rw, rh = region
        lo, hi, best = 12, 256, 24

        for _ in range(16):  # max iterations
            mid = (lo + hi) // 2
            try:
                font = ImageFont.truetype(font_path, mid)
            except Exception:
                break

            lines = self._wrap_text(text, font, rw)
            line_h = font.getbbox("Ayg")[3] - font.getbbox("Ayg")[1]
            total_h = len(lines) * line_h + int((len(lines) - 1) * mid * (line_spacing - 1))
            max_w = max((font.getbbox(l)[2] - font.getbbox(l)[0] for l in lines), default=0)

            if total_h > rh or max_w > rw:
                hi = mid - 1
            else:
                best = mid
                lo = mid + 1

        return max(12, best)

    # ------------------------------------------------------------------ #
    #                        HELPER FUNCTIONS                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _erode(binary, iterations):
        """Pure-numpy binary erosion (no scipy needed)."""
        result = binary.copy()
        for _ in range(iterations):
            padded = np.pad(result, 1, mode='constant', constant_values=0)
            result = np.minimum.reduce([
                padded[:-2, 1:-1],   # top
                padded[2:, 1:-1],    # bottom
                padded[1:-1, :-2],   # left
                padded[1:-1, 2:],    # right
                padded[1:-1, 1:-1],  # center
            ])
        return result

    @staticmethod
    def _downsample(arr, tw, th):
        """Nearest-neighbor downsample a 2D array."""
        h, w = arr.shape
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return arr[np.ix_(row_idx, col_idx)]

    @staticmethod
    def _clamp_region(region, W, H):
        x, y, w, h = region
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(50, min(w, W - x))
        h = max(50, min(h, H - y))
        return (x, y, w, h)

    @staticmethod
    def _load_font(path, size):
        try:
            if path and os.path.isfile(path):
                return ImageFont.truetype(path, size)
        except Exception:
            pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _hex_to_rgb(hex_str):
        hex_str = hex_str.strip().lstrip('#')
        if len(hex_str) == 6:
            return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        if len(hex_str) == 8:  # RRGGBBAA → ignore alpha
            return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        return (255, 255, 255)

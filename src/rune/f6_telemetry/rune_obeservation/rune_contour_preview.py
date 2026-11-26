#!/usr/bin/env python3
"""Preview color masks and contour shapes in local rune images or recordings."""

import argparse
from collections import Counter
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2


SHAPE_COLORS = {
    "Ring": (50, 210, 255),
    "Strip": (105, 245, 140),
    "Region": (190, 190, 190),
}
MORPHOLOGY_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def render_preview(frame, color, threshold, min_area, frame_number):
    height, width = frame.shape[:2]
    scale = min(1.0, 800 / width, 640 / height)
    size = (max(2, round(width * scale) // 2 * 2),
            max(2, round(height * scale) // 2 * 2))
    if size != (width, height):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    blue = cv2.extractChannel(frame, 0)
    red = cv2.extractChannel(frame, 2)
    difference = cv2.subtract(blue, red) if color == "blue" else cv2.subtract(red, blue)
    mask = cv2.inRange(difference, threshold, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPHOLOGY_KERNEL)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()
    counts = Counter()
    if hierarchy is not None:
        links = hierarchy[0]
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if links[index][3] >= 0 or area < min_area:
                continue
            holes = []
            child = links[index][2]
            while child >= 0:
                holes.append(contours[child])
                child = links[child][0]
            hole_area = max((cv2.contourArea(hole) for hole in holes), default=0)
            short_side, long_side = sorted(cv2.minAreaRect(contour)[1])
            if 0.15 <= hole_area / area <= 0.9 and short_side >= 0.4 * long_side:
                shape = "Ring"
            elif long_side >= 3 * short_side:
                shape = "Strip"
            else:
                shape = "Region"
            cv2.drawContours(annotated, [contour, *holes], -1, SHAPE_COLORS[shape], 2, cv2.LINE_AA)
            counts[shape] += 1

    left = cv2.copyMakeBorder(annotated, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(26, 25, 24))
    right = cv2.copyMakeBorder(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 36, 0, 0, 0,
                               cv2.BORDER_CONSTANT, value=(26, 25, 24))
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 12
    for shape, ink in SHAPE_COLORS.items():
        label = f"{shape} {counts[shape]}"
        cv2.putText(left, label, (x, 23), font, 0.5, ink, 1, cv2.LINE_AA)
        x += cv2.getTextSize(label, font, 0.5, 1)[0][0] + 16
    cv2.putText(right, f"{color.capitalize()} mask  |  Frame {frame_number}", (12, 23),
                font, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return cv2.hconcat([left, right]), counts


def write_video(args, output):
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    frames = 0
    counts = Counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"无法读取录像 {args.input}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("录像缺少有效帧率")
        while args.frames is None or frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            preview, shapes = render_preview(frame, args.color, args.threshold,
                                             args.min_area, frames + 1)
            size = (preview.shape[1], preview.shape[0])
            if writer is None:
                output_size = size
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise OSError("无法创建 MP4 预览")
            elif size != output_size:
                raise ValueError("录像中途改变了画面尺寸")
            writer.write(preview)
            frames += 1
            counts.update(shapes)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        raise ValueError("录像中没有可读帧")
    return frames, counts


def main():
    parser = argparse.ArgumentParser(description="为本地图片或录像生成传统视觉轮廓预览")
    parser.add_argument("input", type=Path, help="本地图片或录像")
    parser.add_argument("-o", "--output", required=True, type=Path, help="图片用 .png，录像用 .mp4")
    parser.add_argument("--color", choices=("blue", "red"), default="blue", help="检查的灯光颜色")
    parser.add_argument("--threshold", type=int, default=24, help="颜色通道差阈值，默认 24")
    parser.add_argument("--min-area", type=int, default=35, help="预览画面的最小轮廓面积，默认 35")
    parser.add_argument("--frames", type=int, help="最多处理的录像帧数，默认处理整段")
    args = parser.parse_args()
    if not 1 <= args.threshold <= 255 or args.min_area < 1:
        parser.error("--threshold 必须在 1 到 255 之间，--min-area 必须为正数")
    if args.frames is not None and args.frames < 1:
        parser.error("--frames 必须为正数")
    if not args.input.is_file():
        parser.error(f"输入必须是存在的本地文件 {args.input}")
    if args.output.exists():
        parser.error(f"输出文件已存在 {args.output}")

    try:
        image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
        suffix = ".mp4" if image is None else ".png"
        if args.output.suffix.lower() != suffix:
            raise ValueError(f"此输入需要输出为 {suffix}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as directory:
            temporary = Path(directory) / f"preview{suffix}"
            if image is None:
                frames, counts = write_video(args, temporary)
            else:
                preview, counts = render_preview(image, args.color, args.threshold, args.min_area, 1)
                if not cv2.imwrite(str(temporary), preview):
                    raise OSError("无法写入 PNG 预览")
                frames = 1
            if temporary.stat().st_size == 0:
                raise OSError("预览文件为空")
            args.output.hardlink_to(temporary)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))

    print(f"已输出 {frames} 帧，环形候选 {counts['Ring']}，条形候选 {counts['Strip']}，"
          f"其他轮廓 {counts['Region']}\n{args.output}")


if __name__ == "__main__":
    main()

import argparse
import cv2
from pathlib import Path
from .model import MoveNetMulti
from .infer_core import process_frame
from .io_utils import read_image, read_gif, write_gif, iter_video, get_video_writer, probe_fps

def main():
    ap = argparse.ArgumentParser(description="MoveNet MultiPose runner (image/video/gif).")
    ap.add_argument("--input", required=True, help="Path to image/video/gif")
    ap.add_argument("--output", required=True, help="Output file path")
    ap.add_argument("--kind", choices=["image", "video", "gif"], required=True)
    ap.add_argument("--size", type=int, default=256, help="Square model input (e.g., 256/192/320)")
    ap.add_argument("--threshold", type=float, default=0.11, help="Keypoint draw threshold")
    ap.add_argument("--fps", type=int, default=15, help="Write FPS for GIF/video if probe fails")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = MoveNetMulti(input_size=args.size)

    if args.kind == "image":
        img = read_image(str(in_path))
        out = process_frame(model, img, size=args.size, thresh=args.threshold)
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            raise RuntimeError(f"Could not write image: {out_path}")

    elif args.kind == "gif":
        frames = read_gif(str(in_path))
        outs = [process_frame(model, f, size=args.size, thresh=args.threshold) for f in frames]
        write_gif(str(out_path), outs, fps=args.fps)

    else:  # video
        fps = probe_fps(str(in_path), default_fps=args.fps)
        gen = iter_video(str(in_path))
        first = next(gen, None)
        if first is None:
            raise RuntimeError("No frames read from video.")
        h, w = first.shape[:2]
        writer = get_video_writer(str(out_path), w, h, fps)
        try:
            writer.write(process_frame(model, first, size=args.size, thresh=args.threshold))
            for frame in gen:
                writer.write(process_frame(model, frame, size=args.size, thresh=args.threshold))
        finally:
            writer.release()

if __name__ == "__main__":
    main()

import argparse

from PIL import Image

from carve import carve, load_model
from carve.pipeline import VERSIONS


def parse_args():
    p = argparse.ArgumentParser(description="CARVE: Contrastive Attention Refinement")
    p.add_argument("--model-id", required=True, help="HF id or local path")
    p.add_argument("--family", default="qwen2_5", choices=["qwen2_5", "llava"])
    p.add_argument("--image", required=True, help="path to the input image")
    p.add_argument("--question", required=True, help="question about the image")
    p.add_argument("--version", default="v4", choices=list(VERSIONS),
                   help="method variant (v1: remove-small, v2: by size, "
                        "v3: by attention sum, v4: + all-step time-weighted fusion)")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(20, 26)),
                   help="deep decoder layers to fuse")
    p.add_argument("--lam", type=float, default=1e-8)
    p.add_argument("--top-percent", type=float, default=0.2)
    p.add_argument("--min-keep-regions", type=int, default=3)
    p.add_argument("--outlier-percent", type=float, default=0.1, help="only used by v1")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-dir", default=None,
                   help="if set, dump sum_att_map / mask / crop / marked images here")
    p.add_argument("--plot", action="store_true",
                   help="save pipeline.png and layerwise.png figures (needs matplotlib)")
    return p.parse_args()


def main():
    args = parse_args()
    image = Image.open(args.image).convert("RGB")

    print(f"Loading {args.family} model: {args.model_id}")
    model, processor = load_model(args.model_id, family=args.family, device=args.device)

    print(f"Running CARVE ({args.version}) ...")
    out = carve(
        model, processor, image, args.question,
        family=args.family, version=args.version,
        layers=args.layers, lam=args.lam,
        top_percent=args.top_percent, min_keep_regions=args.min_keep_regions,
        outlier_percent=args.outlier_percent, max_new_tokens=args.max_new_tokens,
        device=args.device, save_dir=args.save_dir,
    )

    print("\n" + "=" * 60)
    print(f"Question        : {args.question}")
    print(f"Version         : {args.version}")
    print(f"Original answer : {out.original_answer}")
    print(f"CARVE  answer   : {out.refined_answer}")
    print(f"Crop bbox       : {out.bbox}")
    if args.save_dir:
        print(f"Saved intermediates to: {args.save_dir}/")
    print("=" * 60)

    if args.plot:
        import os

        from carve.visualize import plot_layerwise, plot_pipeline

        out_dir = args.save_dir or "outputs"
        os.makedirs(out_dir, exist_ok=True)
        plot_pipeline(image, out, save_path=os.path.join(out_dir, "pipeline.png"))
        plot_layerwise(out, lam=args.lam, save_path=os.path.join(out_dir, "layerwise.png"))
        print(f"Saved figures to: {out_dir}/pipeline.png, {out_dir}/layerwise.png")


if __name__ == "__main__":
    main()

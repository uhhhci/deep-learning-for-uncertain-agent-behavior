import argparse, json
from models import BlendShapeVAE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments for a machine learning model.')
    parser.add_argument("--confidence", type=float, required=True,
                        help='Confidence level of the agent (1 = very confident, 0 = not confident)')
    parser.add_argument("--output", type=str, required=True,
                        help="Output filename for the blendshape and metadata sequence")
    parser.add_argument('--model_path', type=str, required=False,
                        default="gpt-gold-model/epoch=499-val_loss=3.50-val_kl=0.00-val_rec=3.42-val_meta=0.00.ckpt",
                        help='Path to the machine learning model file.')
    args = parser.parse_args()

    model = BlendShapeVAE.load_from_checkpoint(
        args.model_path,
        map_location="cpu",
        strict=False
    )
    model = model.to("cuda")
    assert args.confidence >= 0.0 and args.confidence <= 1.0, "Confidence must be between 0 and 1"

    while True:
        try:
            seq, out = model.generate(args.confidence, mean=0.0, std=0.5, use_regressor=True)
            out = out.strip().replace("</s>", "")
            generated = json.loads(out.split("<|assistant|>")[1])
            confidence, intonation, filler, pre_hedge, post_hedge, pre_length, perform_length, post_length = \
                args.confidence, generated["intonation"], generated["filler"], \
                generated["pre_hedge"], generated["post_hedge"], generated["pre_length"], generated["perform_length"], generated["post_length"]

            seq.iloc[:(int(pre_length) + int(perform_length) + int(post_length))].to_csv(f"{args.output}.txt", index=False)
            break
        except Exception as e:
            print(e)
            pass

    with open(f"{args.output}.json", "w") as file:
        json.dump({
            "intonation": intonation,
            "fillerWord": filler,
            "preHedge": pre_hedge,
            "postHedge": post_hedge,
            "prePause": int(pre_length) / 60,
            "postPause": int(post_length) / 60,
            "preLength": int(pre_length),
            "performLength": int(perform_length),
            "postLength": int(post_length),
        }, file)
    
    print(seq)
    print(f"""
Confidence:     {confidence}
Intonation:     {intonation}
Filler:         {filler}
Pre-hedge:      {pre_hedge}
Post-hedge:     {post_hedge}
Pre-length:     {pre_length}
Perform-length: {perform_length}
Post-length:    {post_length}
""")

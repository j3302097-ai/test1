import os, json, argparse
import numpy as np
from PIL import Image

from payload.secret_to_bits import secret_img_to_bits, bits_to_secret_img
from payload.crypto_shuffle import permute_bits, inv_permute_bits, whiten_bits, dewhiten_bits
from payload.ecc_codec import ECC

from attacks.attacks_min import jpeg, resize_bilinear, gaussian_blur, gaussian_noise

def read_rgb_uint8(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_rgb_uint8(path, arr):
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)

def ber(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.uint8).reshape(-1)
    b = b.astype(np.uint8).reshape(-1)
    n = min(a.size, b.size)
    if n == 0: return 1.0
    return float((a[:n] != b[:n]).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True)
    ap.add_argument("--secret", required=True)
    ap.add_argument("--secret_size", type=int, default=16, choices=[16,32,64])
    ap.add_argument("--key", default="demo_key_change_me")
    ap.add_argument("--ecc_mode", default="rs", choices=["rs","rep3"])
    ap.add_argument("--rs_nsym", type=int, default=32)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--delta", type=float, default=12.0)
    ap.add_argument("--device", default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--out_dir", default="tmp_out/run1")
    ap.add_argument("--attack", default="none",
                    choices=["none","jpeg90","jpeg70","resize0.75","blur1.0","noise3.0"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- archive inputs for reproducibility ---
    from PIL import Image
    import shutil
    # copy cover & secret into out_dir
    try:
        shutil.copy2(args.cover, os.path.join(args.out_dir, "cover_input.png"))
    except Exception:
        pass
    try:
        shutil.copy2(args.secret, os.path.join(args.out_dir, "secret_input.png"))
    except Exception:
        pass

    # save run config (key is sensitive; store a hash instead)
    import hashlib
    key_hash = hashlib.sha256(args.key.encode("utf-8")).hexdigest()[:16]
    run_cfg = {
        "cover": args.cover,
        "secret": args.secret,
        "secret_size": args.secret_size,
        "attack": args.attack,
        "ecc_mode": args.ecc_mode,
        "rs_nsym": args.rs_nsym,
        "block": args.block,
        "delta": args.delta,
        "device": args.device,
        "key_sha256_16": key_hash,
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(run_cfg, f, indent=2)
    # ------------------------------------------

    # 1) secret -> bits
    bits_raw, hw = secret_img_to_bits(args.secret, args.secret_size)

    # 2) permute + whiten
    bits_perm, perm = permute_bits(bits_raw, args.key)
    bits_w, mask = whiten_bits(bits_perm, args.key)

    # 3) ECC encode
    ecc = ECC(mode=args.ecc_mode, rs_nsym=args.rs_nsym)
    enc = ecc.encode(bits_w)
    bits_ecc = enc["bits"]
    ecc_meta = enc["meta"]

    np.save(os.path.join(args.out_dir, "bits_ecc.npy"), bits_ecc.astype(np.uint8), allow_pickle=False)
    with open(os.path.join(args.out_dir, "ecc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(ecc_meta, f, indent=2)
    np.save(os.path.join(args.out_dir, "perm.npy"), perm.astype(np.int64), allow_pickle=False)
    np.save(os.path.join(args.out_dir, "white_mask.npy"), mask.astype(np.uint8), allow_pickle=False)

    # 4) embed (call module script)
    stego_path = os.path.join(args.out_dir, "stego.png")
    sidecar_path = os.path.join(args.out_dir, "sidecar.json")

    cmd = (
        f"python -m embed.embed_ll1_qim "
        f"--cover {args.cover} "
        f"--secret_bits_npy {os.path.join(args.out_dir,'bits_ecc.npy')} "
        f"--out_stego {stego_path} "
        f"--out_sidecar {sidecar_path} "
        f"--block {args.block} --delta {args.delta} --device {args.device}"
    )
    print("[RUN]", cmd)
    if os.system(cmd) != 0:
        raise SystemExit("embed failed")

    # 5) attack
    stego = read_rgb_uint8(stego_path)
    attacked = stego.copy()
    if args.attack == "jpeg90":
        attacked = jpeg(attacked, 90)
    elif args.attack == "jpeg70":
        attacked = jpeg(attacked, 70)
    elif args.attack == "resize0.75":
        attacked = resize_bilinear(attacked, 0.75)
    elif args.attack == "blur1.0":
        attacked = gaussian_blur(attacked, 1.0)
    elif args.attack == "noise3.0":
        attacked = gaussian_noise(attacked, 3.0)

    attacked_path = os.path.join(args.out_dir, f"attacked_{args.attack}.png")
    save_rgb_uint8(attacked_path, attacked)

    # 6) extract bits (call module script)
    bits_hat_npy = os.path.join(args.out_dir, "bits_hat_ecc.npy")
    preview_path = os.path.join(args.out_dir, "secret_preview_from_rawbits.png")
    cmd = (
        f"python -m extract.extract_ll1_qim "
        f"--stego_or_attacked {attacked_path} "
        f"--sidecar {sidecar_path} "
        f"--ecc_meta {os.path.join(args.out_dir,'ecc_meta.json')} "
        f"--out_bits_npy {bits_hat_npy} "
        f"--out_secret_png {preview_path} "
        f"--secret_size {args.secret_size} "
        f"--device {args.device}"
    )
    print("[RUN]", cmd)
    if os.system(cmd) != 0:
        raise SystemExit("extract failed")

    bits_hat_ecc = np.load(bits_hat_npy).astype(np.uint8)

    # 7) ECC decode
    bits_hat_w = ecc.decode(bits_hat_ecc, ecc_meta)

    # 8) dewhiten + inv permute
    perm = np.load(os.path.join(args.out_dir, "perm.npy")).astype(np.int64)
    mask = np.load(os.path.join(args.out_dir, "white_mask.npy")).astype(np.uint8)
    bits_hat_perm = dewhiten_bits(bits_hat_w, mask)
    bits_hat_raw = inv_permute_bits(bits_hat_perm, perm)

    # 9) save recovered secret
    out_secret = os.path.join(args.out_dir, "secret_recovered.png")
    bits_to_secret_img(bits_hat_raw, hw, out_secret)

    # 10) metrics
    b = ber(bits_raw, bits_hat_raw)
    print(f"[RESULT] attack={args.attack} secret_size={args.secret_size} BER={b:.6f}")
    print("[OK] recovered secret:", out_secret)

if __name__ == "__main__":
    main()

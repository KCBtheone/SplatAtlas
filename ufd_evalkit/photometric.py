import os
import torch
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as compute_ssim


def _load_rgb(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert('RGB')
    return np.array(img)


def compute_photometric_metrics(render_dir: str, gt_dir: str, return_per_view=False) -> dict:
    loss_fn_vgg = lpips.LPIPS(net='vgg').to("cuda")
    psnr_list, lpips_list, ssim_list = [], [], []
    skipped = 0
    per_view = []

    render_files = sorted(f for f in os.listdir(render_dir) if f.endswith('.png'))
    for img_name in render_files:
        render_path = os.path.join(render_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)
        if not os.path.exists(gt_path):
            skipped += 1
            continue

        try:
            r_np = _load_rgb(render_path)
            g_np = _load_rgb(gt_path)
            if r_np.shape != g_np.shape:
                g_pil = Image.fromarray(g_np).resize(
                    (r_np.shape[1], r_np.shape[0]), Image.LANCZOS)
                g_np = np.array(g_pil)

            render_img = torch.from_numpy(r_np).float().cuda() / 255.0
            gt_img     = torch.from_numpy(g_np).float().cuda() / 255.0
            render_img = render_img.permute(2, 0, 1).unsqueeze(0)
            gt_img     = gt_img.permute(2, 0, 1).unsqueeze(0)

            r8 = (torch.clamp(render_img, 0, 1) * 255).byte().float() / 255.0
            g8 = (torch.clamp(gt_img,     0, 1) * 255).byte().float() / 255.0

            mse = torch.mean((r8 - g8) ** 2)
            psnr = (-10.0 * torch.log10(mse)).item() if mse.item() > 1e-12 else 100.0
            with torch.no_grad():
                lpips_val = loss_fn_vgg(r8 * 2 - 1, g8 * 2 - 1).item()

            r_chw = r8.squeeze().permute(1, 2, 0).cpu().numpy()
            g_chw = g8.squeeze().permute(1, 2, 0).cpu().numpy()
            ssim_val = compute_ssim(r_chw, g_chw, data_range=1.0, channel_axis=-1)

            if not (np.isfinite(psnr) and np.isfinite(lpips_val) and np.isfinite(ssim_val)):
                print(f"[WARN] non-finite on {img_name}: PSNR={psnr} LPIPS={lpips_val} SSIM={ssim_val}")
                skipped += 1
                continue

            psnr_list.append(psnr)
            lpips_list.append(lpips_val)
            ssim_list.append(ssim_val)
            if return_per_view:
                per_view.append(dict(image_name=img_name, PSNR=psnr, SSIM=ssim_val, LPIPS_VGG=lpips_val, MSE=float(mse)))
        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")
            skipped += 1

    n = len(psnr_list)
    print(f"[photometric] {render_dir}: used {n} pairs, skipped {skipped}")
    if n == 0:
        return {"PSNR": float('nan'), "LPIPS": float('nan'), "SSIM": float('nan'), "n": 0}
    result = {
        "PSNR":  float(np.mean(psnr_list)),
        "LPIPS": float(np.mean(lpips_list)),
        "SSIM":  float(np.mean(ssim_list)),
        "n": n,
    }
    if return_per_view:
        result["per_view"] = per_view
    return result

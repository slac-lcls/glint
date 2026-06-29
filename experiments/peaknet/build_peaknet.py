"""Build + load PeakNet-673M (ConvNeXtV2-huge backbone + BiFPN + 2-class seg head),
matching the canonical train.fsdp.py construction and peaknet-673m.yaml exactly.
Persistent re-establishment of milestone 1 (the /tmp copy was node-local and lost).

Weights (2.69 GB) live on the LCLS data FS (mounted on psana/compute nodes, NOT login):
  /sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-673m.bin
Env: /sdf/group/lcls/ds/tools/conda_envs/py3.12-nopsana-torch + transformers==4.44.2
"""
import sys
import torch

sys.path.insert(0, "/sdf/home/s/smarches/git/peaknet")
from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig
from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config

WEIGHTS = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml/peaknet-673m.bin"


def build_model():
    hf = dict(num_channels=1, patch_size=4, num_stages=4,
              hidden_sizes=[352, 704, 1408, 2816], depths=[3, 3, 27, 3],
              hidden_act="gelu", initializer_range=0.02, layer_norm_eps=1e-12,
              drop_path_rate=0.0, image_size=1920,
              out_features=["stage1", "stage2", "stage3", "stage4"])
    backbone = ConvNextV2Config(**hf)
    block = BiFPNBlockConfig(relu_inplace=False, down_scale_factor=0.5, up_scale_factor=2,
                             num_features=512, num_levels=4, base_level=4,   # base_level=4 is THE gotcha
                             bn=BNConfig(eps=1e-5, momentum=0.1), fusion=FusionConfig(eps=1e-5))
    bifpn = BiFPNConfig(num_blocks=4, block=block)
    seg = SegHeadConfig(up_scale_factor=[4, 8, 16, 32], num_groups=32, out_channels=256,
                        num_classes=2, base_scale_factor=2, uses_learned_upsample=True)
    cfg = PeakNetConfig(backbone=backbone, bifpn=bifpn, seg_head=seg)
    return PeakNet(cfg)


def load_model(device="cpu", dtype=torch.float32, weights=WEIGHTS, verbose=True):
    m = build_model()
    sd = torch.load(weights, map_location="cpu")
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    keys = list(sd.keys())
    for p in ("_orig_mod.", "module.", "_fsdp_wrapped_module.", "model."):
        if keys and all(kk.startswith(p) for kk in keys):
            sd = {kk[len(p):]: vv for kk, vv in sd.items()}
            keys = list(sd.keys())
    miss, unexp = m.load_state_dict(sd, strict=False)
    if verbose:
        print(f"loaded: missing={len(miss)} unexpected={len(unexp)}", flush=True)
        if miss:
            print("  missing[:4]:", miss[:4])
        if unexp:
            print("  unexpected[:4]:", unexp[:4])
    m.to(device=device, dtype=dtype).eval()
    return m


def seg_forward(m, x):
    """logits (B, num_classes, H, W). Model.forward == seg in this class."""
    fn = getattr(m, "seg", None) or m.__call__
    return fn(x)


if __name__ == "__main__":
    m = load_model()
    n = sum(p.numel() for p in m.parameters())
    print(f"PeakNet params: {n/1e6:.0f}M")
    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        y = seg_forward(m, x)
    print("seg out shape:", tuple(y.shape))

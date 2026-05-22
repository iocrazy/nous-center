import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
REPO="/media/heygo/Program/models/nous/image/diffusers/Flux2-klein-9B"
FP8="/media/heygo/Program/models/nous/image/diffusion_models/flux/Flux2-Klein-9B-True-v2-fp8mixed.safetensors"
DEVICE="cuda:1"; OUT=Path(__file__).parent/"_smoke_out"
def main():
    from diffusers import Flux2Transformer2DModel, ModularPipeline
    from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
    from src.services.inference.component_spec import ComponentSpec
    from src.services.inference.quant_loaders import QUANT_LOADERS
    pipe = ModularPipeline.from_pretrained(REPO); pipe.load_components(torch_dtype=torch.bfloat16); pipe.to(DEVICE)
    spec = ComponentSpec(kind="unet", file=FP8, device=DEVICE, dtype="bfloat16", adapter_arch="flux2")
    sd = QUANT_LOADERS.dispatch(spec)
    print("dequant keys:", len(sd), "| sample key:", next(iter(sd)))
    conv = convert_flux2_transformer_checkpoint_to_diffusers(dict(sd))
    print("converted keys:", len(conv), "| sample:", next(iter(conv)))
    cfg = Flux2Transformer2DModel.load_config(REPO+"/transformer")
    tr = Flux2Transformer2DModel.from_config(cfg).to(torch.bfloat16)
    missing, unexpected = tr.load_state_dict(conv, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    pipe.update_components(transformer=tr.to(DEVICE))
    gen=torch.Generator(device=DEVICE).manual_seed(42)
    t=time.monotonic()
    img=pipe(prompt="a photo of a red fox sitting in autumn leaves, sharp focus, detailed", generator=gen, num_inference_steps=20, height=1024, width=1024).images[0]
    img.save(OUT/"spike_fp8_converted.png")
    print(f"infer {time.monotonic()-t:.1f}s → spike_fp8_converted.png")
main() if __name__=="__main__" else None
try: main()
except Exception: traceback.print_exc()

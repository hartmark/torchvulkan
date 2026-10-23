# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 44 [Radeon RX 9050 / 9060 XT] (rev c0)

Host: AMD Ryzen 7 2700X Eight-Core Processor, 31 GB RAM, kernel 7.2.8-arch1-2, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | rocm | vulkan |
|---|---|---|
| status | ok | ok |
| images OK | 10 | 10 |
| device | AMD Radeon Graphics (gfx1200) | AMD Radeon RX 9060 XT (RADV GFX1200) |
| first image (cold), s | 45.35 | 195.22 |
| warm image median, s | 18.05 | 131.48 |
| warm image min / max, s | 17.69 / 19.2 | 131.02 / 131.98 |
| sampling, s/it (median) | 1.75 | 14.75 |
| all images, s | 210.3 | 1379.2 |
| GPU busy avg, % | 80.7 | 88.8 |
| VRAM start / peak, MB | 57.0 / 12740.0 | 57.0 / 13889.0 |
| GTT peak, MB | 14.0 | 4654.0 |
| process RSS peak, MB | 13515.0 | 10391.0 |
| host MemAvailable min, MB | 6575.0 | 7505.0 |
| ComfyUI CPU avg / max, % of one core | 195.8 / 416.8 | 15.1 / 230.2 |
| host CPU avg, % | 24.7 | 13.3 |
| GPU edge temp start / avg / max, °C | 42.0 / 50.6 / 57.0 | 45.0 / 55.7 / 59.0 |
| GPU junction temp avg / max, °C | 67.8 / 81.0 | 66.4 / 72.0 |
| GPU mem temp avg / max, °C | 55.1 / 62.0 | 59.8 / 64.0 |
| GPU power avg / max, W | 119.3 / 161.0 | 98.2 / 118.0 |
| GPU sclk avg, MHz | 2663.5 | 2898.3 |
| CPU temp max, °C | 74.1 | 79.2 |

**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0  

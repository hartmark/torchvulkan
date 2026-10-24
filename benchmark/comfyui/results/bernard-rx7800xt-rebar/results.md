# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | rocm | vulkan |
|---|---|---|
| status | ok | ok |
| images OK | 10 | 10 |
| device | AMD Radeon Graphics (gfx1101) | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 34.37 | 94.85 |
| warm image median, s | 19.54 | 91.19 |
| warm image min / max, s | 19.24 / 20.18 | 80.73 / 98.71 |
| sampling, s/it (median) | 2.25 | 9.62 |
| all images, s | 210.6 | 907.0 |
| GPU busy avg, % | 90.1 | 83.9 |
| VRAM start / peak, MB | 2376.0 / 14935.0 | 2667.0 / 15797.0 |
| GTT peak, MB | 282.0 | 5145.0 |
| process RSS peak, MB | 15588.0 | 14376.0 |
| host MemAvailable min, MB | 35858.0 | 32658.0 |
| ComfyUI CPU avg / max, % of one core | 200.8 / 417.8 | 25.0 / 432.2 |
| host CPU avg, % | 17.6 | 8.2 |
| GPU edge temp start / avg / max, °C | 37.0 / 56.1 / 62.0 | 53.0 / 54.6 / 59.0 |
| GPU junction temp avg / max, °C | 79.2 / 90.0 | 69.6 / 85.0 |
| GPU mem temp avg / max, °C | 71.2 / 76.0 | 72.1 / 76.0 |
| GPU power avg / max, W | 201.4 / 270.0 | 123.6 / 287.0 |
| GPU sclk avg, MHz | 2201.2 | 2082.9 |
| GPU mclk / fclk avg, MHz | – / – | – / – |
| CPU temp max, °C | 76.0 | 76.0 |

**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0+budgetfix  

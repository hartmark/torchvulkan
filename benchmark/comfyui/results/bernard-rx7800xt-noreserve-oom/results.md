# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory`.

| | rocm | vulkan |
|---|---|---|
| status | ok | fail |
| images OK | 10 | 3 |
| device | AMD Radeon Graphics (gfx1101) | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 41.9 | 114.13 |
| warm image median, s | 22.07 | 107.06 |
| warm image min / max, s | 21.04 / 25.13 | 100.45 / 113.67 |
| sampling, s/it (median) | 2.25 | 10.75 |
| all images, s | 243.4 | 328.2 |
| GPU busy avg, % | 84.2 | 68.1 |
| VRAM start / peak, MB | 2122.0 / 14666.0 | 2124.0 / 15502.0 |
| GTT peak, MB | 207.0 | 5129.0 |
| process RSS peak, MB | 16080.0 | 15303.0 |
| host MemAvailable min, MB | 29426.0 | 26219.0 |
| ComfyUI CPU avg / max, % of one core | 200.6 / 584.1 | 40.8 / 494.4 |
| host CPU avg, % | 19.1 | 12.2 |
| GPU edge temp start / avg / max, °C | 45.0 / 57.4 / 63.0 | 52.0 / 53.8 / 60.0 |
| GPU junction temp avg / max, °C | 77.5 / 88.0 | 68.2 / 85.0 |
| GPU mem temp avg / max, °C | 73.3 / 78.0 | 72.4 / 76.0 |
| GPU power avg / max, W | 183.5 / 261.0 | 127.0 / 312.0 |
| GPU sclk avg, MHz | 2112.4 | 1712.2 |
| CPU temp max, °C | – | – |

**vulkan error:** `image 4 failed: torch.OutOfMemoryError: torchvulkan [ERROR]: Out of memory on AMD Radeon RX 7800 XT (RADV NAVI32). Tried to allocate 100663296 bytes with 7784628224 of 7763060736 bytes in use.`
**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0  

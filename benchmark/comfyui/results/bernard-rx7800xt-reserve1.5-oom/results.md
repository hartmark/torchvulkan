# torchvulkan vs ROCm in ComfyUI — Advanced Micro Devices, Inc. [AMD/ATI] Navi 32 [Radeon RX 7700 XT / 7800 XT] (rev c8)

Host: AMD Ryzen 9 5900X 12-Core Processor, 63 GB RAM, kernel 7.2.11-arch1-1, podman 6.1.4. 10 images per backend, 1024x1024, 8 steps, Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. ComfyUI args: `--disable-dynamic-vram --disable-pinned-memory --reserve-vram 1.5`.

| | rocm | vulkan |
|---|---|---|
| status | ok | fail |
| images OK | 10 | 1 |
| device | AMD Radeon Graphics (gfx1101) | AMD Radeon RX 7800 XT (RADV NAVI32) |
| first image (cold), s | 32.6 | 94.92 |
| warm image median, s | 21.04 | – |
| warm image min / max, s | 20.04 / 24.55 | – / – |
| sampling, s/it (median) | 2.25 | 5.12 |
| all images, s | 226.2 | 94.9 |
| GPU busy avg, % | 86.7 | 47.9 |
| VRAM start / peak, MB | 2129.0 / 14479.0 | 2129.0 / 15442.0 |
| GTT peak, MB | 195.0 | 4387.0 |
| process RSS peak, MB | 16646.0 | 12841.0 |
| host MemAvailable min, MB | 34100.0 | 31297.0 |
| ComfyUI CPU avg / max, % of one core | 199.3 / 386.9 | 66.1 / 332.7 |
| host CPU avg, % | 14.1 | 8.7 |
| GPU edge temp start / avg / max, °C | 45.0 / 58.1 / 63.0 | 52.0 / 52.6 / 61.0 |
| GPU junction temp avg / max, °C | 78.9 / 87.0 | 67.6 / 85.0 |
| GPU mem temp avg / max, °C | 73.9 / 78.0 | 72.9 / 78.0 |
| GPU power avg / max, W | 192.2 / 251.0 | 123.5 / 279.0 |
| GPU sclk avg, MHz | 2079.2 | 1151.5 |
| CPU temp max, °C | – | – |

**vulkan error:** `image 2 failed: torch.OutOfMemoryError: torchvulkan [ERROR]: Out of memory on AMD Radeon RX 7800 XT (RADV NAVI32). Tried to allocate 536870912 bytes with 7516192768 of 7885952614 bytes in use.`
**vulkan CPU fallbacks logged:** `Silent fallback detected for operation: aten::add.out. Set TORCHVULKAN_STRICT=1 to enable strict mode.`

rocm versions: backend=rocm, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+rocm7.1, torchvision 0.28.0+rocm7.1, triton-rocm 3.7.1  
vulkan versions: backend=vulkan, comfyui=6cbc3862d5c3ceffa9e31ea38b3fe66681bbd6ab, gguf_node=1544f8ea9cb47dc7f5436c10c0db5a5d54a9d105, python 3.14.8-1, vulkan-radeon 1:26.2.5-1, comfy-aimdo 0.6.0, comfy-kitchen 0.2.40, torch 2.13.0+cpu, torchvision 0.28.0+cpu, torchvulkan 0.2.0  

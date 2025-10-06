from PIL import Image

# 读取图像和掩码
image_path = '/home/yanzhang/dataset/customp2p/src/mirror.jpg'  # 原始图像路径
mask_path = '/home/yanzhang/dataset/customp2p/src/mirror_mask.jpg'  # 掩码路径

# 打开图像和掩码
image = Image.open(image_path).convert("RGBA")  # 转换为RGBA模式，确保包含透明通道
mask = Image.open(mask_path).convert("L")  # 转换为灰度模式，掩码应为单通道

# 应用掩码到图像
masked_image = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask).convert("RGB")

# 保存结果
output_path = 'masked_image.jpg'
masked_image.save(output_path)

print(f"Masked image saved to {output_path}")

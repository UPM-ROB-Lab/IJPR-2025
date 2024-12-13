import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import os
'''prompt'''
# 定义解析函数
def parse_prompt(prompt):
    # 按分号分割字符串
    parts = prompt.split('; ')
    
    # 确保提取的部分足够
    if len(parts) != 3:
        raise ValueError("提示信息格式不正确，应该包含三个部分")

    # 解析 Customer Need 部分
    customer_need = parts[0].strip()
    need_parts = customer_need[1:-1].split(', ')
    need_dict = {
        "Modified Position": need_parts[0].strip(),
        "Score": need_parts[1].strip()
    }

    # 解析 Functional Requirement 部分
    functional_requirement = parts[1].strip()

    # 解析 Design Evaluation and Market Trends 部分
    evaluation_and_trends = parts[2].strip().split(' + ')
    evaluation_parts = evaluation_and_trends[0][1:-1].split(', ')
    trends_parts = evaluation_and_trends[1][1:-1].split(', ')
    evaluation_dict = {
        "Modify automotive interior serial number": evaluation_parts[0].strip(),
        "Score": evaluation_parts[1].strip(),
        "Modify requirements": [trend.strip() for trend in trends_parts]
    }

    # 创建最终结果字典
    result = {
        "Customer Need": need_dict,
        "Functional Requirement": functional_requirement,
        "Design Evaluation and Market Trends": evaluation_dict
    }

    return result

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# 从txt文件中读取提示内容
with open('prompt.txt', 'r') as file:
    prompt_text = file.read().strip()

# 解析提示
parsed_result = parse_prompt(prompt_text)
# 输出结果
print(parsed_result)


scheme_name= parsed_result["Design Evaluation and Market Trends"]["Modify automotive interior serial number"]
print(scheme_name)
parsed_result = parse_prompt(prompt_text)
requirements = parsed_result["Design Evaluation and Market Trends"]["Modify requirements"]

if requirements:
    prompt_extraction = ", ".join(requirements)  # 直接用逗号分隔的名词作为 prompt
else:
    prompt_extraction = "" #如果没有提取到，prompt设置为空字符串，避免报错
print(prompt_extraction) #输出：Ergonomics, Aesthetics, Automatic transmission

'''SAM'''
# 构建图像路径
image_folder = 'D:\\CIRP2025\\experiment\\scheme'  # 图片所在的文件夹
image_path = os.path.join(image_folder, f"{scheme_name}.jpg") #使用f-string格式化字符串，添加.jpg后缀

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "D:\\CIRP2025\\segment-anything\\packages\\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
# 输出 masks 的数量
num_masks = len(masks)
print(f"生成的掩码数量：{num_masks}")

# 显示图像和掩码
plt.figure(figsize=(10, 10))
plt.imshow(image)

# 遍历所有生成的掩码并叠加显示
for mask_data in masks:
    mask = mask_data['segmentation']  # 获取实际的掩码
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) # 生成随机颜色和透明度
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)

plt.axis('off')  # 关闭坐标轴
plt.show()


# 可视化掩码的另一种更清晰的方式（显示每个掩码的边界框和分数）
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = img[m] * (1 - 0.35) + color_mask * 0.35
    ax.imshow(img)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()


predictor = SamPredictor(sam)
predictor.set_image(image)
input_boxes_data = {
    "LeftControlButton": [0, 0, 268, 923],
    "Seat": [0, 780, 1350, 920],
    "SteeringWheel": [200, 300, 580, 630],
    "CentrolConsole": [580, 450, 840, 650],
    "GearsbiftLever": [600, 630, 820, 920],
    "RightControlButton": [1150, 0, 1388, 923],
}

# 创建 torch.tensor
input_boxes = torch.tensor(list(input_boxes_data.values()), device=predictor.device)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

print(masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W | output: torch.Size([4, 1, 600, 900])

plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()

'''DM'''
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 检查是否有可用的 CUDA 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用 SD 1.5 模型
model_id = "D:\\CIRP2025\\stable-diffusion\\model\\stable-diffusion-v1-5"

# 加载 pipeline，根据设备类型设置 torch_dtype
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, use_auth_token=True)
except Exception as e:
    print(f"加载模型时发生错误: {e}")
    exit()

# 将 pipeline 移动到指定设备
pipe = pipe.to(device)


# 加载customer need修改部位图像，并调整大小
try:
    init_image = Image.open(image_path).convert("RGB").resize((600,450))
except FileNotFoundError:
    print("找不到指定的图像文件。请检查路径是否正确。")
    exit()
except Exception as e:
    print(f"打开图像文件时发生错误: {e}")
    exit()

# 设置提示词、强度和其他参数
prompt = prompt_extraction
strength = 0.7
guidance_scale = 7.5
num_inference_steps = 50
generator = torch.Generator(device=device).manual_seed(1024)  # 设置随机种子以获得可重复的结果

# 进行图像生成，添加generator
try:
    images = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images
except Exception as e:
    print(f"生成图像时发生错误: {e}")
    exit()

# 保存生成的图像
try:
    images[0].save("D:\\CIRP2025\\experiment\\scheme\\generated_image.png")
    print("图像已保存")
except Exception as e:
    print(f"保存图像时发生错误: {e}")
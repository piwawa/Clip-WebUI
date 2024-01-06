import gradio as gr
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Launch the Wav2Lip WebUI")
# 添加一个server_port的参数
parser.add_argument('--port', default=8077, type=int, help='server port')
# 解析参数
args = parser.parse_args()

models = ["openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"]

# Function to process and display images
def process_and_display(image, url, model_id, text_prompts):

    # Load the model and processor
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # Convert text prompts to list and handle image input
    text_prompts = [prompt.strip() for prompt in text_prompts.split(',')]
    images = []

    if image is not None:
        images.append(image)

    if url != "":
        urls = [url.strip() for url in url.split(',')]
        for url in urls:
            response = requests.get(url, stream=True).raw
            img = Image.open(response.content)
            images.append(img)

    inputs = processor(text=text_prompts, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    num_images = len(images)

    # Plotting
    fig, axes = plt.subplots(len(images), 2, figsize=(10, len(images) * 5))
    fig.suptitle(f'Image classification using {model_id}\n', fontsize=16)
    
    for i, img in enumerate(images):

        if num_images > 1:
            ax_img, ax_prob = axes[i, 0], axes[i, 1]
        else:
            axes = [axes]
            ax_img, ax_prob = axes[i]

        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f'Image {i+1}')
        ax_prob.barh(text_prompts, probs[i].tolist())
        ax_prob.set_xlim(0, 1)
        ax_prob.invert_yaxis()
        ax_prob.set_xlabel('Probability')
        ax_prob.set_title(f'Predicted probabilities for Image {i+1}')

    # 将图表保存为本地文件
    plt.tight_layout()
    plt.savefig('output/clip_chart.png')
    plt.close(fig)
    return 'output/clip_chart.png'

# Create Gradio interface
app = gr.Interface(
    fn=process_and_display,
    inputs=[
        gr.Image(label="上传图片（只能上传一张）"),
        gr.Textbox(label="上传图片的 URL（可不填，多个 URL 用逗号分隔）"),
        gr.Dropdown(choices=models, value=models[0], label="选择 CLIP 模型"),
        gr.Textbox(value='a cat runing in snowy filed, a boy wearing a beanie, a dog, a dog at the beach, cat, woman', label="输入文本提示（多段文本用英文逗号分隔）")
    ],
    outputs=gr.Image(type="filepath", label="Output Image"),  # 输出类型为文件
    title="CLIP Image Classification",
    description="Upload an image or enter an image URL, select a CLIP model, and enter text prompts to classify the image."
)

app.launch(debug=True, auth=("piwawa", 'breakingbad'), \
           server_name='0.0.0.0', server_port=args.port)

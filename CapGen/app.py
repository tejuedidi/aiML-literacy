import gradio as gr
import torch
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

load_dotenv()
client = OpenAI()

device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

vibe_categories = ["serene", "adventurous", "romantic", "playful", "moody", "mysterious", "elegant", "peaceful"]

def gen_img_desc(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    desc = blip_processor.decode(out[0], skip_special_tokens=True)
    return desc

def get_img_vibe(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    vibe_scores = []
    for vibe in vibe_categories:
        text_inputs = clip_processor(text=[vibe], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
        
        similarity = torch.cosine_similarity(image_features, text_features)
        vibe_scores.append((vibe, similarity.item()))
    
    vibe_scores = sorted(vibe_scores, key=lambda x: x[1], reverse=True)
    
    return vibe_scores[0][0]

# Function to generate initial 
def prompt(description, vibe):
    prompt = (
        f"You are a social media content expert. Take the following image description:\n\n '{description}'\n\n and the vibe of the image: {vibe}\n\n."
        "Now write 3 engaging, trendy captions. Make them aesthetic, creative, and include keywords and hashtags where appropriate."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()

def generated_caption(image, user_description):
    blip_description = gen_img_desc(image)
    
    if user_description.strip():
        description = f"{user_description}\n\nBLIP generated description:\n{blip_description}"
    else:
        description = blip_description
    
    vibe = get_img_vibe(image)
    captions = prompt(description, vibe)
    
    return description, vibe, captions, captions

def refined_caption(chat_history, user_input, base_caption):
    if not chat_history or chat_history[0]["role"] != "system":
        chat_history.insert(0, {
            "role": "system",
            "content": (
                f"You are a professional caption editor. The user will ask you to refine a Pinterest caption. "
                f"The original caption is:\n\n{base_caption}\n\n"
                "Use it as the base for all refinements. Never ask the user to re-provide it."
            )
        })

    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
        temperature=0.7,
        max_tokens=200,
    )

    new_caption = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": new_caption})

    return chat_history, ""

# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("# CapGen 📌")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload your pins!")
            description_input = gr.Textbox(label="Enter your own image description: (optional)", interactive=True)
            submit_btn = gr.Button("Generate")

        with gr.Column(scale=1):
            description_box = gr.Textbox(label="BLIP Generate Image Description", interactive=False)
            vibe_box = gr.Textbox(label="CLIP Generated Image Vibe", interactive=False)
            caption_box = gr.Textbox(label="Generated Captions", lines=4, interactive=True, show_copy_button=True)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=420, type="messages")
            msg = gr.Textbox(label="Refine your caption", placeholder="what changes would you like to make?")
            send_btn = gr.Button("Send")
            base_caption_state = gr.State()
        
    submit_btn.click(
        generated_caption,
        inputs=[image_input, description_input],
        outputs=[description_box, vibe_box, caption_box, base_caption_state]
    )

    send_btn.click(
        refined_caption,
        inputs=[chatbot, msg, base_caption_state],
        outputs=[chatbot, msg]
    )

if __name__ == "__main__":
    iface.launch(share=True)
